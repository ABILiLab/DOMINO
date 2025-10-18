import torch
from domino_spatial.model import GDCGraphCL
from tqdm import tqdm
from torch import nn
import contextlib

from domino_spatial.preprocess import preprocess, preprocess_adj_sparse, get_feature, add_contrastive_label, grid_downsample, optimized_construct_interaction

import warnings
warnings.filterwarnings('ignore')   


class DOMINO():
    def __init__(self, 
        adata,
        device,
        learning_rate=0.001,
        weight_decay=1e-5,
        epochs=800,
        hidden_dim=512,
        output_dim=256,
        proj_dim=128,
        a=1,
        b=1,
        is_downsample=False,
        grid_size=100
        ):
        '''

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        device : string, optional
            Using GPU or CPU? The default is 'cuda'.
        learning_rate : float, optional
            Learning rate for the Adam optimizer. The default is 0.001.
        weight_decay : float, optional
            Weight decay (L2 penalty) for regularization. The default is 1e-5.
        epochs : int, optional
            Number of training epochs. The default is 800.
        hidden_dim : int, optional
            Dimension of hidden layers in the GNN model. The default is 512.
        output_dim : int, optional
            Dimension of the output embedding. The default is 256.
        proj_dim : int, optional
            Dimension of the projection head for the mlp layers. The default is 128.
        a : int, optional
            Weight coefficient for feature reconstruction loss (loss_feat). The default is 1.
        b : int, optional
            Weight coefficient for graph contrastive learning loss (loss_sl_1 + loss_sl_2). The default is 1.
        is_downsample : bool, optional
            Whether to use downsampling or not. The default is False.
        grid_size : int, optional
            Grid size for downsampling. The default is 100.

        Returns
        -------
        adata : anndata
            AnnData object of spatial data with added 'emb' key in obsm.
        '''

        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.proj_dim=proj_dim
        self.a=a
        self.b=b
        self.is_downsample=is_downsample
        self.grid_size = grid_size

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

        # Autocast context: bf16 on CUDA, no-op on CPU
        self.use_amp = (device.type == "cuda" and torch.cuda.is_bf16_supported())
        self.autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.use_amp else contextlib.nullcontext()
        
        self.adata = preprocess(self.adata)

        if self.is_downsample:
            self.adata = grid_downsample(self.adata, (self.grid_size, self.grid_size))

        print("Constructing interaction matrix...")
        self.adata = optimized_construct_interaction(self.adata)

        print("Constructing interaction matrix Done!")

        add_contrastive_label(self.adata)
        get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.input_dim = self.features.shape[1]
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        # Symmetric adjacency for neighborhood aggregation
        self.adj = self.adata.obsm['adj']
        self.adj_diffusion = self.adata.obsm['adj_diffusion']
        
        # Adjacency matrix, used as a mask for pooling operations
        print("generating graphs..")

        self.graph_neigh = self._add_self_loops_sparse(self._to_sparse_coo(self.adata.obsm['graph_neigh'])).to(self.device)
        self.graph_diff  = self._add_self_loops_sparse(self._to_sparse_coo(self.adata.obsm['graph_diffusion'])).to(self.device)

        self.adj = preprocess_adj_sparse(self.adj).to(self.device)

        self.adj_diffusion = preprocess_adj_sparse(self.adj_diffusion).to(self.device)

    # function for stop making dense matrix when generating graphs
    def _to_sparse_coo(self, A):
        import scipy.sparse as sp
        if hasattr(A, "tocoo"):             # scipy.sparse
            A = A.tocoo()
            import numpy as np
            idx = torch.from_numpy(np.vstack([A.row, A.col]).astype("int64"))
            val = torch.from_numpy(A.data.astype("float32"))
            return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()
        elif isinstance(A, np.ndarray):
            # if dense，then make it sparse
            ii, jj = np.nonzero(A)
            val = A[ii, jj].astype("float32")
            idx = torch.from_numpy(np.vstack([ii, jj]).astype("int64"))
            val = torch.from_numpy(val)
            return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()
        elif isinstance(A, torch.Tensor) and A.is_sparse:
            return A.coalesce()
        else:
            raise TypeError(f"Unsupported adjacency type: {type(A)}")

    def _add_self_loops_sparse(self, A_sp):
        n = A_sp.shape[0]
        idx = torch.arange(n, device=A_sp.device)
        I = torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(n, device=A_sp.device, dtype=A_sp.dtype), (n, n))
        return (A_sp + I).coalesce()

    def mse_chunked(self, target, pred, chunk=200_000):
        """
        Mean((target - pred)^2) without materializing the full N×F diff.
        Exactly matches F.mse_loss(..., reduction='mean').
        Accumulates in float32 for stability (esp. under AMP/bfloat16).
        """
        assert target.shape == pred.shape
        n = target.shape[0]
        total = target.numel()

        # accumulate in fp32, regardless of AMP autocast dtype
        acc = pred.new_zeros((), dtype=torch.float32)

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            diff = (pred[start:stop] - target[start:stop]).to(torch.float32)
            acc = acc + (diff * diff).sum()

        return acc / total 

    def train(self):
        # Initialize the model 
        print("Initializing model..")
        self.model = GDCGraphCL(self.input_dim, self.hidden_dim, self.output_dim, self.proj_dim, self.graph_neigh, self.graph_diff).to(self.device)

        # Binary cross-entropy loss for contrastive learning
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay = self.weight_decay)

        print('Begin to train ST data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)    
            
            with self.autocast_ctx:
                self.emb, ret, ret_a = self.model(self.features, self.adj, self.adj_diffusion)
                # Calculate loss 
                self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)  # Graph contrastive loss for original view
                self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)  # Graph contrastive loss for augmented view
                self.loss_feat = self.mse_chunked(self.features, self.emb)  # Feature reconstruction loss
                # Total loss
                loss =  self.a*self.loss_feat + self.b*(self.loss_sl_1 + self.loss_sl_2)

            if epoch % 100 == 0:
                print(
                    'Epoch {:0>3d} | Loss:[{:.4f}], loss_feat:[{:.4f}], loss_sl_1:[{:.4f}], loss_sl_2:[{:.4f}]'.format(
                        epoch, loss.item(), self.loss_feat.item(), self.loss_sl_1.item(), self.loss_sl_2.item()))
            
            loss.backward() 
            self.optimizer.step()
        
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            self.emb_rec = self.model(self.features, self.adj, self.adj_diffusion)[0].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
                
            return self.adata
