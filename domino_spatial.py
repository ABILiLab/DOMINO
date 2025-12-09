import torch
from tqdm import tqdm
from torch import nn
import scanpy as sc
import os
import argparse
import contextlib

from domino_spatial.preprocess import preprocess, preprocess_adj_sparse, get_feature, add_contrastive_label, grid_downsample, optimized_construct_interaction
from domino_spatial.model import GDCGraphCL
from domino_spatial.cluster import clustering

import warnings
warnings.filterwarnings('ignore')   

# function for stop making dense matrix when generating graphs
def _to_sparse_coo(A):
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

def _add_self_loops_sparse(A_sp):
    n = A_sp.shape[0]
    idx = torch.arange(n, device=A_sp.device)
    I = torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(n, device=A_sp.device, dtype=A_sp.dtype), (n, n))
    return (A_sp + I).coalesce()

def mse_chunked(target, pred, chunk=200_000):
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


def train_model(adata, device):
    
    # Hyperparameter settings
    print("setting parameters..")
    learning_rate=args.lr
    weight_decay=args.weight_decay
    epochs=args.epochs
    hidden_dim=args.hidden_dim
    output_dim=args.output_dim
    proj_dim=args.proj_dim
    alpha = args.a
    beta = args.b

    features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
    input_dim = features.shape[1]
    #features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device)
    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)
    # Symmetric adjacency for neighborhood aggregation
    adj = adata.obsm['adj']
    adj_diffusion = adata.obsm['adj_diffusion']
    
    # Adjacency matrix, used as a mask for pooling operations
    print("generating graphs..")

    graph_neigh = _add_self_loops_sparse(_to_sparse_coo(adata.obsm['graph_neigh'])).to(device)
    graph_diff  = _add_self_loops_sparse(_to_sparse_coo(adata.obsm['graph_diffusion'])).to(device)

    adj = preprocess_adj_sparse(adj)
    adj = adj.to(device)

    adj_diffusion = preprocess_adj_sparse(adj_diffusion)
    adj_diffusion = adj_diffusion.to(device)

    # Initialize the model 
    print("Initializing model..")
    model = GDCGraphCL(input_dim, hidden_dim, output_dim, proj_dim, graph_neigh, graph_diff).to(device)

    # Binary cross-entropy loss for contrastive learning
    loss_CSL = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay = weight_decay)

    print('Begin to train ST data...')
    model.train()
    
    for epoch in tqdm(range(epochs)): 
        model.train()
        optimizer.zero_grad(set_to_none=True)    
        
        with autocast_ctx:
            emb, ret, ret_a = model(features, adj, adj_diffusion)
            # Calculate loss 
            loss_sl_1 = loss_CSL(ret, label_CSL)  # Graph contrastive loss for original view
            loss_sl_2 = loss_CSL(ret_a, label_CSL)  # Graph contrastive loss for augmented view
            loss_feat = mse_chunked(features, emb)  # Feature reconstruction loss
            # Total loss
            loss =  alpha*loss_feat + beta*(loss_sl_1 + loss_sl_2)

        if epoch % 100 == 0:
            print(
                'Epoch {:0>3d} | Loss:[{:.4f}], loss_feat:[{:.4f}], loss_sl_1:[{:.4f}], loss_sl_2:[{:.4f}]'.format(
                    epoch, loss.item(), loss_feat.item(), loss_sl_1.item(), loss_sl_2.item()))
        
        loss.backward() 
        optimizer.step()
    
    print("Optimization finished for ST data!")
    
    with torch.no_grad():
        model.eval()
        emb_rec = model(features, adj, adj_diffusion)[0].detach().cpu().numpy()
        adata.obsm['emb'] = emb_rec
            
        return adata


if __name__ == '__main__':
    ### param setting ###

    parser = argparse.ArgumentParser()
    # input  and output setting
    parser.add_argument('--input_file', type=str, default='adata.h5ad', help="The filename of the necessary input file provided by the user")
    parser.add_argument('--output_file', type=str, default='domino_output.h5ad', help="The output h5ad filename for saving the final clustering results")

    # data preprocessing setting
    parser.add_argument('--is_downsample', action="store_true", help="Specify to use downsampling")
    parser.add_argument('--grid_size', type=int, default=100, help="Grid division size")
    parser.add_argument('--downsample_by', type=str, default='median', help="Sampling method: divided into 'random' and 'median'")
    parser.add_argument('--keep_sparse', type=bool, default=True, help="Maintain the sparse matrix format or not: True or False")

    # interaction matrix construction setting
    parser.add_argument('--n_neighbors', type=int, default=5, help="Number of nearest neighbors to consider when constructing the spatial adjacency graph")

    # data augmentation setting
    parser.add_argument('--alpha', type=float, default=0.1, help="Teleport probability (restart probability) for random walk diffusion, controls the balance between local and global structure")
    parser.add_argument('--n_iter', type=str, default='auto', help="Maximum number of iterations for Arnoldi iteration ('auto' determines based on spot count: 40 for <2000 spots, 30 for <5000, 25 otherwise)")
    parser.add_argument('--eps', type=float, default=1e-6, help="Threshold for sparsification-values below this are set to zero in the diffusion matrix")
    parser.add_argument('--tol', type=float, default=1e-5, help="Tolerance threshold for convergence check in Arnoldi iteration (relative to matrix size)")
    parser.add_argument('--k', type=int, default=50, help="Limit the maximum number of neighbors retained for each node")

    # training param setting
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the Adam optimizer")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay (L2 penalty) for regularization")
    parser.add_argument('--epochs', type=int, default=800, help="Number of training epochs")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Dimension of hidden layers in the GNN model")
    parser.add_argument('--output_dim', type=int, default=256, help="Dimension of the output embedding")
    parser.add_argument('--proj_dim', type=int, default=128, help="Dimension of the projection head for the mlp layers")
    parser.add_argument('--a', type=int, default=1, help="Weight coefficient for feature reconstruction loss (loss_feat)")
    parser.add_argument('--b', type=int, default=1, help="Weight coefficient for graph contrastive learning loss (loss_sl_1 + loss_sl_2)")

    # clustering param setting
    parser.add_argument('--n_clusters', type=int, default=8, help="The number of spatial domain categories for which the slices need to be clustered")
    parser.add_argument('--cluster_method', type=str, default='mclust', help="The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'")
    parser.add_argument('--radius', type=int, default=50, help="The number of neighbors considered during refinement")
    parser.add_argument('--start', type=float, default=0.1, help="The start value for searching corresponding resolution while cluster method is 'leiden' or 'louvain'")
    parser.add_argument('--end', type=float, default=2.0, help="The end value for searching corresponding resolution while cluster method is 'leiden' or 'louvain'")
    parser.add_argument('--increment', type=float, default=0.001, help="The step size to increase while searching corresponding resolution")
    parser.add_argument('--refinement', type=bool, default=True, help="Refine the predicted labels or not")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    use_amp = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    # Data preprocessing
    data_root = './data/'
    input_file = os.path.join(data_root, args.input_file)

    results_directory = './results/'
    os.makedirs(results_directory, exist_ok=True)

    adata = sc.read_h5ad(input_file)
    adata.var_names_make_unique()

    adata = preprocess(adata)

    slice_id = os.path.splitext(args.input_file)[0]

    print('Processing slice:', slice_id)
    print(f"original size: {adata.n_obs}x{adata.n_vars}")
    if args.is_downsample:
        adata = grid_downsample(adata, grid_size=(args.grid_size, args.grid_size), downsample_by=args.downsample_by, keep_sparse=args.keep_sparse)
    print(f"downsampled size: {adata.n_obs}x{adata.n_vars}")

    print("Constructing interaction matrix...")
    adata = optimized_construct_interaction(adata, n_neighbors=args.n_neighbors, alpha=args.alpha, eps=args.eps, n_iter=args.n_iter, tol=args.tol)

    add_contrastive_label(adata)
    get_feature(adata)

    print("Training model...")

    adata = train_model(adata, device)

    print("Clustering...")

    clustering(adata, radius=args.radius, n_clusters=args.n_clusters, method=args.cluster_method, start=args.start, end=args.end, increment=args.increment, refinement=args.refinement)

    adata.obs['domino'] = adata.obs['domain']

    output_path = os.path.join(results_directory, args.output_file)
    adata.write_h5ad(output_path) 

    print("Done!")






