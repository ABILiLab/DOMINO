import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
    
from domino_spatial.layer import AvgReadout, Discriminator

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, act=F.relu):
        super(GCNEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj_sparse):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        # Sparse multiplication
        if x.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                x = torch.sparse.mm(adj_sparse.float(), x.float())
            x = x.to(x.dtype)  # cast back (bf16/fp16-safe)
        else:
            x = torch.sparse.mm(adj_sparse, x)
        return self.act(x)

class SharedMLP(nn.Module):
    '''A shared MLP with two hidden layers and PReLU activation.'''
    def __init__(self, input_dim, hidden_dim, proj_dim):
        super(SharedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, proj_dim)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        orig_dtype = x.dtype
        x = self.fc1(x)

        if x.is_cuda:
            # PReLU has no bf16/fp16 CUDA kernel: disable autocast and upcast to fp32
            with torch.cuda.amp.autocast(enabled=False):
                x32 = torch.nn.functional.prelu(x.float(), self.prelu.weight.float())
            x = x32.to(orig_dtype)  
        else:
            # CPU path (fp32)
            x = torch.nn.functional.prelu(x, self.prelu.weight)
        x = self.fc3(x)
        return x


    
class GDCGraphCL(Module):
    '''
    A multi-view graph contrastive learning framework based on graph diffusion
    
    Parameters
    ----------
    input_dim : 
        Input feature dimension.
    hidden_dim : 
        Hidden layer dimension.
    output_dim : 
        Output embedding dimension.
    proj_dim : 
        Projection head dimension.
    graph_neigh : 
        Nearest neighbor adjacency matrix.
    graph_diffusion : 
        Diffusion-processed adjacency matrix.
    dropout : 
        Dropout rate.
    act : 
        Activation function.

    Returns
    -------
    emb: 
        Reconstructed feature.
    ret: 
        Discriminator consistency score between the original view node representation and the graph diffusion view graph representation.
    ret_a: 
        Discriminator consistency score between the graph diffusion view node representation and the original view graph representation.
    
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, proj_dim, graph_neigh, graph_diffusion, dropout=0.5, act=F.relu):
        super(GDCGraphCL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.proj_dim = proj_dim
        self.graph_neigh = graph_neigh
        self.graph_diffusion = graph_diffusion
        self.dropout = dropout
        self.act = act
        
        self.encoder = GCNEncoder(input_dim, hidden_dim, output_dim, dropout, act) 
        self.decoder = GCNEncoder(output_dim, hidden_dim, input_dim, dropout, act)
        self.mlp = SharedMLP(output_dim, hidden_dim, proj_dim)  

        self.disc = Discriminator(proj_dim)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
    
    # when reading sparse graphs, using this to avoid dense
    def _sparse_avg(self, H, A_sp):
        deg = torch.sparse.sum(A_sp, dim=1).to_dense().clamp_min(1.0)
        if H.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                out = torch.sparse.mm(A_sp.float(), H.float())
            out = out.to(H.dtype)
            deg = deg.to(H.dtype)
        else:
            out = torch.sparse.mm(A_sp, H)
        return out / deg.unsqueeze(1)   

    def encode(self, feat, adj):
        return self.encoder(feat, adj)

    def decode(self, h, adj):
        return self.decoder(h, adj)

    def forward(self, feat, adj, gdc_adj):
        # Original view
        h = self.encode(feat, adj)
        # Diffusion-augmented view
        h_g = self.encode(feat, gdc_adj) 
        # Feature-shuffled negative sample
        perm = torch.randperm(feat.size(0), device=feat.device)
        shuf_h = self.encode(feat[perm], adj)
        # Reconstructed feature
        emb = self.decode(h, adj)

        z = self.mlp(h)
        z_g = self.mlp(h_g)
        shuf_z = self.mlp(shuf_h)
        
        # Original view graph representation
        g = self._sparse_avg(h, self.graph_neigh) 
        g = self.sigm(g)
        g = self.mlp(g)  

        # Diffusion-augmented view graph representation
        g_g = self._sparse_avg(h_g, self.graph_diffusion)
        g_g = self.sigm(g_g)  
        g_g = self.mlp(g_g)

        ret = self.disc(g_g, z, shuf_z)  
        ret_a = self.disc(g, z_g, shuf_z) 
        
        return emb, ret, ret_a
    


