import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from torch.nn.modules.module import Module
from torch_geometric.utils import dense_to_sparse
    
from layer import AvgReadout, Discriminator
    
class GCNEncoder(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, act=F.relu):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, output_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.conv1(x, edge_index))
    
        return x

class SharedMLP(nn.Module):
    '''A shared MLP with two hidden layers and PReLU activation.'''
    def __init__(self, input_dim, hidden_dim, proj_dim):
        super(SharedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, proj_dim)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.prelu(self.fc1(x))
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

    def encode(self, feat, adj):
        return self.encoder(feat, adj)

    def decode(self, h, adj):
        return self.decoder(h, adj)

    def forward(self, feat, feat_a, adj, gdc_adj):
        # Original view
        h = self.encode(feat, adj)
        # Diffusion-augmented view
        h_g = self.encode(feat, gdc_adj) 
        # Feature-shuffled negative sample
        shuf_h = self.encode(feat_a, adj)  
        # Reconstructed feature
        emb = self.decode(h, adj)

        z = self.mlp(h)
        z_g = self.mlp(h_g)
        shuf_z = self.mlp(shuf_h)
        
        # Original view graph representation
        g = self.read(h, self.graph_neigh) 
        g = self.sigm(g)
        g = self.mlp(g)  

        # Diffusion-augmented view graph representation
        g_g = self.read(h_g, self.graph_diffusion)
        g_g = self.sigm(g_g)  
        g_g = self.mlp(g_g)

        ret = self.disc(g_g, z, shuf_z)  
        ret_a = self.disc(g, z_g, shuf_z) 
        
        return emb, ret, ret_a
