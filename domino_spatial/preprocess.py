# Implementation of functions: get_feature, add_contrastive_label and preprocess_adj_sparse is referencing from GraphST: https://doi.org/10.1038/s41467-023-36796-3

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from scipy import sparse
from scipy.sparse import eye, diags
from scipy.sparse import coo_matrix

def _arnoldi_iteration(T, alpha, max_iter, tol, eps=1e-6):
    """Arnoldi iteration method for accelerating convergence."""
    N = T.shape[0]
    S = alpha * sp.eye(N, dtype=np.float32, format='csr')
    X = (1 - alpha) * T
    
    for _ in range(max_iter):
        delta = alpha * X
        S += delta

        # Immediately filter out the tiny values after each iteration.
        delta.data[delta.data < eps] = 0
        delta.eliminate_zeros()
        
        # Check for convergence.
        if delta.nnz == 0 or np.abs(delta).sum() < tol * N:
            break
            
        # Optimize using matrix-vector product.
        X = T @ X  
        
    return S

def optimized_balanced_gdc(A, alpha=0.15, eps=1e-6, n_iter=35, tol=1e-5, k=50):
    """optimized balance of PPR diffusion."""
    N = A.shape[0]
    
    # Convert to CSR format to achieve the best matrix multiplication performance.
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    
    # Add self-loops and normalize.
    A_loop = A + eye(N, dtype=np.float32, format='csr')
    D_loop_vec = A_loop.sum(axis=1).A.ravel().astype(np.float32)
    D_loop_invsqrt = diags(1 / np.sqrt(D_loop_vec + 1e-12), dtype=np.float32, format='csr')
    # T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    T_sym = D_loop_invsqrt.dot(A_loop).dot(D_loop_invsqrt)

    # Perform Arnoldi iteration.
    S = _arnoldi_iteration(T_sym, alpha, n_iter, tol)
    
    # Efficient sparsification and normalization.
    S.data[S.data < eps] = 0
    S.eliminate_zeros()

    # Enforce sparsity by keeping top-k edges per node.
    if k is not None:
        rows, cols = S.nonzero()
        data = S.data
        coo_dict = {}
        for i, j, v in zip(rows, cols, data):
            if i not in coo_dict:
                coo_dict[i] = []
            coo_dict[i].append((j, v))
        
        # Keep top-k edges per node.
        new_rows, new_cols, new_data = [], [], []
        for i in coo_dict:
            edges = sorted(coo_dict[i], key=lambda x: -x[1])[:k]
            for j, v in edges:
                new_rows.append(i)
                new_cols.append(j)
                new_data.append(v)
        
        S = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=(N, N)).tocsr()
    
    # Use a more stable normalization method.
    D_tilde_vec = S.sum(axis=1).A.ravel()
    D_tilde_inv = diags(1 / np.maximum(D_tilde_vec, 1e-12), dtype=np.float32, format='csr')
    # return S @ D_tilde_inv
    return S.dot(D_tilde_inv)

def build_sparse_adjacency(position, n_neighbors=5):
    """Efficient construction of sparse adjacency matrix."""
    n_spot = position.shape[0]
    
    # Using BallTree to find neighbors.
    nn = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree')
    nn.fit(position)
    distances, indices = nn.kneighbors(position)
    
    # Construct the original neighbor matrix (asymmetric).
    rows = np.repeat(np.arange(n_spot), n_neighbors)
    cols = indices[:, 1:].flatten()  
    data = np.ones_like(rows, dtype=np.int8)
    
    # Create the original neighbor matrix (in COO format).
    adj_original = coo_matrix((data, (rows, cols)), shape=(n_spot, n_spot))
    adj_original = adj_original.tocsr()  
    
    # Obtain the symmetric adjacency matrix by adding the transpose.
    adj_sym = adj_original + adj_original.T
    adj_sym.data = np.ones_like(adj_sym.data)  
    
    return adj_sym, adj_original

def optimized_construct_interaction(adata, n_neighbors=5, alpha=0.15, eps=1e-6, n_iter='auto', tol=1e-5):
    """Optimized function for constructing spatial interaction graphs with diffusion processing."""
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    
    # Automatically determine iteration count based on number of spots.
    if n_iter == 'auto':
        n_iter = 40 if n_spot < 2000 else 30 if n_spot < 5000 else 25
    
    # Build sparse adjacency matrix from spatial coordinates.
    adj, interaction = build_sparse_adjacency(position, n_neighbors=n_neighbors)

    # Original neighbor matrix
    adata.obsm['graph_neigh'] = interaction
    # Symmetric adjacency matrix
    adata.obsm['adj'] = adj

    # Graph diffusion processing.
    graph_diffusion = optimized_balanced_gdc(
        adj, 
        alpha=alpha,
        eps=eps,
        n_iter=n_iter,
        tol=tol
    )
    # Neighbor matrix after diffusion processing.
    adata.obsm['graph_diffusion'] = graph_diffusion

    # Symmetric adjacency matrix after diffusion processing.
    adj_diffusion = 0.5 * (graph_diffusion + graph_diffusion.T)
    adj_diffusion.data = (adj_diffusion.data > 0).astype(np.int8)
    adata.obsm['adj_diffusion'] = adj_diffusion
    
    return adata

def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    return adata

def grid_downsample(
    adata, 
    grid_size=(100, 100), 
    downsample_by='median',  
    keep_sparse=True         
):
    """
    Parameters
    ----------
    adata : 
        AnnData object of spatial data.
    grid_size : 
        Grid division size (x_bins, y_bins).
    downsample_by : 
        Sampling method ('random'/'median').
    keep_sparse : 
        Maintain the sparse matrix format or not.
        
    Returns
    -------
    AnnData object after downsampling.
    
    """
    coords = adata.obsm['spatial']
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    # Generate grid boundaries.
    eps_x, eps_y = (max_x-min_x)*1e-5, (max_y-min_y)*1e-5
    x_bins = np.linspace(min_x, max_x+eps_x, grid_size[0]+1)
    y_bins = np.linspace(min_y, max_y+eps_y, grid_size[1]+1)
    
    # Calculate grid coordinates (vectorization).
    coords = np.asarray(coords)
    x_idx = np.digitize(coords[:,0], x_bins) - 1
    y_idx = np.digitize(coords[:,1], y_bins) - 1
    grid_ids = x_idx * grid_size[1] + y_idx
    
    # Group processing.
    unique_ids = pd.unique(grid_ids)
    sampled_indices = []
    grid_assignments = np.full(len(coords), -1, dtype=int)
    
    for i, cell_id in enumerate(unique_ids):
        cell_mask = (grid_ids == cell_id)
        cell_indices = np.where(cell_mask)[0]
        
        if len(cell_indices) == 0:
            continue
            
        # Select representative points based on the sampling strategy.
        if downsample_by == 'random':
            selected_idx = np.random.choice(cell_indices)
        elif downsample_by == 'median':
            # Calculate the median coordinates of all points within the grid.
            median_coord = np.median(coords[cell_indices], axis=0)
            # Select the point that is closest to the median distance.
            distances = np.linalg.norm(coords[cell_indices] - median_coord, axis=1)
            selected_idx = cell_indices[np.argmin(distances)]
        else:
            raise ValueError("downsample_by must be 'random' or 'median'")
            
        sampled_indices.append(selected_idx)
        grid_assignments[cell_mask] = i
    
    # Constructing downsampled data.
    adata_downsampled = adata[sampled_indices].copy()
    
    # Processing matrix format.
    if keep_sparse and sparse.issparse(adata.X):
        adata_downsampled.X = adata.X[sampled_indices].tocsr()
    
    # Preserve the original spatial coordinates and grid assignments.
    adata_downsampled.obs['grid_id'] = grid_ids[sampled_indices]
    adata_downsampled.uns['grid_assignments'] = grid_assignments
    adata_downsampled.obsm['spatial'] = coords[sampled_indices]
    
    return adata_downsampled

def get_feature(adata):
    '''Extracts and processes feature matrix from AnnData object.'''
    adata_Vars = adata
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    adata.obsm['feat'] = feat
    
def add_contrastive_label(adata):
    '''Adds contrastive label to AnnData object.'''
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def preprocess_adj_sparse(adj):
    """Convert scipy sparse matrix to torch.sparse.FloatTensor."""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    
    indices = torch.from_numpy(np.vstack([adj_normalized.row, adj_normalized.col])).long()
    values = torch.from_numpy(adj_normalized.data).float()
    shape = torch.Size(adj_normalized.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

