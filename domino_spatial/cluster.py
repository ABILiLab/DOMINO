# Implementation of functions: mclust_R and search_res is referencing from GraphST: https://doi.org/10.1038/s41467-023-36796-3

import numpy as np
import pandas as pd
import scanpy as sc
import ot
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import ot.backend as otb

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, radius=50, n_clusters=7, method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """
    Spatial clustering based the learned representation.    

    Parameters
    ----------
    adata : 
        AnnData object of scanpy package.
    n_clusters : 
        The number of clusters. The default is 7.
    radius : 
        The number of neighbors considered during refinement. The default is 50.
    key : 
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : 
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : 
        The start value for searching. The default is 0.1.
    end : 
        The end value for searching. The default is 3.0.
    increment : 
        The step size to increase. The default is 0.01.   
    refinement : 
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    pca = PCA(n_components=20, random_state=42) 
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding
  
    if method == 'mclust':
       adata = mclust_R(adata, num_cluster = n_clusters, modelNames='EEE', used_obsm='emb_pca', random_seed=2020)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
    elif method == 'kmeans':
        # MiniBatchKMeans suitable for large datasets
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            n_init=10,
            init='k-means++',
            max_iter=300,
            tol=1e-3,
            random_state=42
        )
        adata.obs['domain'] = kmeans.fit_predict(adata.obsm['emb_pca']).astype(str)

    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type
        
# using blocking to reduce memory usage
def refine_label(adata, radius=50, key='label', row_block=4096):
    '''Based on the spatial position information of cells, local smoothing and optimization are performed on the initial clustering results.'''
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    print("refining domain..")

    #calculate distance
    position = adata.obsm['spatial']
    if hasattr(position, "toarray"):          
        position = position.toarray()
    elif hasattr(position, "to_numpy"):       
        position = position.to_numpy()
    elif "torch" in type(position).__module__:  
        position = position.detach().cpu().numpy()

    position = np.asarray(position, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] < 2:
        raise ValueError(f"spatial must be (n,>=2), got {position.shape}")
    position = np.ascontiguousarray(position[:, :2]) 

    print("running optimal transport...")
    # use blockwise computation in optimal transport
    # calculating distances only for a subset of rows against 
    # the whole matrix at a time, and keep only the
    # top-k neighbor indices per row, without storing 
    # the full distance matrix in memory.   
    n_cell = position.shape[0]
    if n_cell <= 1 or n_neigh < 1:
        return [str(x) for x in old_type]
    n_neigh = min(n_neigh, n_cell - 1)
    
    cats = pd.Categorical(old_type)
    codes = cats.codes   
    n_classes = len(cats.categories)

    for start in range(0, n_cell, row_block):
        stop = min(start + row_block, n_cell)
        block = position[start:stop]

        D = ot.dist(block, position, metric='euclidean')

        part = np.argpartition(D, kth=n_neigh, axis=1)[:, :n_neigh+1]
        rows = np.arange(part.shape[0])[:, None]
        part = part[rows, np.argsort(D[rows, part], axis=1)]

        for i in range(stop - start):
            idx = part[i]
            self_col = start + i

            if idx[0] == self_col:
                idx = idx[1:n_neigh+1]
            else:
                idx = idx[:n_neigh]
                idx = idx[idx != self_col]

            neigh_codes = codes[idx]
            neigh_codes = neigh_codes[neigh_codes >= 0]
            if neigh_codes.size == 0:
                new_label_code = codes[start + i]
            else:
                cnt = np.bincount(neigh_codes, minlength=n_classes)
                winners = np.flatnonzero(cnt == cnt.max())
                new_label_code = (codes[start + i]
                                   if codes[start + i] in winners
                                   else winners[0])

            new_type.append(cats.categories[new_label_code] if new_label_code >= 0
                            else str(old_type[start + i]))
        del D, part
        
    new_type = [str(i) for i in list(new_type)]

    print("finished refining...")
    return new_type

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : 
        AnnData object of spatial data.
    n_clusters : 
        Targetting number of clusters.
    method : 
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : 
        The indicated representation for clustering.
    start : 
        The start value for searching.
    end : 
        The end value for searching.
    increment : 
        The step size to increase.
        
    Returns
    -------
    res : 
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    
       
