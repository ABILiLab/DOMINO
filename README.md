# Identifying distinct spatial domains from clear cell and endometrioid ovarian carcinoma using DOMINO

![Model architecture of DOMINO](./DOMINO.png)

## Overview

DOMINO is built based on a self-supervised multi-view graph contrastive learning framework. It is designed to integrate spatial coordinates and gene expression information for robust identification of tissue domains from spatial transcriptomics data. DOMINO employs graph neural networks (GNNs) as base encoder, constructing a multi-view graph contrastive learning framework using the original graph and the diffusion graph to learn spot representations in the ST data. After representation learning, the learned low-dimensional embeddings can then be used to identify spatial domains, which can further be used in different downstream analyses, including cell type composition analysis, differential expression, and inference of cell–cell communication.

## Requirements

To facilitate user access to our DOMINO model, we provide the Python package: domino-spatial.
Before using the domino-spatial package, you must install the required environment following the steps below.

The package is developed based on the python 3.8, cuda 11.6, PyTorch 1.13.1, and torch_geometrics 2.5.3

Install packages listed on a pip file:
```
pip install -r requirement.txt
```

Install the corresponding versions of pytorch and torch_geometrics:
```
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
pip install torch-geometric==2.5.3
```

Install `rpy2` package:
```
conda install -c conda-forge r-base=4.1.0
conda install -c conda-forge rpy2
```

Configure the relevant environment variables:
```
export R_HOME=/home/<user_name>/anaconda3/envs/<environment_name>/lib/R
export R_LIBS_USER=/home/<user_name>/anaconda3/envs/<environment_name>/lib/R/library
```

Replace `<user_name>` and `<environment_name>` with your own username and environment name.

Install `mclust` package:
```
conda install -c conda-forge r-mclust
```

## Tutorial

For the step-by-step tutorial, please refer to: https://domino-tutorials.readthedocs.io/en/latest/

## Run the test code

Of course, if you wish to run the test.py file provided in the test directory, please still follow the Installation section of the tutorial to install the domino-spatial package, and then:

We expect the user to provide such as 'adata.h5ad' as the necessary input. And make sure to store this input file in the "./data" directory.

At the same time, we also hope that users can provide the number of spatial domain categories for the slices to be clustered, so as to facilitate the subsequent accurate spatial domain identification.

And the final clustering results will be saved in the h5ad file located in the "./result" directory. After reading this file as an adata object, the clustering results are stored in adata.obs['domain'].

Run the following code for example:

```
python test.py --input_file adata.h5ad --output_file domino_output.h5ad --n_clusters 7
```

Check test.py for overriding default parameters.
