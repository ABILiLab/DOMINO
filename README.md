# DOMINO

## Environment installation 

This code was tested with python 3.8, cuda 11.6, PyTorch 1.13.1, and torch_geometrics 2.5.3


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

## Run the code

We expect the user to provide such as either 'adata.h5' or 'adata.h5ad' as the necessary input. And make sure to store this input file in the "./data" directory.

At the same time, we also hope that users can provide the number of spatial domain categories for the slices to be clustered, so as to facilitate the subsequent accurate spatial domain identification.

And the final clustering results will be saved in the h5ad file located in the "./result" directory. After reading this file as an adata object, the clustering results are stored in adata.obs['domain'].

Run the following code for example:

```
python domino.py --input_file adata.h5ad --output_file domino_output.h5ad --n_clusters 7
```

Check domino.py for overriding default parameters.
