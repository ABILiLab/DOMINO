# DOMINOR

## Environment installation 

This code was tested with python 3.8, cuda 11.6, PyTorch 1.13.1, and torch_geometrics 2.5.3

First, install the conda environment with Python 3.8：
```
conda create -n env_name python=3.8
```
Replace 'env_name' with your own environment name.

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
conda install -c r rpy2
```

Configure the relevant environment variables:
```
export R_HOME=/home/<user_name>/anaconda3/envs/<environment_name>/lib/R
export R_LIBS_USER=/home/<user_name>/anaconda3/envs/<environment_name>/lib/R/library
```

Replace `<user_name>` and `<environment_name>` with your own username and environment name.

Install `mclust` package in R:
```
R
install.packages(“mclust”)
```

## Datasets

### CosMx_data

This dataset is from the CosMx platform and consists of 6 slices.
```
platform==CosMx
dataset_name==CosMx_data
```

### DLPFC

This dataset is from the 10x Genomics platform and consists of 12 slices.
```
platform==10x
dataset_name==DLPFC
```

### MERFISH_and_STARMAP

This dataset contains a total of 6 slices from the MERFISH platform and 1 slice from the STARMAP platform.
```
platform==MERFISH or platform==STARMAP
dataset_name==MERFISH_and_STARMAP
```

### Xenium_and_Stomics

This dataset contains a total of 5 slices from the Xenium platform and 1 slice from the Stomics platform.
```
platform==Xenium or platform==Stomics
dataset_name==Xenium_and_Stomics
```

If you need to test the new data, please make sure that the data is located in the ./data directory.

## Run the code

All code is currently launched through `python train.py`. We provide the slice Lung6 of the CosMx_data dataset for the model. Run the following code for testing:

```
python train.py
```

If you want to test in other datasets, please change the `platform`, `dataset_name` and `slice_id` parameters. such as:

```
python train.py --platform=MERFISH --dataset_name=MERFISH_and_STARMAP --slice_id=0725523_D35_m6_1_slice_3
```

Check train.py for overriding default parameters.
