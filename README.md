# Mix-Up-for-time-series

This repository aims to use MixUp as a data augmentation technique for time series classification by use of Inception Time Architecture

# Data

The data used in this project comes from the UCR/UEA archive.

# Codes

The code is divided as follows:

The [main.py](#../main.py) python file contains the necessary code to run an experiement.
The utils folder contains the necessary functions to read the datasets
The classifiers folder contains two python files: (1) inception.py contains the inception network; (2) nne.py contains the code that ensembles a set of Inception networks.
In the notebooks there is a code Inception_time_clone.ipynb that can help to run the Inception Time

# Running the code on PC

You should first consider changing the following line. This is the root file of everything (data and results) let's call it root_dir.

After that you should create a folder called archives inside your root_dir, which should contain the folder UCR_TS_Archive_2015. The latter will contain a folder for each dataset called dataset_name, which can be downloaded from this website.

The names of the datasets are present here. You can comment this line to run the experiments on all datasets.

