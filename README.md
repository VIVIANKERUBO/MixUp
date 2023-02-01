# Mix-Up-for-time-series

This repository aims to use MixUp as a data augmentation technique for time series classification by use of Inception Time Architecture

# Data

The data used in this project comes from the UCR/UEA archive.

# Codes

The code is divided as follows:

The [main.py](https://github.com/VIVIANKERUBO/MixUp/blob/main/main.py) python file contains the necessary code to run an experiement.
The utils folder contains the necessary functions to read the datasets
The classifiers folder contains two python files: (1) [inception.py](https://github.com/VIVIANKERUBO/MixUp/blob/main/classifiers/inception.py) contains the inception network; (2) [nne.py](https://github.com/VIVIANKERUBO/MixUp/blob/main/classifiers/nne.py) contains the code that ensembles a set of Inception networks.
In the notebooks there is a code Inception_time_clone.ipynb that can help to run the Inception Time

# Running the code on PC

You should first consider changing the following [line](https://github.com/VIVIANKERUBO/MixUp/blob/main/main.py#L218). This is the root file of everything (data and results) let's call it root_dir.

After that you should create a folder called archives inside your root_dir, which should contain the folder UCR_TS_Archive_2015. The latter will contain a folder for each dataset called dataset_name, which can be downloaded from this [website](https://www.cs.ucr.edu/~eamonn/time_series_data/).

# Results
The learning curves for 100% Breizh crops training data
using experiments without any data augmentation(no mix up), with mix up and manifold
mix up are shown below. For each experiment, the
epochs with the least validation loss is highlighted using an orange dotted perpendicular
line. From the graphs we can see that unlike experiments with no mix up and experiments
with manifold mix up, mix up curves seem to converge faster.
![image](https://user-images.githubusercontent.com/28702547/216040542-08f36e5e-8c5a-450b-9c03-ea9b90cb58d3.png)

![image](https://user-images.githubusercontent.com/28702547/216040903-3fca6907-7dee-489a-86d3-7a3336b4fc95.png)

