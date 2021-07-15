from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import random
import sklearn

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
import operator
import utils

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import GROUPED_UNIVARIATE_DATASET_NAMES as GROUPED_DATASET_NAMES
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder


def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def readucr(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def readsits(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, -1]
    X = data[:, :-1]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}

    file_name = root_dir + '/' + dataset_name + '/' + dataset_name
    x_train, y_train = readucr(file_name + '_TRAIN')
    x_test, y_test = readucr(file_name + '_TEST')
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())

    return datasets_dict


def read_all_datasets(root_dir, archive_name, id_partition=-1):
    datasets_dict = {}

    dataset_names_to_sort = []

    if archive_name == 'TSC':
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = root_dir + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
            #dataset_names_to_sort.append((dataset_name, len(x_train)))

        #dataset_names_to_sort.sort(key=operator.itemgetter(1))
        #for i in range(len(DATASET_NAMES)):
        #    DATASET_NAMES[i] = dataset_names_to_sort[i][0]
    elif archive_name == "GROUPED_TSC":
        print(id_partition)
        if id_partition<0 or id_partition> len(GROUPED_DATASET_NAMES):
          print("ERROR: invalid id_partition - " + str(id_partition) + "<0 or >"+ str(len(GROUPED_DATASET_NAMES)) + ".")
        for dataset_name in GROUPED_DATASET_NAMES[id_partition]:
            root_dir_dataset = root_dir + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    elif archive_name == 'InlineSkateXPs':

        for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
            root_dir_dataset = root_dir + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'SITS':
        return read_sits_xps(root_dir)
    else:
        print('error in archive name')
        exit()

    return datasets_dict


def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test

def metrics(y_true, y_pred):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")
    
    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
         )


