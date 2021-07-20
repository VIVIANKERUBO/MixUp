from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES as partition_id
from utils.utils import read_all_datasets
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import metrics
import os
import shutil
import argparse

import utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import classifiers 
from classifiers import inception
from classifiers import inceptionmanifold
from classifiers import nne
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tslearn.metrics import dtw, dtw_path


def prepare_data(datasets_dict, dataset_name, batch_size=64):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        x_train = np.array(x_train)
        x_test = np.array(x_test)
    
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train) #one-hot encoded
    y_test = torch.from_numpy(y_test) # one-hot encoding version of y_true
    y_true = torch.from_numpy(y_true)

    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_true, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers= 0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers= 0)
   
    return train_dataloader, test_dataloader, nb_classes, y_true, enc
    

def mixup_data(x, y, device, alpha=0.4, apply_dtw=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    if apply_dtw:
      mixed_x = torch.zeros(x.shape)
      for b in range(batch_size):
        optimal_path, dtw_score = dtw_path(x[b,:,0].cpu(),x[index[b],:,0].cpu())
        p,q = map(list, zip(*optimal_path))
        tmp_mixed_x = lam * x[b,p,0] + (1 - lam) * x[index[b],q,0]
        mixed_x[b,:,0] = torch.stack([torch.mean(tmp_mixed_x[p==i]) for i in np.unique(p)])
        mixed_x = mixed_x.to(device)
    else:
      mixed_x = lam * x + (1 - lam) * x[index, :]
    
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y 


def train_epoch(model, optimizer, criterion, dataloader, device , use_mixup, apply_dtw):
  model.train()
  losses = list()
  with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
    for idx, batch in iterator:
      optimizer.zero_grad()
      x, y = batch
      x = x.float()
      x, y = x.to(device), y.to(device)
      if use_mixup:
        x, y = mixup_data(x, y, device, alpha=0.4, apply_dtw=apply_dtw)
      output = model.forward(x)
      loss =  criterion(output, torch.max(y,1)[1])
      loss.backward()
      optimizer.step()
      losses.append(loss)
  return torch.stack(losses)

def test_epoch(model,criterion, dataloader,device):      
  model.eval()
  with torch.no_grad():
    losses = list()
    y_true_list = list()
    y_pred_list = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, batch in iterator:
          x_test, y_true,_ = batch
          x_test = x_test.float()
          y_pred = model.forward(x_test.to(device))
          loss = criterion(y_pred, y_true.to(device))
          losses.append(loss)
          y_true_list.append(y_true)
          y_pred_list.append(y_pred.argmax(-1))
    return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list)


#function to save the best model check point
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min training/validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is the best model
    if is_best:
        print ("=> Saving a new best")
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
    else:
        print ("=> Loss did not reduce")

def train(logdir,model,train_loss_min_input,checkpoint_path_train, best_model_path_train,train_dataloader, test_dataloader, epochs, device, apply_mixup, apply_dtw):
  
  criterion = torch.nn.CrossEntropyLoss(reduction="mean")
  optimizer = Adam(model.parameters(), lr=0.001)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001)
     
  os.makedirs(logdir, exist_ok=True)
  print(f"Logging results to {logdir}")

  train_loss_min = train_loss_min_input #??
  log = list()
  train_losses = list()
  test_losses = list()
  
  for epoch in range(epochs):
    train_loss = train_epoch(model, optimizer, criterion,train_dataloader,device,apply_mixup,apply_dtw)
    train_loss = train_loss.cpu().detach().numpy()[0] ## -1 ??
    scheduler.step(train_loss)
    test_loss, y_true_, y_pred = test_epoch(model,criterion, test_dataloader,device)
    test_loss = test_loss.cpu().detach().numpy()[0]

    scores = metrics(y_true_.cpu(), y_pred.cpu())
    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
              
    print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)
    scores["epoch"] = epoch
    scores["trainloss"] = train_loss
    scores["testloss"] = test_loss
    log.append(scores)

    log_df = pd.DataFrame(log).set_index("epoch")
    log_df.to_csv(os.path.join(logdir, "trainlog.csv"))

    
    # create checkpoint variable and add important data to save train and test model checkpoint    
    checkpoint = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
    } 
    # save train checkpoint
    best_flag = False
    if train_loss <= train_loss_min:      
      train_loss_min = train_loss
      best_flag = True
    save_ckp(checkpoint, best_flag, checkpoint_path_train, best_model_path_train)
       
def get_model(modelname, num_classes, input_dim, num_layers, hidden_dims, device):
    #modelname = modelname.lower() #make case invariant
    if modelname == "inception":
        model = inception.InceptionTime(num_classes=num_classes,input_dim=1, num_layers=6, hidden_dims=128, device=device).to(device)
    elif modelname == "nne":        
        model = inception.InceptionTime(num_classes=num_classes,input_dim=1, num_layers=6, hidden_dims=128, device=device).to(device)
    else:
        raise ValueError("invalid model argument. choose from 'InceptionTime', or 'NNE'")

    return model


############################################### main

def experiment(input_dir, output_dir, mixup, dtw, bacth_size, archive_name, id_partition=-1, epochs=1500, use_gpu=False):
    # choice of the device
    if torch.cuda.is_available() and use_gpu:
      device = "cuda"
    else:
      device = "cpu"
    device = torch.device(device)

    root_dir = input_dir
    #root_dir = '/content/gdrive/MyDrive/Inception_time/InceptionTime/archives/UCR_TS_Archive_2015'
   
    # run nb_iter_ iterations of Inception on the whole TSC archive  
    classifier_name = 'inception'
    #archive_name = ARCHIVE_NAMES[0]
    nb_iter_ = 5
 
    datasets_dict = read_all_datasets(root_dir, archive_name, id_partition)
    
    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)
        if archive_name == "GROUPED_TSC":
          tmp_classifier_archive = classifier_name + '/TSC' + trr + '/'
        else:  
          tmp_classifier_archive = classifier_name + '/' + archive_name + trr + '/'
        if mixup:
          if dtw:
            print('this is mixup directory')
            tmp_output_directory = output_dir + '/results-mixupdtw/' + tmp_classifier_archive
          else:
            print('this is mixup directory')
            tmp_output_directory = output_dir + '/results-mixup/' + tmp_classifier_archive
        else:
          print('this is not mixup directory')
          tmp_output_directory = output_dir + '/results-noda/' + tmp_classifier_archive
          
        #DATASET_NAMES = utils.constants.dataset_names_for_archive[archive_name]
        
        #for dataset_name in DATASET_NAMES:
        for dataset_name in datasets_dict.keys():
            print(dataset_name)
            print('\t\t\tdataset_name: ', dataset_name)

            train_dataloader, test_dataloader, nb_classes, y_true, enc = prepare_data(datasets_dict, dataset_name, bacth_size)

            output_directory = tmp_output_directory + dataset_name + '/'
                        
            temp_output_directory = create_directory(output_directory)
            if temp_output_directory is None:
                print('WARNING: ' + output_directory + ' already exists')
                continue
            
            # check point directories
            checkpoint_path_train = temp_output_directory + "/"+ "current_checkpoint.pt"
            best_model_path_train = temp_output_directory + "/"+  "best_model.pt"
            #print('checkpoint_path_train',checkpoint_path_train)

            model = get_model(modelname = classifier_name, num_classes=nb_classes, input_dim=1, num_layers=6, hidden_dims= 128, device = device)                      
            train(output_directory, model, np.inf,checkpoint_path_train,best_model_path_train,train_dataloader, test_dataloader, epochs, device, apply_mixup = mixup, apply_dtw=dtw)
          
            
    #  ensemble of inception time 
    
    #classifier_name = 'nne'

    #datasets_dict = read_all_datasets(root_dir, archive_name)

    #if mixup:
    #  tmp_output_directory = root_dir + '/results-mixup/' + classifier_name + '/' + archive_name + '/'
    #else:
    #  tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'

    #for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
    #    print('\t\t\tdataset_name: ', dataset_name)

    #    train_dataloader, test_dataloader, nb_classes, y_true, enc = prepare_data(datasets_dict, dataset_name)

    #    output_directory = tmp_output_directory + dataset_name + '/'
    #    from classifiers.nne import NNE

    #    model = NNE(output_directory)
    #    model.fit(test_dataloader, nb_classes,output_directory)
        
    

def parse_args():
    
    parser = argparse.ArgumentParser(description='Train an evaluate augmented UCR data using Mixup' 
                                                 'and non-augmented data with Inception time and nne models'
                                                 'This script trains a model on training dataset'
                                                 'partition, evaluates performance on a validation or evaluation partition'
                                                 'and stores progress and model paths in --logdir.')
    parser.add_argument('-i','--input_dir', action="store", help='input directory', type=str)  
    parser.add_argument('-d','--output_dir', action="store", help='output directory', type=str)  
    parser.add_argument(
        '-m','--mixup', default="False", action="store", type=lambda x: (str(x).lower() == 'true'), help='select whether to use mixup or not.')
    parser.add_argument(
        '--dtw', default="False", action="store", type=lambda x: (str(x).lower() == 'true'), help='select whether to use dtw-mixup or not.')
    parser.add_argument(
        '-b','--batchsize', default=64, action="store", help='batch size', type=int)
    parser.add_argument('-a','--archive_name', default="TSC", action="store", help='archive name', type=str)   
    parser.add_argument('-p','--id_partition', default=-1, action="store", help='partition id for grouped datasets', type=int)
    parser.add_argument('-e','--epochs', default=1500, action="store", help='partition id for grouped datasets', type=int)   
    parser.add_argument(
        '-g','--use_gpu', default="False", action="store", type=lambda x: (str(x).lower() == 'true'), help='select whether to use GPU or not.')
    args = parser.parse_args() 
    print(args)
    
    return args
        
if __name__ == "__main__":    
    args = parse_args()
    
    experiment(args.input_dir, args.output_dir, args.mixup, args.dtw, args.batchsize, args.archive_name, args.id_partition, args.epochs, args.use_gpu)
    
