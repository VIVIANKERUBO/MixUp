from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES

from utils.utils import read_all_datasets
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import run_length_xps
from utils.utils import generate_results_csv
from utils.utils import save_test_duration
from utils.utils import save_logs
from utils.utils import metrics
import os
import shutil
import argparse

import utils
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
from classifiers import nne
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cpu")

def prepare_data(datasets_dict, dataset_name):
    
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
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    y_true = torch.from_numpy(y_true)
    
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_true)
    train_dataloader = DataLoader(train_dataset, batch_size= 64, shuffle=True, num_workers= 0)
    test_dataloader = DataLoader(test_dataset, batch_size= 64, shuffle=False, num_workers= 0)
   
    return train_dataloader, test_dataloader, nb_classes, y_true, enc

    #return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc

use_cuda = torch.cuda.is_available()
def mixup_data(x, y, alpha=0.4, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y         

def train(model, optimizer, criterion, dataloader, device, use_mixup):

        
        model.train()
        losses = list()
        start_time = time.time()
        optimizer.zero_grad()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
          for idx, batch in iterator:
            x, y = batch
            x = x.float()
            x, y = x.to(device), y.to(device)
            if use_mixup:
              x, y = mixup_data(x, y, alpha=0.4)
            output = model.forward(x)
            loss =  criterion(output, torch.max(y,1)[1])
            loss.backward()
            optimizer.step()
            losses.append(loss)
        return torch.stack(losses)
    
def test(model,criterion, dataloader,device):
      
      model.eval()
      with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        start_time = time.time()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
              x_test, y_true = batch
              x_test = x_test.float()
              y_pred = model.forward(x_test.to(device))
              loss = criterion(y_pred, y_true.to(device))
              losses.append(loss)
              y_true_list.append(y_true)
              y_pred_list.append(y_pred.argmax(-1))
              y_score_list.append(y_pred.exp())

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)

 #function to save the best model check point
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def training(logdir,model,test_loss_min_input,checkpoint_path, best_model_path,train_dataloader, test_dataloader, apply_mixup):
  global test_loss_min
  device = torch.device("cpu")
  criterion = torch.nn.CrossEntropyLoss(reduction="mean")
  optimizer = Adam(model.parameters(), lr= 0.001)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.01, patience=50, min_lr=0.0001)
  epochs = 1500
            
  #logdir= temp_output_directory
  os.makedirs(logdir, exist_ok=True)
  print(f"Logging results to {logdir}")

  log = list()
  train_losses = list()
  test_losses = list()
  # initialize tracker for minimum validation loss
  test_loss_min = test_loss_min_input
  for epoch in range(epochs):

    
    if apply_mixup:
      train_loss = train(model, optimizer, criterion,train_dataloader,device, True)
    train_loss = train(model, optimizer, criterion,train_dataloader,device, False)
    test_loss, y_true_, y_pred, _ = test(model,criterion, x_test, y_true, x_train, y_train, y_test,device)
    scores = metrics(y_true_.cpu(), y_pred.cpu())
    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
    test_loss = test_loss.cpu().detach().numpy()[0]
    train_loss = train_loss.cpu().detach().numpy()[0]

    scheduler.step(test_loss)  
              
    print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)
    scores["epoch"] = epoch
    scores["trainloss"] = train_loss
    scores["testloss"] = test_loss
    log.append(scores)

    log_df = pd.DataFrame(log).set_index("epoch")
    log_df.to_csv(os.path.join(logdir, "trainlog.csv"))

    # create checkpoint variable and add important data
    checkpoint = {
            'epoch': epoch + 1,
            'test_loss_min': test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
    ## TODO: save the model if validation loss has decreased
    if test_loss <= test_loss_min:
      #print('Testing loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,test_loss))
      # save checkpoint as best model
      save_ckp(checkpoint, True, checkpoint_path, best_model_path)
      test_loss_min = test_loss

   
def get_model(modelname, num_classes, input_dim, num_layers, hidden_dims, device):
    #modelname = modelname.lower() #make case invariant
    if modelname == "inception":
        model = inception.InceptionTime(num_classes=nb_classes,input_dim=1, num_layers=6, hidden_dims=128).to(device)
    elif modelname == "nne":
        
        model = nne.NNE(output_directory,num_classes, input_dim, num_layers, hidden_dims)
    else:
        raise ValueError("invalid model argument. choose from 'InceptionTime', or 'NNE'")

    return model


############################################### main
def experiment(mixup):
    root_dir = '/content/gdrive/MyDrive/Inception_time/InceptionTime/archives/UCR_TS_Archive_2015'
    
    # run nb_iter_ iterations of Inception on the whole TSC archive  
    classifier_name = 'inception'
    archive_name = ARCHIVE_NAMES[0]
    nb_iter_ = 5
    
    
    datasets_dict = read_all_datasets(root_dir, archive_name)
    

    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)

        
        
        if mixup:
          tmp_output_directory = root_dir + '/results-test-mixup/' + classifier_name + '/' + archive_name + trr + '/'
        else:
          tmp_output_directory = root_dir + '/results-test/' + classifier_name + '/' + archive_name + trr + '/'

        for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
            print('\t\t\tdataset_name: ', dataset_name)

            
            train_dataloader, test_dataloader, nb_classes, y_true, enc = prepare_data(datasets_dict, dataset_name)
            

            output_directory = tmp_output_directory + dataset_name + '/'

            temp_output_directory = create_directory(output_directory)

            if temp_output_directory is None:
                print('Already_done', tmp_output_directory, dataset_name)
                continue
            
            
            checkpoint_path = temp_output_directory + "/" + "current_checkpoint.pt"
            best_model_path = temp_output_directory + "/" + "best_model.pt"
            model = get_model(modelname = classifier_name,num_classes=nb_classes,input_dim=1, num_layers=6, hidden_dims= 128, device = device)
            
            
            training(temp_output_directory, model, np.inf,checkpoint_path, best_model_path,train_dataloader,
            test_dataloader,mixup)
   
           
        
        
def parse_args():
    
    parser = argparse.ArgumentParser(description='Train an evaluate augmented UCR data using Mixup' 
                                                 'and non-augmented data with Inception time and nne models'
                                                 'This script trains a model on training dataset'
                                                 'partition, evaluates performance on a validation or evaluation partition'
                                                 'and stores progress and model paths in --logdir.')
    parser.add_argument(
        '-m','--mixup', default="False", action="store",type=lambda x: (str(x).lower() == 'true'),help='select whether to use mixup or not.')

    args = parser.parse_args()

    return args
        
if __name__ == "__main__":
    
    args = parse_args()

    experiment(args.mixup)
           
              
        ############### run the ensembling of these iterations of Inception ################### 
        '''
        #  ensemble of inception time       
    classifier_name = 'inception'

    datasets_dict = read_all_datasets(root_dir, archive_name)

    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)
  

    if mixup:
      print('this is mixup directory')
      tmp_output_directory = root_dir + '/results-test-mixup/' + classifier_name + '/' + archive_name + trr + '/'
      #tmp_output_directory = root_dir + '/results-test-mixup/' + classifier_name + '/' + archive_name + '/'
      print(tmp_output_directory)
    else:
      print('this is not mixup directory')
      tmp_output_directory = root_dir + '/results-test/' + classifier_name + '/' + archive_name + trr + '/'
      #tmp_output_directory = root_dir + '/results-test/' + classifier_name + '/' + archive_name + '/'
    
    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
          
      print('\t\t\tdataset_name: ', dataset_name)

      train_dataloader, test_dataloader, nb_classes, y_true, enc = prepare_data(datasets_dict, dataset_name)
        

      output_directory = tmp_output_directory + dataset_name + '/'

      import glob

      #for layers in glob.glob(tmp_output_directory + "/" + "*.shp" ):
      for layers in glob.glob(output_directory ):
          print(layers)

        
      temp_output_directory = output_directory

      ckp_path = temp_output_directory + "/" + "current_checkpoint_train.pt"

          
          # load the saved inception model
      saved_model = model, optimizer, start_epoch, train_loss_min = load_ckp(ckp_path, model, optimizer)
      models = []
      num_models = 5
      for m in range(num_models):
        models.append(saved_model)
        
      for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        x_test = inputs 
        y_test = targets

        
      y_pred = torch.zeros(y_test.shape)

      ll = 0
      for m_idx, model in enumerate(models):
          

        test_loss, y_true_, y_pred_, _ = test(model,criterion, test_dataloader,device)
        scores = metrics(y_true_.cpu(), y_pred.cpu())
          
        predictions_file_name = output_directory + 'y_pred.npy'
            # check if predictions already made
        if check_if_file_exits(predictions_file_name):
            # then load only the predictions from the file
          curr_y_pred = np.load(predictions_file_name)
        else:
              # then compute the predictions
          curr_y_pred = y_pred_

        np.save(predictions_file_name, curr_y_pred)

        y_pred = y_pred + curr_y_pred

        ll += 1

          # average predictions
      y_pred = y_pred / ll

        # save predictions
      np.save(output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
      y_pred = np.argmax(y_pred, axis=1)


      df_metrics = metrics(y_true, y_pred)

      df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)
      print('\t\t\t\tDONE')

            '''

          

                                              







          
            
            
