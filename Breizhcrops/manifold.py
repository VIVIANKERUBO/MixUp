from utils.utils import metrics
import os
import shutil
import argparse
import breizhcrops
from tqdm import tqdm
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import classifiers
from classifiers import inceptionmanifold
from classifiers.inception import InceptionTime
from classifiers.inceptionmanifold import InceptionTime_Hidden
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cpu")

use_cuda = torch.cuda.is_available() 

# Define beta distribution
def mixup_data(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam
       

def train(model,bce_loss, softmax,optimizer, dataloader, device):

        
        model.train()
        losses = list()
        start_time = time.time()
        optimizer.zero_grad()
        #with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        #for idx, batch in iterator: 
        for batch_idx, (x, y, _) in enumerate(dataloader):
        
            #x, y,_ = batch
            x = x.float()
            
            #x, y = mixup_data(x, y, alpha=0.4)
            # if you use manifold mixup
            lam = mixup_data(alpha=0.4)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).to(device)
            output, reweighted_target = model(x, lam=lam, target=y)
            loss = bce_loss(softmax(output), reweighted_target)
            #x, y_a, y, lam = mixup_data(x, y, alpha=0.4, use_cuda = False)
            #x, y_a, y = map (Variable, (x,y_a, y))
            #output = model(x)
            optimizer.zero_grad()
            #loss = mixup_criterion(criterion, output, y_a, y, lam)
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
        field_ids_list = list()
        start_time = time.time()
        #with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        #for idx, batch in iterator:
        for batch_idx, (x_test, y_true, field_id) in enumerate(dataloader):
              #x_test, y_true, field_id = batch
              x_test = x_test.float()
              y_pred = model.forward(x_test.to(device))
              loss = criterion(y_pred, y_true.to(device))
              losses.append(loss)
              y_true_list.append(y_true)
              y_pred_list.append(y_pred.argmax(-1))
              y_score_list.append(y_pred.exp())
              field_ids_list.append(field_id)

        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list),torch.cat(y_score_list), torch.cat(field_ids_list)

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

def training(logdir,model,train_loss_min_input,checkpoint_path, best_model_path,train_dataloader, test_dataloader):
  train_loss_min = train_loss_min_input
  device = torch.device("cpu")
  criterion = torch.nn.CrossEntropyLoss(reduction="mean")
  #loss_function = nn.CrossEntropyLoss()
  bce_loss = torch.nn.BCELoss()
  softmax = nn.Softmax(dim=1).cpu()

  #optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  optimizer = Adam(model.parameters(), lr= 0.0001)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.01, patience=50, min_lr=0.0001)
  epochs = 10
            
  #logdir= temp_output_directory
  os.makedirs(logdir, exist_ok=True)
  print(f"Logging results to {logdir}")

  log = list()
  train_losses = list()
  test_losses = list()

  for epoch in range(epochs):   
    train_loss = train(model, bce_loss, softmax, optimizer,train_dataloader,device)
    test_loss, y_true_, y_pred, *_ = test(model,criterion, test_dataloader,device)
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
            'test_loss_min': train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)

    ## TODO: save the model if validation loss has decreased
    if train_loss <= train_loss_min:
      #print('Testing loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,test_loss))
      # save checkpoint as best model
      save_ckp(checkpoint, True, checkpoint_path, best_model_path)
      train_loss_min = train_loss

def get_dataloader(datapath, mode, batchsize, workers, preload_ram=False, level="L1C"):
    print(f"Setting up datasets in {os.path.abspath(datapath)}, level {level}")

    datapath = os.path.abspath(datapath)

    frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath,
                                    preload_ram=preload_ram, level=level)

   # frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath,
                                   # preload_ram=preload_ram, level=level,transform = transform)

    #frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath,
                                    #preload_ram=preload_ram, level=level,transform = transform)

    if "evaluation" in mode:
            frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath,
                                            preload_ram=preload_ram, level=level)
    if mode == "evaluation" or mode == "evaluation1":
            '''traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh03])'''
            traindatasets = frh01
            testdataset = frh04
    else:
        raise ValueError("only --mode 'validation' or 'evaluation' allowed")


    
    traindataloader = DataLoader(traindatasets, batch_size=batchsize, shuffle=True, num_workers= 4)

    testdataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers= 4)


    meta = dict(
        ndims=13 if level == "L1C" else 10,
        num_classes=len(belle_ile.classes) if mode == "unittest" else len(frh01.classes),
        sequencelength=45
    )

    return traindataloader, testdataloader, meta


def get_model(modelname, ndims, num_classes, sequencelength, device, **hyperparameter):
    modelname = modelname.lower() #make case invariant
    if modelname == "inceptiontime":
        model = InceptionTime(input_dim=ndims, num_classes=num_classes, device=device,
                              **hyperparameter).to(device) 
    elif modelname == "inceptiontimemanifold":
        model = InceptionTime_Hidden(input_dim=ndims, num_classes=num_classes, device=device,
                              **hyperparameter).to(device) 

    else:
        raise ValueError("invalid model argument. choose from inceptiontime or inceptionmanifold")

    return model         

############################################### main
def experiment(mixup):
    
    root_dir = '/content/drive/MyDrive/Breizhcrops_mixup/breizhcrops_dataset'

    train_dataloader, test_dataloader, meta =  get_dataloader(root_dir, mode = "evaluation1", batchsize = 64, workers = 0, preload_ram=False, 
    level="L1C")
       
    if mixup:    
      temp_out = root_dir + '/' + "L1C" + '/'+ "manifoldmixup_128" 
    else:
      print('No directory')
    
    num_classes = meta["num_classes"]
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]

    
    model = get_model(args.model, ndims, num_classes, sequencelength, device, **args.hyperparameter)
    
    from torchsummary import summary
    print(summary(model,(45, 13)))
    
    

     
    checkpoint_path = temp_out + "/" + "current_checkpoint.pt"
    best_model_path = temp_out + "/" + "best_model.pt"        
    training(temp_out, model, np.inf,checkpoint_path,best_model_path, train_dataloader,
            test_dataloader)
           
        
def parse_args():
    
    parser = argparse.ArgumentParser(description='Train an evaluate augmented Breizhcrops data using Mixup' 
                                                 'and non-augmented data with Inception time'
                                                 'This script trains a model on training dataset'
                                                  'evaluates performance on a validation or test'
                                                 'and stores progress and model paths in --logdir.')
    parser.add_argument(
        'model', type=str, default=" inceptiontime", help='select model inception model architecture')

    parser.add_argument(
        '-H', '--hyperparameter', type=str, default=None, help='model specific hyperparameter as single string, '
                                                               'separated by comma of format param1=value1,param2=value2')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-6, help='optimizer weight_decay (default 1e-6)')
    parser.add_argument(
        '--learning-rate', type=float, default=1e-2, help='optimizer learning rate (default 1e-2)')

    parser.add_argument(
        '-m','--mixup', default="False", action="store",type=lambda x: (str(x).lower() == 'true'),help='select whether to use mixup or not.')
    

    args = parser.parse_args()
    
    hyperparameter_dict = dict()
    if args.hyperparameter is not None:
        for hyperparameter_string in args.hyperparameter.split(","):
            param, value = hyperparameter_string.split("=")
            hyperparameter_dict[param] = float(value) if '.' in value else int(value)
    args.hyperparameter = hyperparameter_dict


    return args
        
if __name__ == "__main__":
    
    args = parse_args()

    experiment(args.mixup)