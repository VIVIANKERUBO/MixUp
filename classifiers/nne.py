import keras
import pandas as pd
import numpy as np
import classifiers
from classifiers import inception
from utils.utils import calculate_metrics
from utils.utils import metrics
from utils.utils import create_directory
from utils.utils import check_if_file_exits
import gc
import os
import torch
from torch.optim import Adam
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
import time


device = torch.device("cpu")

class NNE:

    def __init__(self, output_directory, nb_iterations=5,
                 clf_name='inception'):
        self.classifiers = [clf_name]
        out_add = ''
        for cc in self.classifiers:
            out_add = out_add + cc + '-'
        self.archive_name = ARCHIVE_NAMES[0]
        self.iterations_to_take = [i for i in range(nb_iterations)]
        for cc in self.iterations_to_take:
            out_add = out_add + str(cc) + '-'
        self.output_directory = output_directory.replace('nne',
                                                         'nne' + '/' + out_add)
        create_directory(self.output_directory)
        self.dataset_name = output_directory.split('/')[-2]
        self.models_dir = output_directory.replace('nne', clf_name)
    
    def fit(self, test_dataloader, num_classes, output_directory):
        # no training since models are pre-trained
        start_time = time.time()

        for batch_idx, (x_test, y_true, y_test) in enumerate(test_dataloader):
          
          
          y_pred = torch.zeros(y_test.shape)
          

        ll = 0

          # loop through all classifiers
        for model_name in self.classifiers:
            # loop through different initialization of classifiers
            for itr in self.iterations_to_take:
                if itr == 0:
                    itr_str = ''
                else:
                    itr_str = '_itr_' + str(itr)

                curr_archive_name = self.archive_name + itr_str

                curr_dir = self.models_dir.replace('classifier', model_name).replace(
                    self.archive_name, curr_archive_name)
                print('curr_dir',curr_dir)

                # define checkpoint saved path
                ckp_path = curr_dir + "current_checkpoint.pt"
                print(ckp_path)

                model = inception.InceptionTime(num_classes=num_classes,input_dim=1, num_layers=6, hidden_dims=128).to(device)
                
                checkpoint = torch.load(ckp_path)
    
                model.load_state_dict(checkpoint['state_dict']) 
                model.eval()
                with torch.no_grad():
                  x_test = x_test.float()
                  y_pred_ = model.forward(x_test.to(device))
                  predictions_file_name = curr_dir + 'y_pred.npy'
    
                  curr_y_pred = y_pred_
                  

                np.save(predictions_file_name, curr_y_pred)

                y_pred = y_pred + curr_y_pred
               

                ll += 1

        # average predictions
        y_pred = y_pred / ll

        # save predictions
        #np.save(output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        duration = time.time() - start_time

        df_metrics = metrics(y_true, y_pred)
       

        log_df = pd.DataFrame([df_metrics])
        
        df_metrics_dir = os.path.join(self.output_directory, 'df_metrics.csv') 
        log_df.to_csv(df_metrics_dir)
