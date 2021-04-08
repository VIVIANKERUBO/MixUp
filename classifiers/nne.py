import keras
import numpy as np
from utils.utils import calculate_metrics
from utils.utils import metrics
from utils.utils import create_directory
from utils.utils import check_if_file_exits
import gc
import torch
from torch.optim import Adam
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
import time

#loading function
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    test_loss_min = checkpoint['test_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], test_loss_min.item()

class NNE:

    def create_classifier(self, model_name, num_classes,input_dim, num_layers, hidden_dims, output_directory
                          ):
        if self.check_if_match('inception*', model_name):
            from classifiers import inception
            return inception.InceptionTime(num_classes,input_dim, num_layers, hidden_dims
                                                 )

    def check_if_match(self, rex, name2):
        import re
        pattern = re.compile(rex)
        return pattern.match(name2)

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, nb_iterations=5,
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
        self.verbose = verbose
        self.models_dir = output_directory.replace('nne', 'classifier')

    def fit(self, x_train, y_train, x_test, y_test, y_true, nb_classes,ckp_path, output_directory):
        # no training since models are pre-trained
        start_time = time.time()

        y_pred = np.zeros(shape=y_test.shape)

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

                model = self.create_classifier(model_name, nb_classes, 1, 6, 32,
                                               curr_dir)
                # define optimzer
                optimizer = Adam(model.parameters(), lr=0.001)

                # define checkpoint saved path
                #ckp_path = "./current_checkpoint.pt"

                model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)

                model.eval()

                with torch.no_grad():
                  x_test, y_test = x_test.to_device(), y_test.to_device()
                  output = model(x_test)
                  predictions_file_name = curr_dir + 'y_pred.npy'
                  # check if predictions already made
                  if check_if_file_exits(predictions_file_name):
                    # then load only the predictions from the file
                    curr_y_pred = np.load(predictions_file_name)
                  else:
                    # then compute the predictions
                    curr_y_pred = output

                  np.save(predictions_file_name, curr_y_pred)

                y_pred = y_pred + curr_y_pred

                ll += 1

        # average predictions
        y_pred = y_pred / ll

        # save predictions
        np.save(output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        duration = time.time() - start_time

        metrics(y_true, y_pred, duration)

        df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

        gc.collect()
    
    