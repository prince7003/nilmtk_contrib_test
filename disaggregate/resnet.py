from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
random.seed(10)
np.random.seed(10)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass
 

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_model_state_dict = None

    def step(self, val_loss, model_state_dict):
        if val_loss < self.best_loss:
            if self.verbose > 0:
                print(f"Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
            self.best_loss = val_loss
            self.best_model_state_dict = model_state_dict
            torch.save(model_state_dict, self.filepath)

    def load_best_model(self, model):
        if self.best_model_state_dict is not None:
            model.load_state_dict(self.best_model_state_dict)


class identity_block(nn.Module):
    def __init__(self, filters, kernel_size):
        super(identity_block, self).__init__()
        self.conv1 = nn.Conv1d(filters[0], filters[1], kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(filters[1], filters[2], kernel_size, stride=1, padding='same')
        self.conv3 = nn.Conv1d(filters[2], filters[2], kernel_size, stride=1, padding='same')
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()

    def forward(self, x):
        first_layer = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        residual = x + first_layer
        x = F.relu(residual)
        return x

class convolution_block(nn.Module):
    def __init__(self, filters, kernel_size):
        super(convolution_block, self).__init__()
        self.conv1 = nn.Conv1d(filters[0], filters[1], kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(filters[1], filters[2], kernel_size, stride=1, padding='same')
        self.conv3 = nn.Conv1d(filters[2], filters[2], kernel_size, stride=1, padding='same')
        self.conv4 = nn.Conv1d(filters[0], filters[2], kernel_size, stride=1, padding='same')
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()

    def forward(self, x):
        first_layer = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        first_layer = self.conv4(first_layer)
        convolution = x + first_layer
        x = self.act4(convolution)
        return x

class ResNet(Disaggregator):
    def __init__(self, params):
        super(ResNet, self).__init__()

        self.MODEL_NAME = "ResNet"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 299)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = nn.ModuleDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.load_model_path = params.get('load_model_path', None)
        self.appliance_params = params.get('appliance_params', {})
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True,  **load_kwargs):

        print("...............ResNet partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))

        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = 'ResNet-temp-weights-'+str(random.randint(0,100000))+'.h5'
                    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)
                    train_dataset = TensorDataset(torch.tensor(train_x).permute(0, 2, 1), torch.tensor(train_y))
                    val_dataset = TensorDataset(torch.tensor(v_x).permute(0, 2, 1), torch.tensor(v_y))
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

                    self.train_model(model, train_loader, val_loader, checkpoint)
    

    def train_model(self, model, train_loader, val_loader, checkpoint):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()
        for epoch in range(self.n_epochs):
            model.train()
            train_loss = 0.0
            for i, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{self.n_epochs}, Training Loss: {train_loss:.6f}")

            model.eval()
            val_loss = 0.0
            for i, (x_batch, y_batch) in enumerate(val_loader):
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{self.n_epochs}, Validation Loss: {val_loss:.6f}")

            checkpoint.step(val_loss, model.state_dict())
        checkpoint.load_best_model(model)
        os.remove(checkpoint.filepath)


    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
            if model is not None:
                self.models = model

            if do_preprocessing:
                test_main_list = self.call_preprocessing(
                    test_main_list, submeters_lst=None, method='test')

            test_predictions = []
            for test_mains_df in test_main_list:

                disggregation_dict = {}
                test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
                test_main_array = torch.tensor(test_main_array).permute(0, 2, 1)
                for appliance in self.models:

                    prediction = []
                    model = self.models[appliance]
                    model.eval()
                    with torch.no_grad():
                        prediction = model(test_main_array)
                    prediction = prediction.numpy()
                    
                    #####################
                    # This block is for creating the average of predictions over the different sequences
                    # the counts_arr keeps the number of times a particular timestamp has occured
                    # the sum_arr keeps the number of times a particular timestamp has occured
                    # the predictions are summed for  agiven time, and is divided by the number of times it has occured


                    l = self.sequence_length
                    n = len(prediction) + l - 1
                    sum_arr = np.zeros((n))
                    counts_arr = np.zeros((n))
                    o = len(sum_arr)
                    for i in range(len(prediction)):
                        sum_arr[i:i + l] += prediction[i].flatten()
                        counts_arr[i:i + l] += 1
                    for i in range(len(sum_arr)):
                        sum_arr[i] = sum_arr[i] / counts_arr[i]

                    prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                    valid_predictions = prediction.flatten()
                    valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                    df = pd.Series(valid_predictions)
                    disggregation_dict[appliance] = df
                results = pd.DataFrame(disggregation_dict, dtype='float32')
                test_predictions.append(results)

            return test_predictions

    def return_network(self):
        num_filters=30
        seqlen = self.sequence_length
        seqlen = int((seqlen-48+6)/2) + 1
        seqlen = int((seqlen-3)/2) + 1
        
        model = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=48, stride=2, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            convolution_block([num_filters, num_filters, num_filters], kernel_size=24),
            identity_block([num_filters, num_filters, num_filters], kernel_size=12),
            identity_block([num_filters, num_filters, num_filters], kernel_size=6),
            nn.Flatten(),
            nn.Linear(num_filters * seqlen, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, self.sequence_length)
        )
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':            
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                    app_min=self.appliance_params[app_name]['min']
                    app_max=self.appliance_params[app_name]['max']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()


                processed_app_dfs = []
                for app_df in app_df_lst:                    
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])                    
                    new_app_readings = (new_app_readings - app_mean) / app_std  # /self.max_val
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                    
                    
                appliance_list.append((app_name, processed_app_dfs))
                #new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])
                #print (new_mains.shape, new_app_readings.shape, app_name)

            return processed_mains_lst, appliance_list

        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                #new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self,train_appliances):

        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            app_max=np.max(l)
            app_min=np.min(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std,'max':app_max,'min':app_min}})
