from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn import MSELoss, BCELoss
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mlp

import random
import sys
random.seed(10)
np.random.seed(10)

import copy

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

class ResNet_classification(nn.Module):
    def __init__(self, params):
        super(ResNet_classification, self).__init__()

        self.MODEL_NAME = "ResNet_classification"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        self.mains_params = params.get('mains_params', {})

        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise SequenceLengthError

    def partial_fit(self,train_main,train_appliances,do_preprocessing=True,**load_kwargs):

        print("...............ResNet_classification partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        self.set_mains_params(train_main)  

        if do_preprocessing:
            ## for classification subnetwork creating labels
            classify_appliance=copy.deepcopy(train_appliances)
            classification=self.classify(classify_appliance)

            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        train_main = pd.concat(train_main,axis=0)
        train_main = train_main.values.reshape((-1,self.sequence_length,1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs,axis=0)
            app_df_values = app_df.values.reshape((-1,self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        new_train_appliances_classification = {}
        for app_name, app_df in classification:
            app_df = pd.concat(app_df,axis=0)
            app_df_values = app_df.values.reshape((-1,self.sequence_length))
            new_train_appliances_classification[app_name]=app_df_values
        train_appliances_classification = new_train_appliances_classification

        
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
                    filepath = 'ResNet_classification-temp-weights-'+str(random.randint(0,100000))+'.h5'
                    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)

                    power = pd.DataFrame(power)
                    classification_app = pd.DataFrame(train_appliances_classification[appliance_name])
                    power = pd.concat((power, classification_app), axis=1)
                    power = np.array(power)

                    train_x, v_x, train_class_y, v_class_y = train_test_split(train_main, power, test_size=.15,
                                                                              random_state=10)
                    train_y = train_class_y[:, :self.sequence_length]
                    v_y = v_class_y[:, :self.sequence_length]
                    appliance_train_classification = train_class_y[:, self.sequence_length:]
                    appliance_val_classification = v_class_y[:, self.sequence_length:]

                    history = model.fit(train_x, [train_y, appliance_train_classification],
                                        validation_data=(v_x, [v_y, appliance_val_classification]), epochs=self.n_epochs,
                                        callbacks=[checkpoint], batch_size=self.batch_size)
                    model.load_weights(filepath)

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

            for appliance in self.models:

                prediction = []
                model = self.models[appliance]
                prediction_output, prediction_classification = self.models[appliance].predict(x=test_main_array,
                                                                                                batch_size=self.batch_size)
                #####################
                # This block is for creating the average of predictions over the different sequences
                # the counts_arr keeps the number of times a particular timestamp has occured
                # the sum_arr keeps the number of times a particular timestamp has occured
                # the predictions are summed for  agiven time, and is divided by the number of times it has occured

                l = self.sequence_length
                n = len(prediction_output) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction_output)):
                    sum_arr[i:i + l] += prediction_output[i].flatten()
                    counts_arr[i:i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]

                prediction = (self.appliance_params[appliance]['min'] + (
                            sum_arr * (self.appliance_params[appliance]['max'] - self.appliance_params[appliance][
                        'min'])))

                l = self.sequence_length
                n = len(prediction_classification) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction_classification)):
                    sum_arr[i:i + l] += prediction_classification[i].flatten()
                    counts_arr[i:i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]

                #################
                prediction_classification = sum_arr
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df

                ##################
                # PLOTTING
                self.classification_output_plot(prediction_classification, appliance)

            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions

    def return_network(self):

        filters = 32
        kernel_size = 4
        units = 128

        # This classificcation network is inspired from:-
        # https://github.com/antoniosudoso/attention-nilm

        input_data = nn.Conv1d(1, 30, 10)(input_data)
        x = nn.Conv1d(30, 30, 8)(input_data)
        x = nn.Conv1d(30, 40, 6)(x)
        x = nn.Conv1d(40, 50, 5)(x)
        x = nn.Conv1d(50, 50, 5)(x)
        x = nn.Conv1d(50, 50, 5)(x)
        x = nn.Flatten()(x)
        x = nn.Linear(1024, units)(x)
        classification_output = nn.Linear(self.sequence_length, 1)(x)

        # Regression subnetwork
        y = nn.ZeroPad1d(3)(input_data)
        y = nn.Conv1d(1, 30, 48, stride=2)(y)
        y = nn.BatchNorm1d(30)(y)
        y = nn.ReLU()(y)
        y = nn.MaxPool1d(3, stride=2)(y)

        # Two types of residual block used for resnet
        y = convolution_block([30, 30, 30], 24)(y)
        y = identity_block([30, 30, 30], 12)(y)
        y = identity_block([30, 30, 30], 6)(y)

        # Fully connected layers
        y = nn.Flatten()(y)
        y = nn.Linear(1024, 1024)(y)
        y = nn.Dropout(.2)(y)
        regression_output = nn.Linear(self.sequence_length, 1)(y)

        output = torch.mul(regression_output, classification_output)

        full_model = nn.Sequential(input_data, classification_output, regression_output, output)

        optimizer = SGD(full_model.parameters(), lr=0.01, momentum=0.9)
        full_model.summary()
        # Two outputs of the model the classification output and the final output
        full_model.compile(optimizer=optimizer,
                           loss={"output": MSELoss(), "classification_output": BCELoss()})
        return full_model

    def classify(self, classify_appliance):
        appliance_on_off = []
        # Threshold for on-off
        THRESHOLD = 15
        for app_index, (appliance_name, on_off_list) in enumerate(classify_appliance):
            classification_appliance_dfs = []
            for appliance in on_off_list:
                n = self.sequence_length
                units_to_pad = n // 2
                appliance[appliance <= THRESHOLD] = 0
                appliance[appliance > THRESHOLD] = 1
                new_app_readings = appliance.values.flatten()
                new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant',
                                          constant_values=(0, 0))
                new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                # Return as a list of dataframe
                classification_appliance_dfs.append(pd.DataFrame(new_app_readings))
            appliance_on_off.append((appliance_name, classification_appliance_dfs))
        return appliance_on_off

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                    app_min = self.appliance_params[app_name]['min']
                    app_max = self.appliance_params[app_name]['max']
                else:
                    print("Parameters for ", app_name, " were not found!")
                    raise ApplianceNotFoundError()

                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant',
                                              constant_values=(0, 0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                    # Normalizing data
                    new_app_readings = (new_app_readings - app_min) / (app_max - app_min)
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))

                appliance_list.append((app_name, processed_app_dfs))

            return processed_mains_lst, appliance_list

        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                # new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_mains_params(self, train_main):
        l = []
        for mains in train_main:
            new_mains = mains.values.flatten()
            l.extend(new_mains)

        main_mean = np.mean(l)
        main_std = np.std(l)
        main_min = np.min(l)
        main_max = np.max(l)
        self.mains_params.update({'mean': main_mean, 'std': main_std, 'min': main_min, 'max': main_max})

    def set_appliance_params(self, train_appliances):

        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            app_max = np.max(l)
            app_min = np.min(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std, 'min': app_min, 'max': app_max}})

    
    

