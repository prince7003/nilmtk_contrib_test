from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from nilmtk.disaggregate import Disaggregator
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from sklearn.model_selection import train_test_split


 


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

    def step(self, val_loss, model_state_dict):
        if val_loss < self.best_loss:
            if self.verbose > 0:
                print(f"Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
            self.best_loss = val_loss
            torch.save(model_state_dict.state_dict(), self.filepath)


class Seq2Seq(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "Seq2Seq"
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise SequenceLengthError

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        print("...............Seq2Seq partial_fit running...............")
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
            # # Now we will print model summary
            # total_params = 0
            # for name, param in model.named_parameters():  
            #     if param.requires_grad:
            #         num_params = param.numel()
            #         print(f'Layer: {name} | Number of parameters: {num_params}')
            #         total_params += num_params
            # print(f'Total number of trainable parameters: {total_params}')

            if train_main.size > 0 and len(train_main) > 10:
                filepath = self.file_prefix +".pth"
                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)

                train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)
                train_dataset = TensorDataset(torch.tensor(train_x).permute(0, 2, 1), torch.tensor(train_y))
                val_dataset = TensorDataset(torch.tensor(v_x).permute(0, 2, 1), torch.tensor(v_y))
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

                self.train_model(model, train_loader, val_loader, checkpoint)

                self.models[appliance_name].load_state_dict(torch.load(filepath))
                 

    def train_model(self, model, train_loader, val_loader=None, checkpoint=None):
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters())

        for epoch in range(self.n_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            if val_loader:
                model.eval()
                val_loss = self.evaluate_model(model, val_loader, criterion)
                print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {epoch_loss}, Val Loss: {val_loss}")
                checkpoint.step(val_loss, model)
            else:
                print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {epoch_loss}")
 

    def evaluate_model(self, model, val_loader, criterion):
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        return val_loss / len(val_loader)

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
            print(test_main_array.shape)
            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_array)
                print(prediction.shape)

                # # Now we will print model summary
                # total_params = 0
                # for name, param in model.named_parameters():  
                #   if param.requires_grad:
                #     num_params = param.numel()
                #     print(f'Layer: {name} | Number of parameters: {num_params}')
                #     total_params += num_params
                # print(f'Total number of trainable parameters: {total_params}')

 

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

        seqlen = self.sequence_length
        seqlen = int((seqlen-10)/2) + 1
        seqlen = int((seqlen-8)/2) + 1
        seqlen = int((seqlen-6)/1) + 1
        seqlen = int((seqlen-5)/1) + 1
        seqlen = int((seqlen-5)/1) + 1
        model = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(50, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(50 *seqlen, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
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
            #new_mains = pd.DataFrame(new_mains)
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
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
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
