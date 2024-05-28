from nilmtk.disaggregate import Disaggregator
import pandas as pd
import numpy as np
from collections import OrderedDict 
import torch
from nilmtk.disaggregate import Disaggregator
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from sklearn.model_selection import train_test_split
from statistics import mean


class DAEencoder(nn.Module):
    def __init__(self, sequence_length):
        super(DAEencoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=4, stride=1, padding='same')
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(8 * sequence_length, 8 * sequence_length)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(8 * sequence_length, 128)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(128, 8 * sequence_length)
        self.act3 = nn.ReLU()
        self.conv2 = nn.Conv1d(8, 1, kernel_size=4, stride=1, padding="same")

    def forward(self, x):
        x = self.conv1(x)
        x = self.flat(x)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        x = x.view(x.size(0), 8, -1)
        x = self.conv2(x)
        return x


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
            torch.save(model_state_dict.state_dict(), self.filepath)

class DAE(Disaggregator):

    def __init__(self, params):
        """
        Iniititalize the moel with the given parameters
        """
        self.MODEL_NAME = "DAE"
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',512)
        self.mains_mean = params.get('mains_mean',1000)
        self.mains_std = params.get('mains_std',600)
        self.appliance_params = params.get('appliance_params',{})
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.models = OrderedDict()
        if self.load_model_path:
            self.load_model()


    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """
        The partial fit function
        """

        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        # To preprocess the data and bring it to a valid shape
        if do_preprocessing:
            print ("Preprocessing")
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')
        train_main = pd.concat(train_main, axis=0).values
        train_main = train_main.reshape((-1, self.sequence_length, 1))
        new_train_appliances  = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0).values
            app_df = app_df.reshape((-1, self.sequence_length, 1))
            new_train_appliances.append((app_name, app_df))

        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                self.models[appliance_name] = self.return_network()
                # print(self.models[appliance_name].summary())

            print("Started Retraining model for", appliance_name)
            model = self.models[appliance_name]
            # # Now we will print model summary
            # total_params = 0
            # for name, param in model.named_parameters():  
            #     if param.requires_grad:
            #         num_params = param.numel()
            #         print(f'Layer: {name} | Number of parameters: {num_params}')
            #         total_params += num_params
            # print(f'Total number of trainable parameters: {total_params}')


            filepath = self.file_prefix +".pth"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
            # print("shape of train_main",train_main.dtype)
            # print("shape of power",power.dtype)
            train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)
            train_dataset = TensorDataset(torch.tensor(train_x).permute(0, 2, 1).float(), torch.tensor(train_y).float())
            val_dataset = TensorDataset(torch.tensor(v_x).permute(0, 2, 1).float(), torch.tensor(v_y).float())
             
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            self.train_model(model, train_loader, val_loader, checkpoint)

            self.models[appliance_name].load_state_dict(torch.load(filepath))

    def train_model(self, model, train_loader, val_loader, checkpoint):
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



    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list,submeters_lst=None,method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1,self.sequence_length,1))
            test_main_array = torch.tensor(test_main).permute(0, 2, 1)
            test_main_array = test_main_array.float()
            disggregation_dict = {}
            for appliance in self.models:
                prediction = []
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_array)
                prediction = prediction.numpy()
                app_mean = self.appliance_params[appliance]['mean']
                app_std = self.appliance_params[appliance]['std']
                prediction = self.denormalize_output(prediction,app_mean,app_std)
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
                series = pd.Series(valid_predictions)
                disggregation_dict[appliance] = series
            results = pd.DataFrame(disggregation_dict,dtype='float32')
            test_predictions.append(results)
        return test_predictions
            
    def return_network(self):
        model = DAEencoder(self.sequence_length)
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        sequence_length  = self.sequence_length
        if method=='train':
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std,True)
                processed_mains.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name,app_df_list) in submeters_lst:
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                processed_app_dfs = []
                for app_df in app_df_list:
                    data = self.normalize_output(app_df.values, sequence_length,app_mean,app_std,True)
                    processed_app_dfs.append(pd.DataFrame(data))                    
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method=='test':
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std,False)
                processed_mains.append(pd.DataFrame(mains))
            return processed_mains
    
        
    def normalize_input(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0)   
        if overlapping:
            windowed_x = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_x = arr.reshape((-1,sequence_length))
        windowed_x = windowed_x - mean
        windowed_x = windowed_x/std
        return (windowed_x/std).reshape((-1,sequence_length))

    def normalize_output(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0) 
        if overlapping:  
            windowed_y = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_y = arr.reshape((-1,sequence_length))        
        windowed_y = windowed_y - mean
        return (windowed_y/std).reshape((-1,sequence_length))

    def denormalize_output(self,data,mean,std):
        return mean + data*std
    
    def set_appliance_params(self,train_appliances):
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})