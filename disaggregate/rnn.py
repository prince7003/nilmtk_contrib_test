from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
import torch
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
        self.best_model_state_dict = None

    def step(self, val_loss, model_state_dict):
        if val_loss < self.best_loss:
            if self.verbose > 0:
                print(f"Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
            self.best_loss = val_loss
            self.best_model_state_dict = model_state_dict
            torch.save(model_state_dict.state_dict(), self.filepath)

class BiLSTM(nn.Module):
    def __init__(self, sequence_length):
 
        super(BiLSTM, self).__init__()
        self.seq_length = sequence_length
        self.conv = nn.Conv1d(1, 16, 4, stride = 1, padding= 'same')
        self.lstm_1 = nn.LSTM(input_size = 16, hidden_size = 64, batch_first = True, bidirectional = True)
        self.lstm_2 = nn.LSTM(input_size = 2*64, hidden_size = 128, batch_first = True, bidirectional = True)
        self.fc_1 = nn.Linear(self.seq_length * 128 * 2,128)
        self.act = nn.Tanh()
        self.fc_2 = nn.Linear(128,1)
         


    def forward(self, x):
        # print('x: ', x.dtype)
        conved_x = self.conv(x)
        # print('conved_x: ', conved_x.shape)
        lstm_out_1,_ = self.lstm_1(conved_x.permute(0,2,1))
        # print('lstm_out_1: ', lstm_out_1.shape)
        lstm_out_2,_ = self.lstm_2(lstm_out_1)
        # print('lstm_out_2: ', lstm_out_2.shape)
        out = self.fc_2(self.act(self.fc_1(lstm_out_2.reshape(-1,self.seq_length * 128 * 2))))
        return out

class RNN(Disaggregator):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """

        self.MODEL_NAME = "RNN"
        self.models = OrderedDict()
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',1800)
        self.mains_std = params.get('mains_std',600)
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)


    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print("...............RNN partial_fit running...............")
        # Do the pre-processing, such as  windowing and normalizing
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape(( -1, 1 ))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network(self.sequence_length)
            # Retrain the particular appliance
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = self.file_prefix + "-{}-epoch{}.pth".format(
                            "_".join(appliance_name.split()),
                            current_epoch,
                    )
                    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)

                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)
                    train_dataset = TensorDataset(torch.tensor(train_x).permute(0, 2, 1), torch.tensor(train_y))
                    val_dataset = TensorDataset(torch.tensor(v_x).permute(0, 2, 1), torch.tensor(v_y))
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

                    self.train_model(model, train_loader, val_loader, checkpoint)
                    self.models[appliance_name].load_state_dict(torch.load(filepath))
    

    def train_model(self, model, train_loader, val_loader, checkpoint):
        criterion = nn.MSELoss(reduction='mean')
        optimizer = Adam(model.parameters())

        for epoch in range(self.n_epochs):
            model.train()
            running_loss = 0.0
            for i , (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                # print('inputs: ', inputs.shape, 'outputs: ', outputs.shape, 'lables: ',labels.shape)
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

    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model

        # Preprocess the test mains such as windowing and normalizing

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            test_main_array = torch.tensor(test_main).permute(0, 2, 1)
            disggregation_dict = {}
            for appliance in self.models:
                prediction = []
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_array)
                prediction = prediction.numpy()
                prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self, sequence_length):
        '''Creates the RNN module described in the paper
        '''
        model = BiLSTM(sequence_length)
        return model
    

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()

                processed_appliance_dfs = []

                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    # This is for choosing windows
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    # Return as a list of dataframe
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list

        else:
            mains_df_list = []

            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self,train_appliances):
        # Find the parameters using the first
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
        print (self.appliance_params)
