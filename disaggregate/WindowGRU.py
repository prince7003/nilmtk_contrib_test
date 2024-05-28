from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from sklearn.model_selection import train_test_split

class BiGRU(nn.Module):
    def __init__(self, sequence_length):
        super(BiGRU, self).__init__()
        self.seq_length = sequence_length
        
        # conv1d layer
        self.conv = nn.Conv1d(1, 16, 4, stride = 1, padding= 'same')
        # Bi-directional GRUs
        self.GRU_1 = nn.GRU(input_size = 16, hidden_size = 64, batch_first = True, bidirectional = True)
        self.dp1 = nn.Dropout(0.5)
        self.GRU_2 = nn.GRU(input_size = 2*64, hidden_size = 128, batch_first = True, bidirectional = True)
        self.dp2 = nn.Dropout(0.5)
        # Fully Connected Layers
        self.fc_1 = nn.Linear(self.seq_length * 128 * 2,128)
        self.act = nn.ReLU()
        self.dp3 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(128,1)
         

    def forward(self, x):
        # print('x: ', x.dtype)
        conved_x = self.conv(x)
        # print('conved_x: ', conved_x.shape)
        GRU_out_1,_ = self.GRU_1(conved_x.permute(0,2,1))
        dp1_out = self.dp1(GRU_out_1)
        # print('lstm_out_1: ', lstm_out_1.shape)
        GRU_out_2,_ = self.GRU_2(dp1_out)
        dp2_out = self.dp2(GRU_out_2)
        # print('lstm_out_2: ', lstm_out_2.shape)
        out = self.fc_2(self.dp3(self.act(self.fc_1(dp2_out.reshape(-1,self.seq_length * 128 * 2)))))
        return out

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

class WindowGRU(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "WindowGRU"
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path', None)
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 19)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.max_val = 800
        self.batch_size = params.get('batch_size', 512)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0).values
        train_main = train_main.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0).values
            app_df = app_df.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df))

        train_appliances = new_train_appliances
        for app_name, app_df in train_appliances:
            if app_name not in self.models:
                print("First model training for", app_name)
                self.models[app_name] = self.return_network()
            else:
                print("Started re-training model for", app_name)

            model = self.models[app_name]
            mains = train_main.reshape((-1, self.sequence_length, 1))
            app_reading = app_df.reshape((-1, 1))
            filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                            "_".join(app_name.split()),
                            current_epoch,
                    )
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)

            train_x, v_x, train_y, v_y = train_test_split(train_main, app_df, test_size=.15, random_state=10)
            train_dataset = TensorDataset(torch.tensor(train_x).permute(0, 2, 1).float(), torch.tensor(train_y).float())
            val_dataset = TensorDataset(torch.tensor(v_x).permute(0, 2, 1).float(), torch.tensor(v_y).float())
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            self.train_model(model, train_loader, val_loader, checkpoint)

    def train_model(self, model, train_loader, val_loader, checkpoint):
        criterion = nn.MSELoss(reduction='mean')
        optimizer = Adam(model.parameters())

        for epoch in range(self.n_epochs):
            model.train()
            running_loss = 0.0
            for i , (inputs, labels) in enumerate(train_loader):
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
                checkpoint.step(val_loss, model.state_dict())
            else:
                print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {epoch_loss}")

        # Load the best model state dict after training
        checkpoint.load_best_model(model)

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
        for mains in test_main_list:
            disggregation_dict = {}
            mains = mains.values.reshape((-1, self.sequence_length, 1))
            test_main_array = torch.tensor(mains).permute(0, 2, 1).float()
            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_array)
                prediction = prediction.numpy()
                prediction = np.reshape(prediction, len(prediction))
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                valid_predictions = self._denormalize(valid_predictions, self.max_val)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        max_val = self.max_val
        if method == 'train':
            print("Training processing")
            processed_mains = []

            for mains in mains_lst:
                # add padding values
                padding = [0 for i in range(0, self.sequence_length - 1)]
                paddf = pd.DataFrame({mains.columns.values[0]: padding})
                mains = pd.concat([mains,paddf])
                mainsarray = self.preprocess_train_mains(mains)
                processed_mains.append(pd.DataFrame(mainsarray))

            tuples_of_appliances = []
            for (appliance_name, app_dfs_list) in submeters_lst:
                processed_app_dfs = []
                for app_df in app_dfs_list:
                    data = self.preprocess_train_appliances(app_df)
                    processed_app_dfs.append(pd.DataFrame(data))
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method == 'test':
            processed_mains = []
            for mains in mains_lst:
                # add padding values
                padding = [0 for i in range(0, self.sequence_length - 1)]
                paddf = pd.DataFrame({mains.columns.values[0]: padding})
                mains = pd.concat([mains,paddf])
                mainsarray = self.preprocess_test_mains(mains)
                processed_mains.append(pd.DataFrame(mainsarray))

            return processed_mains

    def preprocess_test_mains(self, mains):

        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        mainsarray = mainsarray.reshape((-1, self.sequence_length))
        return pd.DataFrame(mainsarray)

    def preprocess_train_appliances(self, appliance):

        appliance = self._normalize(appliance, self.max_val)
        appliancearray = np.array(appliance)
        appliancearray = appliancearray.reshape((-1, 1))
        return pd.DataFrame(appliancearray)

    def preprocess_train_mains(self, mains):

        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        mainsarray = mainsarray.reshape((-1, self.sequence_length))
        return pd.DataFrame(mainsarray)

    def _normalize(self, chunk, mmax):

        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):

        tchunk = chunk * mmax
        return tchunk

    def return_network(self):
        '''Creates the GRU architecture described in the paper
        '''
        model = BiGRU(self.sequence_length)
        return model
         
