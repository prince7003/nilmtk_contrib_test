from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
random.seed(10)
np.random.seed(10)

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        maxlen = x.size(1)
        positions = torch.arange(0, maxlen, device=x.device).expand_as(x)
        x = self.token_emb(x) + self.pos_emb(positions)
        return x

class LPpool(nn.Module):
    def __init__(self, pool_size, strides=None, padding='same'):
        super(LPpool, self).__init__()
        self.avgpool = nn.AvgPool1d(pool_size, stride=strides, padding=padding)

    def forward(self, x):
        x = torch.pow(torch.abs(x), 2)
        x = self.avgpool(x)
        x = torch.pow(x, 1.0 / 2)
        return x

class BERT(Disaggregator):
    def __init__(self, params):
        super(BERT, self).__init__()
        self.MODEL_NAME = "BERT"
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

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        print("...............BERT partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')
        train_main = torch.tensor(np.concatenate(train_main, axis=0), dtype=torch.float32)
        train_main = train_main.view(-1, self.sequence_length, 1)

        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = np.concatenate(app_dfs, axis=0)
            app_df = torch.tensor(app_df, dtype=torch.float32)
            app_df = app_df.view(-1, self.sequence_length)
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size(0) > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = 'BERT-temp-weights-' + str(random.randint(0, 100000)) + '.pth'
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=0.15, random_state=10)
                    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
                    val_dataset = torch.utils.data.TensorDataset(v_x, v_y)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                    self.train_model(model, train_loader, val_loader, filepath)

    def train_model(self, model, train_loader, val_loader, filepath):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

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
            print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {epoch_loss}")

            val_loss = self.validate_model(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss}")

            if epoch == 0 or val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), filepath)

    def validate_model(self, model, val_loader, criterion):
        model.eval()
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
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = torch.tensor(test_mains_df.values, dtype=torch.float32)
            test_main_array = test_main_array.view(-1, self.sequence_length, 1)

            for appliance in self.models:
                prediction = []
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_array).numpy()

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
        embed_dim = 32
        num_heads = 2
        ff_dim = 32
        vocab_size = 20000
        maxlen = self.sequence_length

        model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=2),
            LPpool(pool_size=2),
            TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim),
            TransformerBlock(embed_dim, num_heads, ff_dim),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * maxlen, self.sequence_length),
            nn.Dropout(0.1)
        )
        print(model)
        return model

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
                processed_mains_lst.append(torch.tensor(new_mains, dtype=torch.float32))
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print("Parameters for ", app_name, " were not found!")
                    raise ApplianceNotFoundError()

                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant',
                                              constant_values=(0, 0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in
                                                 range(len(new_app_readings) - n + 1)])
                    new_app_readings = (new_app_readings - app_mean) / app_std
                    processed_app_dfs.append(torch.tensor(new_app_readings, dtype=torch.float32))

                appliance_list.append((app_name, processed_app_dfs))

            return processed_mains_lst, appliance_list
        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(torch.tensor(new_mains, dtype=torch.float32))
            return processed_mains_lst

    def set_appliance_params(self, train_appliances):
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
