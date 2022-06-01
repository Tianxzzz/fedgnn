import os
import pickle
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import models
import util
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
from models import TCN,client_model,graphmodel
import torch.optim as optim
from standalone import unscaled_metrics
import time
import torch.nn.functional as F
from copy import deepcopy
from model import STGCN
import copy
import util
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_tensor_type(torch.DoubleTensor)
class Client_side(nn.Module):

    def __init__(self,  feature_scaler, batch_size,
        *args, **kwargs):
        super().__init__()

        self.feature_scaler = feature_scaler
        self.batch_size = batch_size
        self.base_model = client_model().to('cuda')
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=0.001, weight_decay=0.0001)

    def forward(self, x, stage):
        if stage == 'enc':
            for k in x:
                x[k] = x[k].to(next(self.parameters()).device)
            return self.base_model.forward_encoder(x)
        elif stage == 'dec':
            for k in x['data']:
                x['data'][k] = x['data'][k].to(next(self.parameters()).device)
            for k in ['h_encode', 'server_graph_encoding']:
                x[k] = x[k].to(next(self.parameters()).device)
            data = x['data']
            h_encode = x['h_encode']

            server_graph_encoding = x['server_graph_encoding']
            return self.base_model.forward_decoder(data, h_encode, False, server_graph_encoding)
        else:
            raise NotImplementedError()

    def local_encode_forward(self, x , y, name):
        if name=='train':
            self.base_model.train()
        else:
            self.base_model.eval()
        y = y.to(next(self.parameters()).device)
        self.encoding = self({'x': x, 'y': y}, stage='enc') # L x (B x N) x F
        return self.encoding

    def local_decode_forward(self, x,y,server_graph_encoding):

        self.server_graph_encoding = server_graph_encoding.clone().detach().requires_grad_(True)
        self.server_graph_encoding.retain_grad()
        self.y_pred = self({'data': {'x':x ,'y':y} ,
            'h_encode': self.encoding,
            'server_graph_encoding': self.server_graph_encoding}, stage='dec')

    def local_backward(self, y,grads=None, stage='dec'):
        if stage == 'dec':
            self.y_pred = self.feature_scaler.inverse_transform(self.y_pred)
            # print(self.y.shape,self.y_pred.shape)
            loss = util.masked_mae(self.y_pred, y[:,:,0].unsqueeze(-1),0.0)
            loss.backward(retain_graph=True)
            mape = util.masked_mape(self.y_pred, y[:,:,0].unsqueeze(-1), 0.0).item()
            rmse = util.masked_rmse(self.y_pred, y[:,:,0].unsqueeze(-1), 0.0).item()

            return {
                'train_loss': loss.item(),'train_mape':mape, 'train_rmse':rmse , 'grad': self.server_graph_encoding.grad
            }
        elif stage == 'enc':
            self.encoding.backward(grads)
        else:
            raise NotImplementedError()

    def local_optimizer_step(self):
        self.optimizer.step()

    def local_optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def local_validation(self, y):

        self.y_pred = self.feature_scaler.inverse_transform(self.y_pred)
        loss = util.masked_mae(self.y_pred, y[:, :, 0].unsqueeze(-1), 0.0)
        mape = util.masked_mape(self.y_pred, y[:, :, 0].unsqueeze(-1), 0.0).item()
        rmse = util.masked_rmse(self.y_pred, y[:, :, 0].unsqueeze(-1), 0.0).item()
        return {
            'val_loss': loss.item(), 'val_mape': mape, 'val_rmse': rmse
        }
    def local_test(self, y):

        self.y_pred = self.feature_scaler.inverse_transform(self.y_pred)
        loss = util.masked_mae(self.y_pred, y[:, :, 0].unsqueeze(-1), 0.0)
        mape = util.masked_mape(self.y_pred, y[:, :, 0].unsqueeze(-1), 0.0).item()
        rmse = util.masked_rmse(self.y_pred, y[:, :, 0].unsqueeze(-1), 0.0).item()
        return {
            'test_loss': loss.item(), 'test_mape': mape, 'test_rmse': rmse
        }

def aggregate_local_train_state_dicts(local_train_state_dicts):

    agg_state_dict = copy.deepcopy(local_train_state_dicts[0])
    for k in agg_state_dict.keys():
        for i in range(1, len(local_train_state_dicts)):
            agg_state_dict[k] += local_train_state_dicts[i][k]
        agg_state_dict[k] = torch.true_divide(agg_state_dict[k], len(local_train_state_dicts))
    return agg_state_dict

def aggregate_local_logs(local_logs):
    agg_log = copy.deepcopy(local_logs[0])
    for k in agg_log:
        agg_log[k] = 0
        for local_log in local_logs:
            if k == 'num_samples':
                agg_log[k] += local_log[k]
            else:
                agg_log[k] += local_log[k] * local_log['num_samples']
    for k in agg_log:
        if k != 'num_samples':
            agg_log[k] /= agg_log['num_samples']
    return agg_log

def setup():

    device = torch.device('cuda')
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj('data/sensor_graph/adj_mx.pkl', 'scalap')
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join('data/METR-LA', category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = util.StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    # shuffle trainset in advance with a fixed seed (42)
    data['train_loader'] = util.DataLoader(data['x_train'], data['y_train'], 32)
    data['val_loader'] = util.DataLoader(data['x_val'], data['y_val'], 32)
    data['test_loader'] = util.DataLoader(data['x_test'], data['y_test'], 32)
    data['scaler'] = scaler

    clients = []
    for client_i in range(207):
        client = Client_side(feature_scaler=scaler,batch_size=32)
        clients.append(client)
    clients = nn.ModuleList(clients)
    prop_model = graphmodel(num_nodes=207).to(device)
    prop_model_optimizer = torch.optim.Adam(prop_model.parameters(), lr=0.001, weight_decay=0.0001)

    num_clients=207

    val_time = []
    train_time = []
    for epoch in range(100):
        train_loss = {}
        train_mape = {}
        train_rmse = {}
        mtrain_loss = []
        mtrain_mape = []
        mtrain_rmse = []
        valid_loss = {}
        valid_mape = {}
        valid_rmse = {}
        mvalid_loss = []
        mvalid_mape = []
        mvalid_rmse = []
        t1 = time.time()
        for name in range(num_clients):
            train_loss[str(name)] = []
            train_mape[str(name)] = []
            train_rmse[str(name)] = []
            valid_loss[str(name)] = []
            valid_mape[str(name)] = []
            valid_rmse[str(name)] = []
        for iter, dataset in enumerate(data['train_loader'].get_iterator()):
            prop_model.train()
            (x, y) = dataset
            trainx = torch.Tensor(x).to(device)
            # print(trainx.shape) ([64, 12, 207, 2])
            trainy = torch.Tensor(y).to(device)
            for client_i, client in enumerate(clients):
                client.local_optimizer_zero_grad()
            prop_model_optimizer.zero_grad()
            encodings = []  # list of L x (B x 1) x F
            for client_i, client in enumerate(clients):
                encodings.append(client.local_encode_forward(trainx[:,:,client_i,:],trainy[:,:,client_i,:],'train'))

            stacked_encodings = torch.cat(encodings, dim=2)  # L x B x N x F
            # print(stacked_encodings.shape) #torch.Size([64, 32, 207, 12])
            # data = nn.functional.pad(stacked_encodings, (1, 0, 0, 0))
            stacked_encodings = stacked_encodings.permute(0, 2, 3, 1) #torch.Size([64, 207, 12, 32])
            # keep gradients
            stacked_encodings = stacked_encodings.clone().detach().requires_grad_(True)
            stacked_encodings.retain_grad()

            # 2. run propagation with the known graph structure and collected embeddings
            hiddens = prop_model(stacked_encodings) #([64, 207, 2, 64])

            # 3. run decoding on all clients
            for client_i, client in enumerate(clients):
                client.local_decode_forward(trainx[:,:,client_i,:],trainy[:,:,client_i,:],hiddens[:,:,client_i,:])
            # 4. run zero_grad for all optimizers


            hiddens_msg_grad=[]
            # 4. run backward on all clients
            for client_i, client in enumerate(clients):
                local_train_result = client.local_backward(trainy[:,:,client_i,:],stage='dec')
                train_loss[str(client_i)].append(local_train_result['train_loss'])
                train_mape[str(client_i)].append(local_train_result['train_mape'])
                train_rmse[str(client_i)].append(local_train_result['train_rmse'])
                # hiddens_msg_grad.append(local_train_result['grad'])
            # hiddens_msg_grad = torch.stack(hiddens_msg_grad, dim=1)
            # 4.1 run backward on server graph model
            # hiddens.backward(hiddens_msg_grad.transpose(1,2))
            # 4.2 collect grads on data and run backward on clients
            # for client_i, client in enumerate(clients):
            #     if stacked_encodings.grad is not None:
            #         data_grads= stacked_encodings.grad[:,client_i,:,:].unsqueeze(2)
            #         data_grads = data_grads.permute(0, 3, 2, 1)
            #     else:
            #         data_grads = None
                # client.local_backward(trainy[:,:,client_i,:], grads=data_grads, stage='enc')
            # 5. run optimizers on the server and all clients
            prop_model_optimizer.step()
            for client_i, client in enumerate(clients):
                client.local_optimizer_step()
            # agg_log = aggregate_local_logs(local_train_logs)
            if iter % 50 == 0:
                log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(id, log.format(iter, train_loss['0'][iter], train_mape['0'][iter], train_rmse['0'][iter]), flush=True)
        t2 = time.time()

        train_time.append(t2 - t1)

        s1 = time.time()
        for iter, dataset in enumerate(data['val_loader'].get_iterator()):
            prop_model.eval()
            (x, y) = dataset
            val_x = torch.Tensor(x).to(device)
            val_y = torch.Tensor(y).to(device)
            encodings = []  # list of L x (B x 1) x F
            for client_i, client in enumerate(clients):
                encodings.append(client.local_encode_forward(val_x[:, :, client_i, :], val_y[:, :, client_i, :], 'val'))
            stacked_encodings = torch.cat(encodings, dim=2)
            stacked_encodings = stacked_encodings.permute(0, 2, 3, 1)
            hiddens = prop_model(stacked_encodings)
            for client_i, client in enumerate(clients):
                client.local_decode_forward(val_x[:, :, client_i, :], val_y[:, :, client_i, :], hiddens[:, :, client_i, :])
                # 4. run eval on all clients

            for client_i, client in enumerate(clients):
                local_val_result = client.local_validation(val_y[:, :, client_i, :])
                valid_loss[str(client_i)].append(local_val_result['val_loss'])
                valid_mape[str(client_i)].append(local_val_result['val_mape'])
                valid_rmse[str(client_i)].append(local_val_result['val_rmse'])
        s2 = time.time()
        val_time.append(s2 - s1)
        # train_time.append(t2 - t1)
        for name in range(207):
            mtrain_loss.append(np.mean(train_loss[str(name)]))
            mtrain_mape.append(np.mean(train_mape[str(name)]))
            mtrain_rmse.append(np.mean(train_rmse[str(name)]))
            mvalid_loss.append(np.mean(valid_loss[str(name)]))
            mvalid_mape.append(np.mean(valid_mape[str(name)]))
            mvalid_rmse.append(np.mean(valid_rmse[str(name)]))
        mmtrain_loss = np.mean(mtrain_loss)
        mmtrain_mape = np.mean(mtrain_mape)
        mmtrain_rmse = np.mean(mtrain_rmse)
        mmvalid_loss = np.mean(mvalid_loss)
        mmvalid_mape = np.mean(mvalid_mape)
        mmvalid_rmse = np.mean(mvalid_rmse)
        # his_loss.append(mvalid_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(id, log.format(epoch, mmtrain_loss, mmtrain_mape, mmtrain_rmse, mmvalid_loss, mmvalid_mape, mmvalid_rmse,(t2 - t1)),flush=True)
        # torch.save(self.modelA1.state_dict(),args.saveA1 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")

        if epoch%5==0:
            agg_state_dict = aggregate_local_train_state_dicts([client.state_dict() for client in clients])

            for client_i, client in enumerate(clients):
                client.load_state_dict(deepcopy(agg_state_dict))

setup()

