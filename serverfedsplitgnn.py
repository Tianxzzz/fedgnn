import os
import pickle
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import util
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
from models import TCN
import torch.optim as optim
from standalone import unscaled_metrics
import time
import torch.nn.functional as F
from copy import deepcopy
import base_models
from collections import defaultdict
import itertools

class SplitFedNodePredictorClient(nn.Module):
    def __init__(self, base_model_name, optimizer_name, train_dataset, val_dataset, test_dataset, feature_scaler,
                 sync_every_n_epoch, lr, weight_decay, batch_size, client_device, start_global_step,
                 *args, **kwargs):
        super().__init__()
        self.base_model_name = base_model_name
        self.optimizer_name = optimizer_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_scaler = feature_scaler
        self.sync_every_n_epoch = sync_every_n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.base_model_kwargs = kwargs
        self.device = client_device
        self.base_model_class = getattr(base_models, self.base_model_name)
        self.init_base_model(None)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        if self.val_dataset:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.val_dataloader = self.train_dataloader
        if self.test_dataset:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_dataloader = self.train_dataloader
        self.global_step = start_global_step

    def forward(self, x, server_graph_encoding):
        return self.base_model(x, self.global_step, server_graph_encoding=server_graph_encoding)

    def init_base_model(self, state_dict):
        self.base_model = self.base_model_class(**self.base_model_kwargs).to(self.device)
        if state_dict is not None:
            self.base_model.load_state_dict(state_dict)

        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def local_train(self, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.train()
        with torch.enable_grad():
            for epoch_i in range(self.sync_every_n_epoch):
                num_samples = 0
                epoch_log = defaultdict(lambda: 0.0)
                for batch in self.train_dataloader:
                    x, y, server_graph_encoding = batch
                    server_graph_encoding = server_graph_encoding.permute(1, 0, 2, 3)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    server_graph_encoding = server_graph_encoding.to(self.device)
                    data = dict(x=x, y=y)
                    y_pred = self(data, server_graph_encoding)
                    loss = nn.MSELoss()(y_pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    num_samples += x.shape[0]
                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 'train')
                    epoch_log['train/loss'] += loss.detach() * x.shape[0]
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0]
                    self.global_step += 1
                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()
        # self.cpu()
        state_dict = self.base_model.to('cpu').state_dict()
        epoch_log['num_samples'] = num_samples
        epoch_log['global_step'] = self.global_step
        epoch_log = dict(**epoch_log)

        return {
            'state_dict': state_dict, 'log': epoch_log
        }

    def local_eval(self, dataloader, name, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.eval()
        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda: 0.0)

            for batch in dataloader:

                x, y, server_graph_encoding = batch
                server_graph_encoding = server_graph_encoding.permute(1, 0, 2, 3)
                x = x.to(self.device)
                y = y.to(self.device)
                server_graph_encoding = server_graph_encoding.to(self.device)
                data = dict(x=x, y=y)
                y_pred = self(data, server_graph_encoding)
                loss = nn.MSELoss()(y_pred, y)
                num_samples += x.shape[0]
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler, name)
                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0]
                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]

            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()

        # self.cpu()

        epoch_log['num_samples'] = num_samples

        epoch_log = dict(**epoch_log)

        return {'log': epoch_log}

    def local_validation(self, state_dict_to_load):
        return self.local_eval(self.val_dataloader, 'val', state_dict_to_load)

    def local_test(self, state_dict_to_load):
        return self.local_eval(self.test_dataloader, 'test', state_dict_to_load)

    @staticmethod
    def client_local_execute(device, state_dict_to_load, order, hparams_list):

        torch.cuda.set_device(device)
        res_list = []
        for hparams in hparams_list:
            client = SplitFedNodePredictorClient(client_device=device, **hparams)
            if order == 'train':
                res = client.local_train(state_dict_to_load)
            elif order == 'val':
                res = client.local_validation(state_dict_to_load)
            elif order == 'test':
                res = client.local_test(state_dict_to_load)
            else:
                del client
                raise NotImplementedError()
            del client
            res_list.append(res)
        return res_list


def setup(self, step):
    # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
    if self.base_model is not None:
        return
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
    train_sample_num=data['x_train'].shape[0]
    shufflerng = np.random.RandomState(42)
    shuffled_train_idx = shufflerng.permutation(train_sample_num).astype(int)

    self.clients = []
    for name in ['x', 'y']:
        data[name+'_train']= (data[name+'_train'])[shuffled_train_idx]

    # print(data['x_train'].shape)(23974, 12, 207, 2)  (S,T,N,F)
    self.num_clients = data['x_train'].shape[2]
    # Each node (client) has its own model and optimizer
    #
    client_params_list = []
    for client_i in range(self.num_clients):
        client_datasets = {}
        for name in ['train', 'val', 'test']:
            client_datasets[name] = TensorDataset(
                data['x_'+name][:, :, client_i:client_i + 1, :],
                data['y_'+name][:, :, client_i:client_i + 1, :],
                torch.zeros(1, data['x_'+name].shape[0], 12, 2).float().permute(1, 0, 2, 3)) # default_server_graph_encoding
        client_params = {}
        client_params.update(
            optimizer_name='Adam',
            train_dataset=client_datasets['train'],
            val_dataset=client_datasets['val'],
            test_dataset=client_datasets['test'],
            feature_scaler=data['feature_scaler'],
            start_global_step=0,
            **self.hparams
        )
        client_params_list.append(client_params)
    self.client_params_list = client_params_list

    self.base_model = getattr(base_models, self.hparams.base_model_name)(input_size=input_size, output_size=output_size, **self.hparams)

    self.gcn = GraphNets(

        node_input_size=self.hparams.hidden_size,
        edge_input_size=1,
        global_input_size=self.hparams.hidden_size,
        hidden_size=256,
        updated_node_size=128,
        updated_edge_size=128,
        updated_global_size=128,
        node_output_size=self.hparams.hidden_size,
        # gn_layer_num=2,
        gn_layer_num=self.hparams.server_gn_layer_num,
        activation='ReLU', dropout=self.hparams.dropout)

    self.server_optimizer = getattr(torch.optim, 'Adam')(self.gcn.parameters(), lr=self.hparams.lr,
                                                         weight_decay=self.hparams.weight_decay)

    self.server_datasets = {}
    for name in ['train', 'val', 'test']:
        self.server_datasets[name] = TensorDataset(
            data['x_'+name], data['y_'+name])

def _train_server_gcn_with_agg_clients(self, device):
    # here we assume all clients are aggregated! Simulate running on clients with the aggregated copy on server
    # this only works when (1) clients are aggregated and (2) no optimization on client models
    self.base_model.to(device)
    self.gcn.to(device)
    server_train_dataloader = DataLoader(self.server_datasets['train'], batch_size=self.hparams.server_batch_size,shuffle=True)
    updated_graph_encoding = None
    global_step = self.client_params_list[0]['start_global_step']
    with torch.enable_grad():
        self.base_model.train()
        self.gcn.train()
        for epoch_i in range(self.hparams.server_epoch + 1):
            updated_graph_encoding = []
            if epoch_i == self.hparams.server_epoch:
                server_train_dataloader = DataLoader(self.server_datasets['train'],batch_size=self.hparams.server_batch_size, shuffle=False)
            for batch in server_train_dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                data = dict(x=x, y=y)
                h_encode = self.base_model.forward_encoder(data)  # L x (B x N) x F
                batch_num, node_num = data['x'].shape[0], data['x'].shape[2]
                graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2,1,0,3)  # N x B x L x F
                graph_encoding = self.gcn(Data(x=graph_encoding,edge_index=self.data['train']['edge_index'].to(graph_encoding.device)))  # N x B x L x F
                if epoch_i == self.hparams.server_epoch:
                    updated_graph_encoding.append(graph_encoding.detach().clone().cpu())
                else:
                    y_pred = self.base_model.forward_decoder(data, h_encode, batches_seen=global_step, return_encoding=False, server_graph_encoding=graph_encoding)
                    loss = nn.MSELoss()(y_pred, y)
                    self.server_optimizer.zero_grad()
                    loss.backward()
                    self.server_optimizer.step()
                    global_step += 1
    # update global step for all clients
    for client_params in self.client_params_list:
        client_params.update(start_global_step=global_step)
    # update server_graph_encoding
    updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N x B x L x F
    sel_client_i = 0
    for client_i, client_params in enumerate(self.client_params_list):

        client_params.update(train_dataset=TensorDataset(
            self.data['train']['x'][:, :, client_i:client_i + 1, :],
            self.data['train']['y'][:, :, client_i:client_i + 1, :],
            updated_graph_encoding[sel_client_i:sel_client_i + 1, :, :, :].permute(1, 0, 2, 3)
        ))
        sel_client_i += 1

def _eval_server_gcn_with_agg_clients(self, name, device):
    assert name in ['val', 'test']
    self.base_model.to(device)
    self.gcn.to(device)
    server_dataloader = DataLoader(self.server_datasets[name], batch_size=self.hparams.server_batch_size,shuffle=False)
    updated_graph_encoding = []
    with torch.no_grad():
        self.base_model.eval()
        self.gcn.eval()
        for batch in server_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            data = dict(x=x, y=y)
            h_encode = self.base_model.forward_encoder(data)  # L x (B x N) x F
            batch_num, node_num = data['x'].shape[0], data['x'].shape[2]
            graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0,3)  # N x B x L x F
            graph_encoding = self.gcn(Data(x=graph_encoding,edge_index=self.data[name]['edge_index'].to(graph_encoding.device)))  # N x B x L x F
            updated_graph_encoding.append(graph_encoding.detach().clone().cpu())
    # update server_graph_encoding
    updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N x B x L x F
    for client_i, client_params in enumerate(self.client_params_list):
        keyname = '{}_dataset'.format(name)
        client_params.update({
            keyname: TensorDataset(
                self.data[name]['x'][:, :, client_i:client_i + 1, :],
                self.data[name]['y'][:, :, client_i:client_i + 1, :],
                updated_graph_encoding[client_i:client_i + 1, :, :, :].permute(1, 0, 2, 3)
            )
        })

def train_dataloader(self):
    # return a fake dataloader for running the loop
    return DataLoader([0, ])

def val_dataloader(self):
    return DataLoader([0, ])

def test_dataloader(self):
    return DataLoader([0, ])


def training_step(self, batch, batch_idx):
    # 1. train locally and collect uploaded local train results
    local_train_results = []
    server_device = next(self.gcn.parameters()).device
    self.base_model.to('cpu')
    self.gcn.to('cpu')
    for client_i, client_params in enumerate(self.client_params_list):
        if 'selected' in self.data['train']:
            if self.data['train']['selected'][client_i, 0].item() is False:
                continue
        local_train_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(self.base_model.state_dict()), 'train', **client_params)
        local_train_results.append(local_train_result)

    # update global steps for all clients
    for ltr, client_params in zip(local_train_results, self.client_params_list):
        client_params.update(start_global_step=ltr['log']['global_step'])
    # 2. aggregate (optional? kept here to save memory, otherwise need to store 1 model for each node)
    agg_local_train_results = self.aggregate_local_train_results(local_train_results)
    # 2.1. update aggregated weights

    if agg_local_train_results['state_dict'] is not None:
        self.base_model.load_state_dict(agg_local_train_results['state_dict'])

    # TODO: 3. train GNN on server in split learning way (optimize server-side params only)
    # client_local_execute, return_encoding
    # run decoding on all clients, run backward on all clients
    # run backward on server-side GNN and optimize GNN
    # TODO: 4. run forward on updated GNN to renew server_graph_encoding

    self._train_server_gcn_with_agg_clients(server_device)
    agg_log = agg_local_train_results['log']
    log = agg_log
    return {'loss': torch.tensor(0).float(), 'progress_bar': log, 'log': log}


def aggregate_local_train_results(self, local_train_results):
    return {
        'state_dict': self.aggregate_local_train_state_dicts(
            [ltr['state_dict'] for ltr in local_train_results]
        ),
        'log': self.aggregate_local_logs(
            [ltr['log'] for ltr in local_train_results]
        )
    }

def aggregate_local_logs(self, local_logs):
    agg_log = deepcopy(local_logs[0])
    for k in agg_log:
        agg_log[k] = 0
        for local_log_idx, local_log in enumerate(local_logs):
            if k == 'num_samples':
                agg_log[k] += local_log[k]
            else:
                agg_log[k] += local_log[k] * local_log['num_samples']
    for k in agg_log:
        if k != 'num_samples':
            agg_log[k] /= agg_log['num_samples']


    return agg_log

def training_epoch_end(self, outputs):
    # already averaged!
    log = outputs[0]['log']
    log.pop('num_samples')
    return {'log': log, 'progress_bar': log}

def validation_step(self, batch, batch_idx):
    server_device = next(self.gcn.parameters()).device
    self._eval_server_gcn_with_agg_clients('val', server_device)
    # 1. vaidate locally and collect uploaded local train results
    local_val_results = []
    self.base_model.to('cpu')
    self.gcn.to('cpu')

    for client_i, client_params in enumerate(self.client_params_list):
        local_val_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(
            self.base_model.state_dict()), 'val', **client_params)
        local_val_results.append(local_val_result)

    self.base_model.to(server_device)
    self.gcn.to(server_device)
    # 2. aggregate
    log = self.aggregate_local_logs([x['log'] for x in local_val_results])
    return {'progress_bar': log, 'log': log}

def validation_epoch_end(self, outputs):
    return self.training_epoch_end(outputs)

def test_step(self, batch, batch_idx):
    server_device = next(self.gcn.parameters()).device
    self._eval_server_gcn_with_agg_clients('test', server_device)
    # 1. vaidate locally and collect uploaded local train results
    local_val_results = []
    self.base_model.to('cpu')
    self.gcn.to('cpu')

    for client_i, client_params in enumerate(self.client_params_list):
        local_val_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(
            self.base_model.state_dict()), 'test', **client_params)
        local_val_results.append(local_val_result)

    self.base_model.to(server_device)
    self.gcn.to(server_device)
    # 2. aggregate
    # separate seen and unseen nodes if necessary
    if 'selected' in self.data['train']:
        log = self.aggregate_local_logs([x['log'] for x in local_val_results], self.data['train']['selected'])
    else:
        log = self.aggregate_local_logs([x['log'] for x in local_val_results])
    return {'progress_bar': log, 'log': log}

def test_epoch_end(self, outputs):
    return self.training_epoch_end(outputs)


def aggregate_local_train_state_dicts(self, local_train_state_dicts):
    agg_state_dict = {}
    for k in local_train_state_dicts[0]:
        agg_state_dict[k] = 0
        for ltsd in local_train_state_dicts:
            agg_state_dict[k] += ltsd[k]
        agg_state_dict[k] /= len(local_train_state_dicts)
    return agg_state_dict