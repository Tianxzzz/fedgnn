import os
import time
from argparse import ArgumentParser
import random
from copy import deepcopy
from collections import defaultdict
import itertools

import numpy as np
import torch
import torch.nn as nn

import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
from tqdm import tqdm

import util
# from play import EarlyStopping
from st_datasets import load_dataset
import base_model
from standalone import unscaled_metrics
from base_model.GraphNets import GraphNet
from base_model.GRUSeq2Seq import GRUSeq2Seq,GRUSeq2SeqWithGraphNet
from torch_geometric.utils import dense_to_sparse
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
class SplitFedNodePredictorClient(nn.Module):
    def __init__(self, train_dataset, val_dataset, test_dataset, feature_scaler,start_global_step,
        sync_every_n_epoch=1, lr=0.001, weight_decay=0.0, batch_size=128):
        super().__init__()
        self.base_model = GRUSeq2SeqWithGraphNet().to('cuda')
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_scaler = feature_scaler
        self.sync_every_n_epoch = sync_every_n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = 'cuda'
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
        # self.base_model = self.base_model.to('cuda')
        if state_dict is not None:
            self.base_model.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def local_train(self, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.train()
        with torch.enable_grad():
            for epoch_i in range(self.sync_every_n_epoch):
                num_samples = 0
                epoch_log = defaultdict(lambda : 0.0)
                for batch in self.train_dataloader:
                    x, y, x_attr, y_attr, server_graph_encoding = batch
                    server_graph_encoding = server_graph_encoding.permute(1, 0, 2, 3)
                    x = x.to(self.device) if (x is not None) else None
                    y = y.to(self.device) if (y is not None) else None
                    x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                    y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                    server_graph_encoding = server_graph_encoding.to(self.device)
                    data = dict(
                        x=x, x_attr=x_attr, y=y, y_attr=y_attr
                    )
                    self.optimizer.zero_grad()
                    y_pred = self(data, server_graph_encoding)
                    loss = util.masked_mae(y_pred,y,0.0)
                    loss.backward()
                    self.optimizer.step()
                    num_samples += x.shape[0]
                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 0.0)
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
        # print(epoch_log)
        return {
            'state_dict': state_dict, 'log': epoch_log
        }

    def local_eval(self, dataloader, name, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.eval()
        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda : 0.0)
            for batch in dataloader:
                x, y, x_attr, y_attr, server_graph_encoding = batch
                server_graph_encoding = server_graph_encoding.permute(1, 0, 2, 3)
                x = x.to(self.device) if (x is not None) else None
                y = y.to(self.device) if (y is not None) else None
                x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                server_graph_encoding = server_graph_encoding.to(self.device)
                data = dict(
                    x=x, x_attr=x_attr, y=y, y_attr=y_attr
                )
                y_pred = self(data, server_graph_encoding)
                loss = nn.MSELoss(y_pred,y,0.0)
                num_samples += x.shape[0]
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 0.0)
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
    def client_local_execute(state_dict_to_load, order, **hparams_list):

        res_list = []
        # for hparams in hparams_list:
        client = SplitFedNodePredictorClient(**hparams_list)
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

def setup():
    # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
    device='cuda'
    data = load_dataset(dataset_name='METR-LA')
    dataset = data
    # Each node (client) has its own model and optimizer
    # Assigning data, model and optimizer for each client
    num_clients = data['train']['x'].shape[2]
    input_size = dataset['train']['x'].shape[-1] + dataset['train']['x_attr'].shape[-1]
    output_size = dataset['train']['y'].shape[-1]
    # print(data['train']['x'].shape)  torch.Size([23974, 12, 207, 1])
    adj_mx_path = os.path.join('data/', 'sensor_graph', 'adj_mx.pkl')
    _, _, adj_mx = load_pickle(adj_mx_path)
    adj_mx_ts = torch.from_numpy(adj_mx).float()
    client_params_list = []
    for client_i in range(num_clients):
        client_datasets = {}
        for name in ['train', 'val', 'test']:
            client_datasets[name] = TensorDataset(
                data[name]['x'][:, :, client_i:client_i+1, :],
                data[name]['y'][:, :, client_i:client_i+1, :],
                data[name]['x_attr'][:, :, client_i:client_i+1, :],
                data[name]['y_attr'][:, :, client_i:client_i+1, :],
                torch.zeros(1, data[name]['x'].shape[0], 1, 64).float().permute(1, 0, 2, 3) # default_server_graph_encoding
            )
        client_params = {}
        client_params.update(
            train_dataset=client_datasets['train'],
            val_dataset=client_datasets['val'],
            test_dataset=client_datasets['test'],
            feature_scaler=dataset['feature_scaler'],
            # input_size=input_size,
            # output_size=output_size,
            start_global_step=0
        )
        client_params_list.append(client_params)

    client_model = GRUSeq2SeqWithGraphNet()
    server_model = GraphNet(
        node_input_size=64,
        edge_input_size=1,
        global_input_size=64,
        hidden_size=256,
        updated_node_size=128,
        updated_edge_size=128,
        updated_global_size=128,
        node_output_size=64,
        gn_layer_num=2,
        activation='ReLU', dropout=0
    )
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001, weight_decay=0.0)
    server_datasets = {}
    for name in ['train', 'val', 'test']:
        server_datasets[name] = TensorDataset(
        data[name]['x'], data[name]['y'],
            data[name]['x_attr'], data[name]['y_attr']
   )
        # 1. train locally and collect uploaded local train results

    for m in range(50):
        t1=time.time()
        local_train_results = []
        index = random.sample(range(0, 207), 20)
        for i in index:
            adj_mx_ts[[i], :] = 0
            adj_mx_ts[:, [i]] = 0
        train_edge_index, train_edge_attr = dense_to_sparse(adj_mx_ts)
        for client_i, client_params in enumerate(client_params_list):

            local_train_result = SplitFedNodePredictorClient.client_local_execute(deepcopy(client_model.state_dict()),'train', **client_params)
            local_train_results.append(local_train_result)

        # update global steps for all clients
        for ltr, client_params in zip(local_train_results, client_params_list):
            client_params.update(start_global_step=ltr[0]['log']['global_step'])

        # 2. aggregate (optional? kept here to save memory, otherwise need to store 1 model for each node)
        agg_local_train_results = aggregate_local_train_results(local_train_results)
        # 2.1. update aggregated weights
        if agg_local_train_results['state_dict'] is not None:
            client_model.load_state_dict(agg_local_train_results['state_dict'])
        # TODO: 3. train GNN on server in split learning way (optimize server-side params only)
        # client_local_execute, return_encoding
        # run decoding on all clients, run backward on all clients
        # run backward on server-side GNN and optimize GNN
        # TODO: 4. run forward on updated GNN to renew server_graph_encoding
        client_model.to(device)
        server_model.to(device)
        server_train_dataloader = DataLoader(server_datasets['train'], batch_size=48, shuffle=True)
        updated_graph_encoding = None
        global_step = client_params_list[0]['start_global_step']
        with torch.enable_grad():
            client_model.train()
            server_model.train()
            metric = []
            for epoch_i in range(2):
                updated_graph_encoding = []
                if epoch_i == 1:
                    server_train_dataloader = DataLoader(server_datasets['train'], batch_size=48, shuffle=False)
                for batch in server_train_dataloader:
                    x, y, x_attr, y_attr = batch
                    x = x.to(device) if (x is not None) else None
                    y = y.to(device) if (y is not None) else None
                    x_attr = x_attr.to(device) if (x_attr is not None) else None
                    y_attr = y_attr.to(device) if (y_attr is not None) else None
                    # if 'selected' in data['train']:
                    #     train_mask = data['train']['selected'].flatten()
                    #     x, y, x_attr, y_attr = x[:, :, train_mask, :], y[:, :, train_mask, :], x_attr[:, :, train_mask,
                    #                                                                            :], y_attr[:, :, train_mask, :]
                    input = dict(
                        x=x, x_attr=x_attr, y=y, y_attr=y_attr
                    )
                    h_encode = client_model.forward_encoder(input)  # L x (B x N) x F
                    batch_num, node_num = input['x'].shape[0], input['x'].shape[2]
                    graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0,3)  # N x B x L x F
                    graph_encoding = server_model(
                        Data(x=graph_encoding,edge_index=train_edge_index.to('cuda'),
                             edge_attr=train_edge_attr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to('cuda'))
                    )  # N x B x L x F
                    if epoch_i == 1:
                        updated_graph_encoding.append(graph_encoding.detach().clone().cpu())
                    else:
                        y_pred = client_model.forward_decoder(
                            input, h_encode, return_encoding=False,
                            server_graph_encoding=graph_encoding
                        )
                        loss = nn.MSELoss(y_pred,y,0.0)
                        server_optimizer.zero_grad()
                        loss.backward()
                        server_optimizer.step()
                        metrics = unscaled_metrics(y_pred, y, dataset['feature_scaler'], 0.0)
                        metric.append(metrics)
        global_step += 1
        t2=time.time()
        # update global step for all clients
        for client_params in client_params_list:
            client_params.update(start_global_step=global_step)
        # update server_graph_encoding
        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N x B x L x F
        sel_client_i = 0
        for client_i, client_params in enumerate(client_params_list):
            if 'selected' in data['train']:
                if data['train']['selected'][client_i, 0].item() is False:
                    continue
            client_params.update(train_dataset=TensorDataset(
                data['train']['x'][:, :, client_i:client_i + 1, :],
                data['train']['y'][:, :, client_i:client_i + 1, :],
                data['train']['x_attr'][:, :, client_i:client_i + 1, :],
                data['train']['y_attr'][:, :, client_i:client_i + 1, :],
                updated_graph_encoding[sel_client_i:sel_client_i + 1, :, :, :].permute(1, 0, 2, 3)
            ))
            sel_client_i += 1
        mse=[]
        rmse=[]
        mae=[]
        mape=[]
        for i, val in enumerate(metric):

            mse.append(val['mse'])
            rmse.append(val['rmse'])
            mae.append(val['mae'])
            mape.append(val['mape'])

        log = 'global epoch: {:03d}, MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}, Training Time: {:.4f}/global epoch'
        print(id, log.format(m, np.mean(mse),np.mean(rmse), np.mean(mae),np.mean(mape),(t2 - t1), flush=True))
        # torch.save(server_model.state_dict(), "dropnode/" + "server_epoch_" + str(m) + "_" + str(round(np.mean(mae), 4)) + ".pth")
        # torch.save(client_model.state_dict(),
        #            "dropnode/" + "client_epoch_" + str(m) + "_" + str(round(np.mean(mae), 4)) + ".pth")
        local_val_results = []
        server_val_dataloader = DataLoader(server_datasets['val'], batch_size=48, shuffle=True)
        updated_graph_encoding = []
        client_model.eval()
        server_model.eval()
        with torch.no_grad():

            for batch in server_val_dataloader:
                x, y, x_attr, y_attr = batch
                x = x.to(device) if (x is not None) else None
                y = y.to(device) if (y is not None) else None
                x_attr = x_attr.to(device) if (x_attr is not None) else None
                y_attr = y_attr.to(device) if (y_attr is not None) else None
                input = dict(
                    x=x, x_attr=x_attr, y=y, y_attr=y_attr
                )
                h_encode = client_model.forward_encoder(input)  # L x (B x N) x F
                batch_num, node_num = input['x'].shape[0], input['x'].shape[2]
                graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0,3)  # N x B x L x F
                # print(graph_encoding.shape)torch.Size([207, 48, 1, 64])
                graph_encoding = server_model(
                    Data(x=graph_encoding, edge_index=data['test']['edge_index'].to('cuda'),
                         edge_attr=data['test']['edge_attr'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to('cuda'))
                )  # N x B x L x F
                # print(graph_encoding.shape)torch.Size([207, 48, 1, 64])
                updated_graph_encoding.append(graph_encoding.detach().clone().cpu())
        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N x B x L x F
        for client_i, client_params in enumerate(client_params_list):
            keyname = '{}_dataset'.format('val')
            client_params.update({
                keyname: TensorDataset(
                    data['val']['x'][:, :, client_i:client_i + 1, :],
                    data['val']['y'][:, :, client_i:client_i + 1, :],
                    data['val']['x_attr'][:, :, client_i:client_i + 1, :],
                    data['val']['y_attr'][:, :, client_i:client_i + 1, :],
                    updated_graph_encoding[client_i:client_i + 1, :, :, :].permute(1, 0, 2, 3)
                )
            })
        for client_i, client_params in enumerate(client_params_list):
            local_val_result = SplitFedNodePredictorClient.client_local_execute(deepcopy(client_model.state_dict()), 'val',
                                                                                **client_params)
            local_val_results.append(local_val_result)

        client_log = aggregate_local_logs([x[0]['log'] for x in local_val_results])
        EarlyStopping(client_log['val/loss'],client_model,server_model)

        print(client_log)

    # return {'loss': torch.tensor(0).float(), 'progress_bar': log, 'log': log}


def _eval_server_gcn_with_agg_clients(self, name, device):
    assert name in ['val', 'test']
    self.base_model.to(device)
    self.gcn.to(device)
    server_dataloader = DataLoader(self.server_datasets[name], batch_size=self.hparams.server_batch_size, shuffle=False)
    updated_graph_encoding = []
    with torch.no_grad():
        self.base_model.eval()
        self.gcn.eval()
        for batch in server_dataloader:
            x, y, x_attr, y_attr = batch
            x = x.to(device) if (x is not None) else None
            y = y.to(device) if (y is not None) else None
            x_attr = x_attr.to(device) if (x_attr is not None) else None
            y_attr = y_attr.to(device) if (y_attr is not None) else None
            data = dict(
                x=x, x_attr=x_attr, y=y, y_attr=y_attr
            )
            h_encode = self.base_model.forward_encoder(data) # L x (B x N) x F
            batch_num, node_num = data['x'].shape[0], data['x'].shape[2]
            graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # N x B x L x F
            graph_encoding = self.gcn(
                Data(x=graph_encoding,
                edge_index=self.data[name]['edge_index'].to(graph_encoding.device),
                edge_attr=self.data[name]['edge_attr'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(graph_encoding.device))
            ) # N x B x L x F
            updated_graph_encoding.append(graph_encoding.detach().clone().cpu())
    # update server_graph_encoding
    updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1) # N x B x L x F
    for client_i, client_params in enumerate(self.client_params_list):
        keyname = '{}_dataset'.format(name)
        client_params.update({
            keyname: TensorDataset(
                self.data[name]['x'][:, :, client_i:client_i+1, :],
                self.data[name]['y'][:, :, client_i:client_i+1, :],
                self.data[name]['x_attr'][:, :, client_i:client_i+1, :],
                self.data[name]['y_attr'][:, :, client_i:client_i+1, :],
                updated_graph_encoding[client_i:client_i+1, :, :, :].permute(1, 0, 2, 3)
            )
        })

# def train_dataloader():
#     # return a fake dataloader for running the loop
#     return DataLoader([0,])
#
# def val_dataloader():
#     return DataLoader([0,])
#
# def test_dataloader():
#     return DataLoader([0,])

def aggregate_local_train_results(local_train_results):
    return {
        'state_dict': aggregate_local_train_state_dicts(
            [ltr[0]['state_dict'] for ltr in local_train_results]
        ),
        'log': aggregate_local_logs(
            [ltr[0]['log'] for ltr in local_train_results]
        )
    }

def aggregate_local_train_state_dicts(local_train_state_dicts):
    agg_state_dict = {}
    for k in local_train_state_dicts[0]:
        agg_state_dict[k] = 0
        for ltsd in local_train_state_dicts:
            agg_state_dict[k] += ltsd[k]
        agg_state_dict[k] /= len(local_train_state_dicts)
    return agg_state_dict

# def aggregate_local_train_state_dicts(local_train_state_dicts):
#
#     agg_state_dict = deepcopy(local_train_state_dicts[0])
#     for k in agg_state_dict.keys():
#         for i in range(1, len(local_train_state_dicts)):
#             agg_state_dict[k] += local_train_state_dicts[i][k]
#         agg_state_dict[k] = torch.true_divide(agg_state_dict[k], len(local_train_state_dicts))
#     return agg_state_dict

def aggregate_local_logs(local_logs, selected=None):
    agg_log = deepcopy(local_logs[0])
    if selected is not None:
        agg_log_t = deepcopy(local_logs[0])
        agg_log_i = deepcopy(local_logs[0])
    for k in agg_log:
        agg_log[k] = 0
        if selected is not None:
            agg_log_t[k] = 0
            agg_log_i[k] = 0
        for local_log_idx, local_log in enumerate(local_logs):
            if k == 'num_samples':
                agg_log[k] += local_log[k]
            else:
                agg_log[k] += local_log[k] * local_log['num_samples']
            if selected is not None:
                is_trans = selected[local_log_idx, 0].item()
                if is_trans:
                    if k == 'num_samples':
                        agg_log_t[k] += local_log[k]
                    else:
                        agg_log_t[k] += local_log[k] * local_log['num_samples']
                else:
                    if k == 'num_samples':
                        agg_log_i[k] += local_log[k]
                    else:
                        agg_log_i[k] += local_log[k] * local_log['num_samples']
    for k in agg_log:
        if k != 'num_samples':
            agg_log[k] /= agg_log['num_samples']
            if selected is not None:
                agg_log_t[k] /= agg_log_t['num_samples']
                agg_log_i[k] /= agg_log_i['num_samples']
    if selected is not None:
        for k in agg_log_t:
            agg_log[k + '_trans'] = agg_log_t[k]
        for k in agg_log_i:
            agg_log[k + '_induc'] = agg_log_i[k]
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
        local_val_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(self.base_model.state_dict()), 'val', **client_params)
        local_val_results.append(local_val_result)

    self.base_model.to(server_device)
    self.gcn.to(server_device)
    # 2. aggregate
    log = self.aggregate_local_logs([x['log'] for x in local_val_results])
    return {'progress_bar': log, 'log': log}

def validation_epoch_end(self, outputs):
    return self.training_epoch_end(outputs)


def test_epoch_end(self, outputs):
    return self.training_epoch_end(outputs)

setup()