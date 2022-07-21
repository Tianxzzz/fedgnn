import os
import time
from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
import util
import torch.multiprocessing as mp
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
from tqdm import tqdm
from st_datasets import load_dataset
import base_model
from standalone import unscaled_metrics
from base_model.GraphNets import GraphNet
from base_model.GRUSeq2Seq import GRUSeq2Seq,GRUSeq2SeqWithGraphNet

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

        if state_dict is not None:
            self.base_model.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def local_eval(self, dataloader, name, state_dict_to_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
        self.eval()
        output=[]
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
                y_pred= self(data, server_graph_encoding)
                output.append(y_pred)

                loss = nn.MSELoss()(y_pred,y)
                num_samples += x.shape[0]
                metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 0.0)
                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0]
                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]
            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()
        output=torch.cat(output,dim=0)
            # client_encodings=torch.cat(client_encodings,1)

        # self.cpu()
        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)

        return {'log': epoch_log,'result':output}

    def local_validation(self, state_dict_to_load):
        return self.local_eval(self.val_dataloader, 'val', state_dict_to_load)

    def local_test(self, state_dict_to_load):
        return self.local_eval(self.test_dataloader, 'test', state_dict_to_load)

    @staticmethod
    def client_local_execute(state_dict_to_load, order, **hparams_list):

        res_list = []
        # for hparams in hparams_list:
        client = SplitFedNodePredictorClient(**hparams_list)

        if order == 'val':
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

    device='cuda'
    data = load_dataset(dataset_name='METR-LA')
    dataset = data

    num_clients = data['train']['x'].shape[2]
    input_size = dataset['train']['x'].shape[-1] + dataset['train']['x_attr'].shape[-1]
    output_size = dataset['train']['y'].shape[-1]
    # print(data['train']['x'].shape)  torch.Size([23974, 12, 207, 1])
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
    server_datasets = {}
    for name in ['train', 'val', 'test']:
        server_datasets[name] = TensorDataset(
            data[name]['x'], data[name]['y'],
            data[name]['x_attr'], data[name]['y_attr']
        )
    local_val_results = []
    client_state_dict=torch.load(r'CNFGNN/METR-LA/client_epoch_25_tensor(8.7684, dtype=torch.float64).pth')
    server_state_dict= torch.load(r'CNFGNN/METR-LA/server_epoch_25_tensor(8.7684, dtype=torch.float64).pth')
    client_model.to(device)
    server_model.to(device)
    client_model.load_state_dict(client_state_dict)
    server_model.load_state_dict(server_state_dict)
    server_test_dataloader = DataLoader(server_datasets['test'], batch_size=48, shuffle=False)
    updated_graph_encoding = []


    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        # for client_i, client_params in enumerate(client_params_list):
        #     local_val_result = SplitFedNodePredictorClient.client_local_execute(deepcopy(client_state_dict), 'test',
        #                                                                         **client_params)
        #     local_val_results.append(local_val_result)
        #
        # client_log = aggregate_local_logs([x[0]['log'] for x in local_val_results])
        # print(client_log)
        for batch in server_test_dataloader:
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

            graph_encoding = server_model(
                Data(x=graph_encoding,edge_index=data['test']['edge_index'].to('cuda'),
                     edge_attr=data['test']['edge_attr'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to('cuda'))
            )  # N x B x L x F

            updated_graph_encoding.append(graph_encoding.detach().clone().cpu())
    updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)  # N x B x L x F
    for client_i, client_params in enumerate(client_params_list):
        keyname = '{}_dataset'.format('test')
        client_params.update({
            keyname: TensorDataset(
                data['test']['x'][:, :, client_i:client_i + 1, :],
                data['test']['y'][:, :, client_i:client_i + 1, :],
                data['test']['x_attr'][:, :, client_i:client_i + 1, :],
                data['test']['y_attr'][:, :, client_i:client_i + 1, :],
                updated_graph_encoding[client_i:client_i + 1, :, :, :].permute(1, 0, 2, 3)
            )
        })
    # local_val_results=[]
    for client_i, client_params in enumerate(client_params_list):

        local_val_result = SplitFedNodePredictorClient.client_local_execute(deepcopy(client_state_dict),'test', **client_params)
        local_val_results.append(local_val_result)

    client_log = aggregate_local_logs([x[0]['log'] for x in local_val_results])
    print(client_log)
    y_pred=[]
    for x in local_val_results:
        y_pred.append(x[0]['result'])
    y_pred=torch.cat(y_pred,dim=2)

    amae=[]
    armse=[]
    amape=[]
    y_pred = dataset['feature_scaler'].inverse_transform(y_pred.detach().cpu())
    real = dataset['feature_scaler'].inverse_transform(data['test']['y'].detach().cpu())
    for i in range(12):

        metrics=util.metric(y_pred[:,11-i,:,:],real[:,11-i,:,:])
        log='horizon{:d}, Test MAE:{:.4f},Test RMSE:{:.4f},Test MAPE:{:.4f}'
        print(log.format(i+1,metrics[0],metrics[2],metrics[1]))
        amae.append(metrics[0])
        armse.append(metrics[2])
        amape.append(metrics[1])
    log='On average 12 horizon,Test MAE:{:.4f},Test RMSE:{:.4f},Test MAPE:{:.4f}'
    print(log.format(np.mean(amae),np.mean(armse),np.mean(amape)))




def aggregate_local_train_results(local_train_results):
    return {

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

def test_step(self, batch, batch_idx):
    server_device = next(self.gcn.parameters()).device
    self._eval_server_gcn_with_agg_clients('test', server_device)
    # 1. vaidate locally and collect uploaded local train results
    local_val_results = []
    self.base_model.to('cpu')
    self.gcn.to('cpu')

    for client_i, client_params in enumerate(self.client_params_list):
        local_val_result = SplitFedNodePredictorClient.client_local_execute(batch.device, deepcopy(self.base_model.state_dict()), 'test', **client_params)
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

setup()