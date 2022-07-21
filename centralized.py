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
from torch_geometric.data import DataLoader
from torch_geometric.data import Data,Batch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import util
from st_datasets import load_dataset
import base_model
from standalone import unscaled_metrics
from base_model.GraphNets import GraphNet,GATGraphNet
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
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 2
        hidden_size = 64
        output_size = 1
        dropout = 0
        gru_num_layers = 1
        self.encoder = nn.GRU(
            input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.decoder = nn.GRU(
            input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.gcn = GraphNet(
            node_input_size=hidden_size,
            edge_input_size=1,
            global_input_size=hidden_size,
            hidden_size=256,
            updated_node_size=128,
            updated_edge_size=128,
            updated_global_size=128,
            node_output_size=hidden_size,
            gn_layer_num=2,
            activation='ReLU', dropout=dropout
            ).to('cuda')
        # self.gcn = GATGraphNet(
        #     node_input_size=64,
        #     edge_input_size=1,
        #     global_input_size=64,
        #     hidden_size=256,
        #     updated_node_size=128,
        #
        #     updated_global_size=128,
        #     node_output_size=64,
        #     gn_layer_num=2,
        #     activation='ReLU', dropout=0
        # ).to('cuda')
        self.out_net = nn.Linear(hidden_size * 2, output_size)
        self.batch_num=64
        self.node_num=207

    def forward_encoder(self, x):
        # print(x.shape)torch.Size([64, 12, 207, 2])


        x_input = x.permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F
        # print(x_input.shape)torch.Size([12, 13248, 2])
        _, h_encode = self.encoder(x_input)
        # print(h_encode.shape)torch.Size([1, 13248, 64])
        return h_encode # L x (B x N) x F

    def forward_decoder(self, x, y, edge_index, edge_attr, h_encode):

        x_input = x.permute(1, 0, 2, 3).flatten(1, 2)
        # print(x_input.shape)torch.Size([12, 13248, 2])
        graph_encoding = h_encode.view(h_encode.shape[0], self.batch_num, self.node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # N x B x L x F
        # CNFGNN
        # print(graph_encoding.shape)torch.Size([207, 64, 1, 64])
        graph_encoding = self.gcn(
            Data(x=graph_encoding, edge_index=edge_index, edge_attr=edge_attr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).to('cuda')
        ) # N x B x L x F
        # GAT+GN(-edge)
        # graph_encoding = self.gcn(
        #     Data(x=graph_encoding, edge_index=edge_index).to('cuda'))  # N x B x L x F
        graph_encoding = graph_encoding.permute(2, 1, 0, 3).flatten(1, 2) # L x (B x N) x F
        h_encode = torch.cat([h_encode, graph_encoding], dim=-1)
        # print(h_encode.shape)torch.Size([1, 13248, 128])
        y_input = y.permute(1, 0, 2, 3).flatten(1, 2)
        y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)
        # print(y_input.shape)torch.Size([12, 13248, 2])
        out_hidden, _ = self.decoder(y_input, h_encode)
        # print(out_hidden.shape)torch.Size([12, 13248, 128])
        out = self.out_net(out_hidden)
        # print(out.shape)torch.Size([12, 13248, 1])
        out = out.view(out.shape[0], self.batch_num, self.node_num, out.shape[-1]).permute(1, 0, 2, 3)
        # print(out.shape)torch.Size([64, 12, 207, 1])

        return out

    def forward(self, x, y, edge_index, edge_attr):
        h_encode = self.forward_encoder(x)
        return self.forward_decoder( x, y, edge_index, edge_attr, h_encode)


def setup():
    # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
    device = torch.device('cuda')
    sensor_ids, sensor_id_to_idx, adj_mat = util.load_adj('data/sensor_graph/adj_mx.pkl', 'doubletransition')
    dataloader = util.load_dataset('data/METR-LA', 64, 64, 64)
    scaler = dataloader['scaler']

    model=Net()
    model.to('cuda')
    adj_mat = torch.from_numpy(adj_mat).float()
    train_edge_index, train_edge_attr = dense_to_sparse(adj_mat)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    loss_list = []
    train_time, val_time = [], []
    for epoch in range(100):
        # index = random.sample(range(0, 207), 20)
        # for i in index:
        #     adj_mat[[i], :] = 0
        #     adj_mat[:, [i]] = 0
        # train_edge_index, train_edge_attr = dense_to_sparse(adj_mat)
        train_edge_index.to('cuda')
        train_edge_attr.to('cuda')
        train_loss = []
        train_mae=[]
        train_mape = []
        train_rmse = []
        t1 = time.time()
        # dataloader['train_loader'].shuffle()
        with torch.enable_grad():
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                model.train()
                trainx = torch.Tensor(x).to('cuda')
                trainy = torch.Tensor(y).to('cuda')

                output = model(trainx,trainy,train_edge_index,train_edge_attr)
                # print(output.shape)
                # print(trainy[:,:,:,0].shape)
                predict = scaler.inverse_transform(output)
                trainy = scaler.inverse_transform(trainy)
                optimizer.zero_grad()
                loss = util.masked_mse(predict, trainy[:,:,:,0].unsqueeze(-1),0.0)
                loss.backward()
                optimizer.step()
                mae=util.masked_mae(predict, trainy[:,:,:,0].unsqueeze(-1), 0.0).item()
                mape = util.masked_mape(predict, trainy[:,:,:,0].unsqueeze(-1), 0.0).item()
                rmse = util.masked_rmse(predict, trainy[:,:,:,0].unsqueeze(-1), 0.0).item()
                train_loss.append(loss.item())
                train_mae.append(mae)
                train_rmse.append(rmse)
                train_mape.append(mape)
                if iter % 50 == 0:
                    log = "Iter: {:03d} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Train MAE: {:.4f} | Train MAPE: {:.4f}"
                    print(log.format(iter, train_loss[-1], train_rmse[-1],train_mae[-1],train_mape[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)

        val_loss = []
        val_mae=[]
        val_mape = []
        val_rmse = []
        s1 = time.time()
        with torch.no_grad():
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                model.eval()
                valx = torch.Tensor(x).to('cuda')
                valy = torch.Tensor(y).to('cuda')

                output = model(valx,valy,train_edge_index,train_edge_attr)

                predict = scaler.inverse_transform(output)
                valy = scaler.inverse_transform(valy)
                loss = util.masked_mse(predict, valy[:,:,:,0].unsqueeze(-1),0.0)

                mae=util.masked_mae(predict, valy[:,:,:,0].unsqueeze(-1), 0.0).item()
                mape = util.masked_mape(predict, valy[:,:,:,0].unsqueeze(-1), 0.0).item()
                rmse = util.masked_rmse(predict, valy[:,:,:,0].unsqueeze(-1), 0.0).item()
                val_loss.append(loss.item())
                val_mae.append(mae)
                val_rmse.append(rmse)
                val_mape.append(mape)
        s2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(epoch, (t2 - t1)))
        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(epoch, (s2 - s1)))
        val_time.append(s2 - s1)
        mean_train_loss = np.mean(train_loss)
        mean_train_mae = np.mean(train_mae)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        mean_val_loss = np.mean(val_loss)
        mean_val_mae = np.mean(val_mae)
        mean_val_mape = np.mean(val_mape)
        mean_val_rmse = np.mean(val_rmse)
        loss_list.append(mean_val_loss)
        log = "Epoch: {:03d} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Train MAE: {:.4f} | Train MAPE: {:.4f} | Val Loss: {:.4f} | Val RMSE: {:.4f} | Val MAE: {:.4f}|Val MAPE: {:.4f}"
        print(log.format(epoch, mean_train_loss, mean_train_rmse,mean_train_mae,mean_train_mape, mean_val_loss, mean_val_rmse,mean_val_mae,mean_val_mape), flush=True)
        # torch.save(model.state_dict(),
        #             "centra/CNFGNN/METR-LA/no_learning/drop0/" + "_epoch_" + str(epoch) + "_" + str(round(mean_val_rmse,4)) + ".pth")
#     test
#     model_state_dict = torch.load(
#         r'centra/2layerGN/MSE/_epoch_6_3.6905.pth')
#     model.load_state_dict(model_state_dict)
#     outputs = []
#     realy = torch.Tensor(dataloader['y_test']).to('cuda')
#     realy = realy.transpose[:, :, :, 0].unsquueze(-1)
#     with torch.no_grad():
#         for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
#             model.eval()
#             testx = torch.Tensor(x).to('cuda')
#             testy = torch.Tensor(y).to('cuda')
#             output = model(testx, testy, train_edge_index, train_edge_attr)
#
#             outputs.append(output)
#
#     yhat = torch.cat(outputs, dim=0)
#     amae = []
#     amape = []
#     armse = []
#     for i in range(12):
#         pred = scaler.inverse_transform(yhat[:, i, :,:])
#         real = realy[:, i, :,:]
#         metrics = util.metric(pred, real)
#         log = "Evaluate best model on test data for horizon {:d} | Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}"
#         print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
#         amae.append(metrics[0])
#         amape.append(metrics[1])
#         armse.append(metrics[2])
#     log = "On average over 12 horizons | Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}"
#     print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))



setup()