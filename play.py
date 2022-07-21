# import os
# import time
# from argparse import ArgumentParser
# import random
# from copy import deepcopy
# from collections import defaultdict
# import itertools
import random

import numpy as np
import torch
# import torch.nn as nn
# import pickle
# from torch_geometric.data import DataLoader
# from torch_geometric.data import Data,Batch
# from torch.utils.data import TensorDataset
# from tqdm import tqdm
# import util
# from st_datasets import load_dataset
# import base_model
# from standalone import unscaled_metrics
# from base_model.GraphNets import GraphNet,GATGraphNet
# from base_model.GRUSeq2Seq import GRUSeq2Seq,GRUSeq2SeqWithGraphNet
# from torch_geometric.utils import dense_to_sparse
np.random.seed(42)
print(np.random.randint(0,207,size=20))
np.random.seed(42)
print(np.random.randint(0,207,size=20))
print(np.random.randint(0,207,size=20))



# def load_pickle(pickle_file):
#     try:
#         with open(pickle_file, 'rb') as f:
#             pickle_data = pickle.load(f)
#     except UnicodeDecodeError as e:
#         with open(pickle_file, 'rb') as f:
#             pickle_data = pickle.load(f, encoding='latin1')
#     except Exception as e:
#         print('Unable to load data ', pickle_file, ':', e)
#         raise
#     return pickle_data
# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         input_size = 2
#         hidden_size = 64
#         output_size = 1
#         dropout = 0
#         gru_num_layers = 1
#         self.encoder = nn.GRU(
#             input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
#         )
#         self.decoder = nn.GRU(
#             input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout
#         )
#         self.gcn = GraphNet(
#             node_input_size=hidden_size,
#             edge_input_size=1,
#             global_input_size=hidden_size,
#             hidden_size=256,
#             updated_node_size=128,
#             updated_edge_size=128,
#             updated_global_size=128,
#             node_output_size=hidden_size,
#             gn_layer_num=2,
#             activation='ReLU', dropout=dropout
#             ).to('cuda')
#         # self.gcn = GATGraphNet(
#         #     node_input_size=64,
#         #     edge_input_size=1,
#         #     global_input_size=64,
#         #     hidden_size=256,
#         #     updated_node_size=128,
#         #
#         #     updated_global_size=128,
#         #     node_output_size=64,
#         #     gn_layer_num=2,
#         #     activation='ReLU', dropout=0)
#         self.out_net = nn.Linear(hidden_size * 2, output_size)
#         self.batch_num=64
#         self.node_num=207
#
#     def forward_encoder(self, x):
#
#         x_input = x.permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F
#         _, h_encode = self.encoder(x_input)
#         return h_encode # L x (B x N) x F
#
#     def forward_decoder(self, x, y, edge_index, edge_attr, h_encode):
#
#         x_input = x.permute(1, 0, 2, 3).flatten(1, 2)
#
#         graph_encoding = h_encode.view(h_encode.shape[0], self.batch_num, self.node_num, h_encode.shape[2]).permute(2, 1, 0, 3) # N x B x L x F
#         graph_encoding = self.gcn(
#             Data(x=graph_encoding, edge_index=edge_index, edge_attr=edge_attr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).to('cuda')) # N x B x L x F
#         # graph_encoding = self.gcn(
#         #     Data(x=graph_encoding, edge_index=edge_index).to('cuda'))  # N x B x L x F
#         graph_encoding = graph_encoding.permute(2, 1, 0, 3).flatten(1, 2) # L x (B x N) x F
#         h_encode = torch.cat([h_encode, graph_encoding], dim=-1)
#         y_input = y.permute(1, 0, 2, 3).flatten(1, 2)
#         y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)
#         out_hidden, _ = self.decoder(y_input, h_encode)
#         out = self.out_net(out_hidden)
#         out = out.view(out.shape[0], self.batch_num, self.node_num, out.shape[-1]).permute(1, 0, 2, 3)
#
#
#         return out
#
#     def forward(self, x, y, edge_index, edge_attr):
#         h_encode = self.forward_encoder(x)
#         return self.forward_decoder( x, y, edge_index, edge_attr, h_encode)
# def setup():
#
#
#
#     sensor_ids, sensor_id_to_idx, adj_mat = util.load_adj('data/sensor_graph/adj_mx.pkl', 'doubletransition')
#     dataloader = util.load_dataset('data/METR-LA', 64, 64, 64)
#     scaler = dataloader['scaler']
#     model=Net()
#     model.to('cuda')
#     adj_mat = torch.from_numpy(adj_mat).float()
#     train_edge_index, train_edge_attr = dense_to_sparse(adj_mat)
#     train_edge_index.to('cuda')
#     train_edge_attr.to('cuda')
#     model_state_dict = torch.load(r'centra/CNFGNN/METR-LA/no_learning/drop0/_epoch_17_4.1844.pth')
#     model.load_state_dict(model_state_dict)
#     outputs = []
#     realy=[]
#     with torch.no_grad():
#         for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
#             model.eval()
#             testx = torch.Tensor(x).to('cuda')
#             testy = torch.Tensor(y).to('cuda')
#             output = model(testx, testy, train_edge_index, train_edge_attr)
#             realy.append(testy)
#             outputs.append(output)
#     realy=torch.cat(realy,dim=0)
#     yhat = scaler.inverse_transform(torch.cat(outputs, dim=0))
#     realy = scaler.inverse_transform(realy)
#     amae = []
#     amape = []
#     armse = []
#     amse=[]
#     for i in range(12):
#         pred = yhat[:, i, :, :]
#         real = realy[:, i, :, 0].unsqueeze(-1)
#         metrics = util.metric(pred, real)
#         log = "horizon {:d} | Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}"
#         print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
#
#         amae.append(metrics[0])
#         amape.append(metrics[1])
#         armse.append(metrics[2])
#         amse.append(metrics[3])
#     log = "average over 12 horizons | Test MAE: {:.4f} | Test MAPE: {:.4f} | Test RMSE: {:.4f}| Test MSE: {:.4f}"
#     print(log.format(np.mean(amae), np.mean(amape), np.mean(armse),np.mean(amse)))
#
# setup()