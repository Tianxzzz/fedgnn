from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model.models import GATMetaLayer
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Batch
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv,GATv2Conv,GATConv
from torch_geometric.utils import to_dense_adj,dense_to_sparse
import random
class MLP_GN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        hidden_layer_num, activation='ReLU', dropout=0.0):
        super().__init__()
        self.net = []
        last_layer_size = input_size
        for _ in range(hidden_layer_num):
            self.net.append(nn.Linear(last_layer_size, hidden_size))
            self.net.append(getattr(nn, activation)())
            self.net.append(nn.Dropout(p=dropout))
            last_layer_size = hidden_size
        self.net.append(nn.Linear(last_layer_size, output_size))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class EdgeModel(nn.Module):
    def __init__(self, 
        node_input_size, edge_input_size, global_input_size, 
        hidden_size, edge_output_size, activation, dropout):
        super(EdgeModel, self).__init__()
        edge_mlp_input_size = 2 * node_input_size + edge_input_size + global_input_size
        self.edge_mlp = MLP_GN(edge_mlp_input_size, hidden_size, edge_output_size, 2, activation, dropout)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.

        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], -1)
        if u is not None:
            out = torch.cat([out, u[batch]], -1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self,
        node_input_size, edge_input_size, global_input_size,
        hidden_size, node_output_size, activation, dropout):
        super(NodeModel, self).__init__()
        node_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.node_mlp = MLP_GN(node_mlp_input_size, hidden_size, node_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        received_msg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, received_msg], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
        return self.node_mlp(out)


class GlobalModel(nn.Module):
    def __init__(self,
        node_input_size, edge_input_size, global_input_size,
        hidden_size, global_output_size, activation, dropout):
        super(GlobalModel, self).__init__()
        global_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.global_mlp = MLP_GN(global_mlp_input_size, hidden_size, global_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        agg_node = scatter_add(x, batch, dim=0)
        agg_edge = scatter_add(scatter_add(edge_attr, col, dim=0, dim_size=x.size(0)), batch, dim=0)
        out = torch.cat([agg_node, agg_edge, u], dim=-1)
        return self.global_mlp(out)


class GraphNet(nn.Module):
    def __init__(self, node_input_size, edge_input_size, global_input_size, 
        hidden_size,
        updated_node_size, updated_edge_size, updated_global_size,
        node_output_size,
        gn_layer_num, activation, dropout, *args, **kwargs):
        super().__init__()

        self.global_input_size = global_input_size
        self.nodevec1 = nn.Parameter(torch.randn(207, 10).to('cuda'), requires_grad=True).to('cuda')
        self.nodevec2 = nn.Parameter(torch.randn(10, 207).to('cuda'), requires_grad=True).to('cuda')
        self.net = []

        last_node_input_size = node_input_size
        last_edge_input_size = edge_input_size
        last_global_input_size = global_input_size
        for _ in range(gn_layer_num):
            edge_model = EdgeModel(last_node_input_size, last_edge_input_size, last_global_input_size, hidden_size, updated_edge_size,
                activation, dropout)
            last_edge_input_size += updated_edge_size
            node_model = NodeModel(last_node_input_size, updated_edge_size, last_global_input_size, hidden_size, updated_node_size,
                activation, dropout)
            last_node_input_size += updated_node_size
            global_model = GlobalModel(updated_node_size, updated_edge_size, last_global_input_size, hidden_size, updated_global_size,
                activation, dropout)
            last_global_input_size += updated_global_size
            self.net.append(MetaLayer(
                edge_model, node_model, global_model
            ))
        self.net = nn.ModuleList(self.net)
        self.node_out_net = nn.Linear(last_node_input_size, node_output_size)
    
    def forward(self, data):
        # if not hasattr(data, 'batch'):
        data = Batch.from_data_list([data])
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # batch = batch.to('cuda')

        edge_attr = edge_attr.expand(-1, x.shape[1], x.shape[2], -1)

        u = x.new_zeros(*([batch[-1] + 1] + list(x.shape[1:-1]) + [self.global_input_size]))
        for net in self.net:
            updated_x, updated_edge_attr, updated_u = net(x, edge_index, edge_attr, u, batch)
            x = torch.cat([updated_x, x], dim=-1)
            edge_attr = torch.cat([updated_edge_attr, edge_attr], dim=-1)
            u = torch.cat([updated_u, u], dim=-1)
        node_out = self.node_out_net(x)
        return node_out

class GATNodeModel(nn.Module):
    def __init__(self,
        node_input_size, global_input_size,
        hidden_size, node_output_size, activation, dropout):
        super(GATNodeModel, self).__init__()
        node_mlp_input_size = node_input_size  + global_input_size
        self.node_mlp = MLP_GN(node_mlp_input_size, hidden_size, node_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, u, batch):


        out = torch.cat([x, u[batch]], dim=-1)
        return self.node_mlp(out)


class GATGlobalModel(nn.Module):
    def __init__(self,
        node_input_size, global_input_size,
        hidden_size, global_output_size, activation, dropout):
        super(GATGlobalModel, self).__init__()
        global_mlp_input_size = node_input_size  + global_input_size
        self.global_mlp = MLP_GN(global_mlp_input_size, hidden_size, global_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, u, batch):

        agg_node = scatter_add(x, batch, dim=0)
        out = torch.cat([agg_node, u], dim=-1)
        return self.global_mlp(out)

class GATGraphNet(nn.Module):
    def __init__(self, node_input_size, edge_input_size, global_input_size,
                 hidden_size,
                 updated_node_size, updated_global_size,
                 node_output_size,
                 gn_layer_num, activation, dropout, *args, **kwargs):
        super().__init__()

        self.global_input_size = global_input_size

        self.net = []
        last_node_input_size = node_input_size
        last_edge_input_size = edge_input_size
        last_global_input_size = global_input_size
        for _ in range(gn_layer_num):
            # edge_model = EdgeModel(last_node_input_size, last_edge_input_size, last_global_input_size, hidden_size,
            #                        updated_edge_size,
            #                        activation, dropout)
            # last_edge_input_size += updated_edge_size
            node_model = GATNodeModel(last_node_input_size, last_global_input_size, hidden_size,
                                   updated_node_size,
                                   activation, dropout)
            last_node_input_size += updated_node_size
            global_model = GATGlobalModel(updated_node_size, last_global_input_size, hidden_size,
                                       updated_global_size,
                                       activation, dropout)
            last_global_input_size += updated_global_size
            self.net.append(GATMetaLayer(node_model, global_model))
        self.net = nn.ModuleList(self.net)
        self.node_out_net = nn.Linear(last_node_input_size, node_output_size)
        # self.linear = nn.Linear(64,1)
        self.gat = GATv2Conv(48*64, 48*64)
        self.gat1 = GATv2Conv(48 * 64, 48 * 64)

    def forward(self, data):
        # if not hasattr(data, 'batch'):
        data = Batch.from_data_list([data])
        x, edge_index,batch = data.x,data.edge_index, data.batch
        gatinput=x

        # gatinput=self.linear(gatinput)
        # gatinput = gatinput.reshape(x.shape[0], 48)
        gatinput = gatinput.reshape(x.shape[0], 48*64)
        gatoutput,edge=self.gat(gatinput,edge_index,return_attention_weights=True)
        edge_index = edge[0]
        gatoutput=F.relu(gatoutput)
        gatoutput, edge = self.gat(gatoutput, edge_index, return_attention_weights=True)
        gatoutput = F.relu(gatoutput)
        edge_index = edge[0]
        edge_attr = edge[1]
        x=gatoutput.reshape(x.shape[0],48,1,64)
        # edge_attr = edge_attr.unsqueeze(-1).unsqueeze(-1)
        # # batch = batch.to('cuda')
        # edge_attr = edge_attr.expand(-1, x.shape[1], x.shape[2], -1)
        u = x.new_zeros(*([batch[-1] + 1] + list(x.shape[1:-1]) + [self.global_input_size]))
        for net in self.net:
            updated_x, updated_u = net(x, edge_index, u, batch)
            x = torch.cat([updated_x, x], dim=-1)
            # edge_attr = torch.cat([updated_edge_attr, edge_attr], dim=-1)
            u = torch.cat([updated_u, u], dim=-1)
        node_out = self.node_out_net(x)
        return node_out
class GATGN(nn.Module):
    def __init__(self, node_input_size, edge_input_size, global_input_size,
        hidden_size,
        updated_node_size, updated_edge_size, updated_global_size,
        node_output_size,
        gn_layer_num, activation, dropout, *args, **kwargs):
        super().__init__()

        self.global_input_size = global_input_size
        self.nodevec1 = nn.Parameter(torch.randn(207, 10).to('cuda'), requires_grad=True).to('cuda')
        self.nodevec2 = nn.Parameter(torch.randn(10, 207).to('cuda'), requires_grad=True).to('cuda')
        self.net = []

        last_node_input_size = node_input_size
        last_edge_input_size = edge_input_size
        last_global_input_size = global_input_size
        for _ in range(2):
            edge_model = EdgeModel(last_node_input_size, last_edge_input_size, last_global_input_size, hidden_size, updated_edge_size,
                activation, dropout)
            last_edge_input_size += updated_edge_size
            node_model = NodeModel(last_node_input_size, updated_edge_size, last_global_input_size, hidden_size, updated_node_size,
                activation, dropout)
            last_node_input_size += updated_node_size
            global_model = GlobalModel(updated_node_size, updated_edge_size, last_global_input_size, hidden_size, updated_global_size,
                activation, dropout)
            last_global_input_size += updated_global_size
            self.net.append(MetaLayer(
                edge_model, node_model, global_model
            ))
        self.net = nn.ModuleList(self.net)
        self.node_out_net = nn.Linear(last_node_input_size, node_output_size)
        self.gat = GATv2Conv(48 * 64, 48 * 64)
        self.gat1 = GATv2Conv(48 * 32, 48 * 32)
        self.BN=nn.BatchNorm1d(48 * 64)
        self.BN2 = nn.BatchNorm1d(48 * 32)

    def forward(self, data,epoch):
        # if not hasattr(data, 'batch'):
        data = Batch.from_data_list([data])
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # batch = batch.to('cuda')
        gatinput = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[-1])
        # print(x.shape)torch.Size([207, 48, 12, 64])
        adp = np.ones((x.shape[0], x.shape[0]))
        adp = torch.from_numpy(adp).float()
        edge_index, edge_attr = dense_to_sparse(adp)
        edge_index = edge_index.to('cuda')
        # if self.training:
        #     index = np.random.randint(0, 207, size=20)
        #     for i in index:
        #         adp[[i], :] = 0
        #         adp[:, [i]] = 0
        #     edge_index, edge_attr = dense_to_sparse(adp)
        #     edge_attr = edge_attr.to('cuda')
        #     edge_index = edge_index.to('cuda')
        #     edge_attr = edge_attr.unsqueeze(-1)
        gatoutput, edge = self.gat(x=gatinput, edge_index=edge_index, return_attention_weights=True)
        edge_index = edge[0]
        edge_attr = edge[1]
        gatoutput = F.relu(gatoutput)
        # gatoutput=self.BN(gatoutput)

        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr).cpu()
        adj = adj.squeeze(0).squeeze(-1)
        b=torch.zeros(x.shape[0],x.shape[0])
        adj=torch.where(adj>0.1,adj,b)
        edge_index, edge_attr = dense_to_sparse(adj)
        edge_attr = edge_attr.to('cuda')
        edge_index=edge_index.to('cuda')
        edge_attr = edge_attr.unsqueeze(-1)
        # if self.training:
        #     adj=to_dense_adj(edge_index=edge_index,edge_attr=edge_attr).cpu()
        #
        #     adj=adj.squeeze(0).squeeze(-1)
        #     # np.random.seed(epoch)
        #     # adp = np.ones((num_clients, num_clients))
        #     # adp= torch.from_numpy(adp).float()
        #     index = np.random.randint(0, 207, size=20)
        #     for i in index:
        #         adj[[i], :] = 0
        #         adj[:, [i]] = 0
        #     edge_index, edge_attr = dense_to_sparse(adj)
        #     edge_attr=edge_attr.to('cuda')
        #     edge_index=edge_index.to('cuda')
        #     edge_attr = edge_attr.unsqueeze(-1)

        # gatoutput=torch.cat((x,gatoutput.reshape(x.shape[0],x.shape[1],1,64)),dim=-1)
        gatoutput=gatoutput.reshape(x.shape[0],x.shape[1],1,64)
        x = gatoutput
        edge_attr = edge_attr.unsqueeze(-1).unsqueeze(-1)
        # #
        # # # # batch = batch.to('cuda')
        edge_attr = edge_attr.expand(-1, x.shape[1], x.shape[2], -1)
        # # x = torch.cat([gatoutput, x], dim=-1)
        #
        u = x.new_zeros(*([batch[-1] + 1] + list(x.shape[1:-1]) + [self.global_input_size]))
        #
        for net in self.net:
            updated_x, updated_edge_attr, updated_u = net(x, edge_index, edge_attr, u, batch)
            x = torch.cat([updated_x, x], dim=-1)
            edge_attr = torch.cat([updated_edge_attr, edge_attr], dim=-1)
            u = torch.cat([updated_u, u], dim=-1)
        node_out = self.node_out_net(x)
        return node_out