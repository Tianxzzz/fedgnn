import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import scipy.sparse as sp
import networkx as nx
shufflerng = np.random.RandomState(42)
shuffled_train_idx = shufflerng.permutation(200).astype(int)
print(shufflerng)
print(shuffled_train_idx)

# num_nodes = 5
# k = num_nodes // 2 + 1
# G = nx.newman_watts_strogatz_graph(num_nodes, k, 0.1)
# num_feats = 2
# feat = torch.randn(num_nodes, num_feats)
# adj_matrix = nx.to_scipy_sparse_matrix(G)
# adj_matrix += sp.eye(adj_matrix.shape[0], format='csr')
# neighbors = [torch.as_tensor(row) for row in adj_matrix.tolil().rows]
# aggregation = []
# print(feat)
# for nbr in neighbors:
#     print(nbr)
#     print(feat[nbr])
#     message = torch.median(feat[nbr], dim=0).values
#     print(message)
#     aggregation.append(message)
# h = torch.stack(aggregation)
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#
#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)
#
#         # Step 3: Compute normalization.
#         row, col = edge_index
#         print(edge_index)
#         print(row)
#         print(col)
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         print(deg)
#         deg_inv_sqrt = deg.pow(-0.5)
#         print(deg_inv_sqrt)
#         print(deg_inv_sqrt[row])
#         print(deg_inv_sqrt[col])
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#         print(norm)
#
#         # Step 4-5: Start propagating messages.
#         return self.propagate(edge_index, x=x, norm=norm)
#
#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]
#         print(x_j)
#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j
#
# edge_index = torch.as_tensor([[0, 1, 2], [2, 0, 1]])
# x = torch.randn(3, 5)
# print(x)
# conv = GCNConv(5, 2)
# y= conv(x, edge_index)