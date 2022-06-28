##################################
#       总的模型
#     模型的输入维数为[batch,node_feature,num_nodes,time_step+1] [64, 2, 207,13] 做填充
#     模型输出维数为[batch, time_step, num_nodes, node_feature]  [64, 12, 207, 1]
#
################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import global_add_pool,global_mean_pool,SAGEConv,GraphConv,GATConv,GINConv,GCNConv

import math
# class Net(torch.nn.Module):
#     ## 将dilation_channels以及gcn_channels改大会提升性能
#     def __init__(self,device,num_fea,batch,dilation_channels=32,gcn_channels=32,vae_hidden_dim=128):  ##
#         super(Net, self).__init__()
#         self.device=device
#         self.channels=gcn_channels
#         self.num_fea=num_fea
#         self.batch=batch
#
#         self.dilation=dilation_conv(num_fea,dilation_channels,batch)##[64X207,6,4]
#         self.graph_conv_0 = MyGAT(dilation_channels,gcn_channels,self.device,self.batch)
#         self.graph_conv_1 = MyGAT(dilation_channels, gcn_channels,self.device,self.batch)
#         self.graph_conv_2 = MyGAT(dilation_channels,gcn_channels,self.device,self.batch)
#         self.graph_conv_3 = MyGAT(dilation_channels, gcn_channels,self.device,self.batch) ##[64X207,12]
#         self.graph_conv_4 = MyGAT(dilation_channels,gcn_channels,self.device,self.batch)
#         self.graph_conv_5 = MyGAT(dilation_channels, gcn_channels,self.device,self.batch)
#         self.graph_conv_6 = MyGAT(dilation_channels,gcn_channels,self.device,self.batch)
#         self.graph_conv_7 = MyGAT(dilation_channels,gcn_channels,self.device,self.batch)
#         self.vae=VAE(device,self.batch,time_step=12,in_dim=dilation_channels,hidden_dim=vae_hidden_dim,out_dim=1)
#         self.fc=nn.Linear(8*dilation_channels,dilation_channels)
#         self.gcn=SAGEConv(gcn_channels,gcn_channels)
#     def forward(self,x,edge_index):##[64,2,207,13]
#         # print(x.size())#[64,2,num_node,13]
#         x=x.permute(0,2,1,3)
#         size=x.size()
#         num_node=size[1]
#
#         x=torch.reshape(x,(-1,size[2],size[3]))##[64X207,2,13]
#
#         # 输入：[batch * num_nodes, node_feature, time_step + 1]
#         # 输出：[batch,num_nodes,dilation_channels,1]
#         x0,x1,x2,x3,x4,x5,x6,x7=self.dilation(x,num_node)
#         # print(x0.size())
#
#         x=torch.cat((x0,x1,x2,x3,x4,x5,x6,x7),dim=-1)
#         x=torch.reshape(x,(self.batch,num_node,-1))
#         x=self.fc(x)
#         re=[]
#         data_list0 = []
#         data_list1 = []
#         data_list2 = []
#         data_list3 = []
#         data_list4 = []
#         data_list5 = []
#         data_list6 = []
#         data_list7 = []
#
#         for i in range(self.batch):
#             data_list0.append(Data(x[i, :, :].squeeze(-1), edge_index.to(self.device)))
#             # data_list1.append(Data(x1[i, :, :].squeeze(-1)))
#             # data_list2.append(Data(x2[i, :, :].squeeze(-1)))
#             # data_list3.append(Data(x3[i, :, :].squeeze(-1)))
#             # data_list4.append(Data(x4[i, :, :].squeeze(-1)))
#             # data_list5.append(Data(x5[i, :, :].squeeze(-1)))
#             # data_list6.append(Data(x6[i, :, :].squeeze(-1)))
#             # data_list7.append(Data(x7[i, :, :].squeeze(-1)))
#         loader0 = DataLoader(data_list0, batch_size=self.batch, shuffle=False)
#         # loader1 = DataLoader(data_list1, batch_size=self.batch, shuffle=False)
#         # loader2 = DataLoader(data_list2, batch_size=self.batch, shuffle=False)
#         # loader3 = DataLoader(data_list3, batch_size=self.batch, shuffle=False)
#         # loader4 = DataLoader(data_list4, batch_size=self.batch, shuffle=False)
#         # loader5 = DataLoader(data_list5, batch_size=self.batch, shuffle=False)
#         # loader6 = DataLoader(data_list6, batch_size=self.batch, shuffle=False)
#         # loader7 = DataLoader(data_list7, batch_size=self.batch, shuffle=False)
#         for data in loader0:
#             x_0, _, edge_index = data.x, data.batch, data.edge_index
#             # print(x_0.size())
#             x_0=self.gcn(x_0,edge_index)
#             # x_0, re0,p0 = self.graph_conv_0(x_0, edge_index,num_node)
#             # print(x_0.size())
#             # re.append(re0)
#
#         # for data in loader1:
#         #     x_1 = data.x
#         #     x_1,re1 ,p1= self.graph_conv_1(x_1, edge_index,num_node)
#         #     re.append(re1)
#         #
#         # for data in loader2:
#         #     x_2 = data.x
#         #     x_2, re2 ,p2= self.graph_conv_2(x_2, edge_index,num_node)
#         #     re.append(re2)
#         #
#         # for data in loader3:
#         #     x_3 = data.x
#         #     x_3,  re3,p3 = self.graph_conv_3(x_3, edge_index,num_node)
#         #     re.append(re3)
#         #
#         # for data in loader4:
#         #     x_4= data.x
#         #     x_4, re4,p4 = self.graph_conv_4(x_4, edge_index,num_node)
#         #     re.append(re4)
#         #
#         # for data in loader5:
#         #     x_5 = data.x
#         #     x_5,re5,p5 = self.graph_conv_5(x_5, edge_index,num_node)
#         #     re.append(re5)
#         #
#         # for data in loader6:
#         #     x_6= data.x
#         #     x_6, re6 ,p6= self.graph_conv_6(x_6, edge_index,num_node)
#         #     re.append(re6)
#         #
#         # for data in loader7:
#         #     x_7= data.x
#         #     x_7, re7 ,p7= self.graph_conv_7(x_7, edge_index,num_node)
#         #     re.append(re7)
#
#         # p=torch.cat([p0,p1,p2,p3,p4,p5,p6,p7],dim=-1)
#         # p=torch.sigmoid(p0)
#             # p_list.append(p)
#         # print(p.size())
#         # print(x_0.size())
#         hidden=x_0
#
#         # hidden=torch.cat([x_0*p[:,0].unsqueeze(-1),x_1*p[:,1].unsqueeze(-1),x_2*p[:,2].unsqueeze(-1),x_3*p[:,3].unsqueeze(-1),
#         #                   x_4*p[:,4].unsqueeze(-1),x_5*p[:,5].unsqueeze(-1),x_6*p[:,6].unsqueeze(-1),x_7*p[:,7].unsqueeze(-1)],dim=-1)
#
#         # print(hidden.size())
#
#         ##输入：[batch*num_nodes,8*(dilation_channels+gcn_channels)]  [13248, 512]
#         # 输出：[batch,num_nodes,time_step]  [64, 207, 12]
#         predict,mu,logvar=self.vae(hidden,num_node)
#         # print(predict.size())
#
#         predict=predict.permute(0,2,1).unsqueeze(-1)##为了适应外部的矩阵形状
#         # print(predict.size())##[64, 12, 207, 1]
#
#         return predict,mu,logvar,re

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        # print(X.shape)
        temp = self.conv1(X) +torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)

        self.Theta1 = nn.Parameter(torch.DoubleTensor(out_channels, spatial_channels))

        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", A_hat, t.permute(1, 0, 2, 3))

        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = F.relu(self.temporal2(t2))

        # t3=self.batch_norm(t3)
        return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.num_nodes=num_nodes
        self.block1 = STGCNBlock(in_channels=32, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5)*64,
                               num_timesteps_output)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)



    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        A_hat = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        out1 = self.block1(X, A_hat)

        out2 = self.block2(out1, A_hat)

        out3 = self.last_temporal(out2)

        # out4 = self.fully(out3)


        return out3
        # return out4










