import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
from torch_geometric.nn import GCNConv
from median_pyg import MedianConv
torch.set_default_tensor_type(torch.DoubleTensor)
# class TCN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.filter_convs = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(1, 1), dilation=1)
#
#         self.gate_convs = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(1, 1), dilation=1)
#
#         self.residual_convs = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(1, 1))
#
#         self.skip_convs = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(1, 1))
#
#         self.start_conv = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=(1, 1))
#
#         self.bn = nn.BatchNorm2d(32)
#
#     def forward(self, input):
#
#         x = self.start_conv(input)
#         residual = x
#         # dilated convolution
#         filter = self.filter_convs(residual)
#         filter = torch.tanh(filter)
#         gate = self.gate_convs(residual)
#         gate = torch.sigmoid(gate)
#         x = filter * gate
#         # parametrized skip connection
#
#         # x = self.residual_convs(x)
#         # x = x + residual[:, :, :, -x.size(3):]
#         # x = self.bn(x)
#         # x = F.relu(skip)
#         # x = F.relu(self.end_conv_1(x))
#         # x = self.end_conv_2(x)
#         return x

class TCN(nn.Module):
    def __init__(self,  in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2):
        super().__init__()


        # dilated convolutions
        self.filter_convs=nn.Conv2d(in_channels=residual_channels,
                                           out_channels=dilation_channels,
                                           kernel_size=(1, 2), dilation=1)

        self.gate_convs=nn.Conv1d(in_channels=residual_channels,
                                         out_channels=dilation_channels,
                                         kernel_size=(1, 2), dilation=1)


        # 1x1 convolution for skip connection
        # self.bn=nn.BatchNorm2d(residual_channels)

        self.start_conv = nn.Conv1d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

    def forward(self, input):

        x = self.start_conv(input)

        # calculate the current adaptive adj matrix once per iteration

        residual = x
        # print(x.shape)(64,32,1,12)
        # dilated convolution
        filter = self.filter_convs(residual)
        filter = torch.tanh(filter)
        gate = self.gate_convs(residual)
        gate = torch.sigmoid(gate)
        x = filter * gate
        x = x + residual[:, :, :, -x.size(3):]
            # x = self.bn[i](x)

        # x = F.relu(skip)
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        return x
class outlayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.end_conv_1 = nn.Conv2d(in_channels=32,
        #                             out_channels=512,
        #                             kernel_size=(1, 1),
        #                             bias=True)
        #
        # self.end_conv_2 = nn.Conv2d(in_channels=512,
        #                             out_channels=2,
        #                             kernel_size=(1, 1),
        #                             bias=True)
        self.fully = nn.Linear(9*32, 12)
    def forward(self,x):
        # x=F.relu(self.end_conv_1(x))
        # x=self.end_conv_2(x)
        # x = x.permute(0, 2, 1, 3)
        # x = x.reshape(64, 9 * 32)
        x=self.fully(x)
        return x

class client_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder=TCN()
        self.decoder=outlayer()

    def _format_input_data(self, data):
        x, y = data['x'], data['y']
        # N x T x F
        x = x.unsqueeze(-1).permute(0,2,3,1).to(torch.double)

        y = y.unsqueeze(-1).permute(0,2,3,1).to(torch.double)
        # print(x.type(), y.type())  # (64,12,2)
        batch_num, node_num = x.shape[0], x.shape[2]
        return x, y, batch_num, node_num

    def forward_encoder(self, data):
        x,  y,  batch_num, node_num = self._format_input_data(data)
        # T x (B x N) x F
        h_encode = self.encoder(x)
        return h_encode # L x (B x N) x F

    def forward_decoder(self, data, h_encode, return_encoding, server_graph_encoding):
        x,  y, batch_num, node_num = self._format_input_data(data)
        x_input = x
        encoder_h = h_encode

        graph_encoding = server_graph_encoding

        graph_encoding=graph_encoding.reshape((graph_encoding.shape[0],-1))

        out= self.decoder(graph_encoding)

        out = out.unsqueeze(-1)
        return out

    def forward(self, data, return_encoding, server_graph_encoding):
        h_encode = self.forward_encoder(data)
        return self.forward_decoder(data, h_encode, return_encoding, server_graph_encoding)


class graphmodel(nn.Module):

    def __init__(self,num_nodes):
        super().__init__()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.layers=2
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to('cuda'), requires_grad=True).to('cuda')
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to('cuda'), requires_grad=True).to('cuda')
        for i in range(2):
            new_dilation=1
            # dilated convolutions
            self.filter_convs.append(nn.Conv2d(in_channels=32,
                                               out_channels=32,
                                               kernel_size=(1, 2), dilation=new_dilation))

            self.gate_convs.append(nn.Conv1d(in_channels=32,
                                             out_channels=32,
                                             kernel_size=(1, 2), dilation=new_dilation))

            self.bn.append(nn.BatchNorm2d(32))
            new_dilation *= 2
        self.gcn1=GCNConv(32,32)
        self.gcn2=GCNConv(32,32)

    def forward(self,x):

        x=x.to('cuda')
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adp=adp.cpu().detach().numpy()
        edge_index=torch.tensor(np.where(adp>0),dtype=torch.long).to('cuda')
        #64, 207, 11, 32])
        # x=x[:,:,0:2,:]
        # print(x.shape)
        res=x
        x=x.permute(0,2,1,3)
        x=x.reshape(res.shape[0]*res.shape[2],res.shape[1],res.shape[3])#(64*11,207,32)
        x=self.gcn1(x,edge_index)
        x=x.reshape(res.shape[0],res.shape[2],res.shape[1],res.shape[3])
        x=x.permute(0,3,2,1)

        for i in range(self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = x + residual[:, :, :, -x.size(3):]
        # print(x.shape)[64, 32, 207, 9]
        y=x.permute(0,3,2,1)
        y=y.reshape(x.shape[0]*x.shape[3],x.shape[2],x.shape[1])
        y=self.gcn2(y,edge_index)
        y = y.reshape(x.shape[0],x.shape[3], x.shape[2], x.shape[1])
        # print(y.shape)[64, 9, 207, 32])

        return y

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            #(dilation, init_dilation) = self.dilations[i]
            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x




