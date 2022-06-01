###########################################################
#     函数 trainer（） 用于初始化模型以及训练模型，模型修改输出个数之后要修改，约束项也在这部分
#     函数 eval（）  用于测试模型训练结果，模型修改输出的个数  之后这部分模型的输出需要修改

#     模型的输入维数为[batch,node_feature,num_nodes,time_step+1] 做填充[64,2,207,13]
#     模型输出维数为[batch, time_step, num_nodes, node_feature]  [64,12,207,1]

#     计算loss真实值与预测值的维数都要是[batch,node_feature,num_nodes,time_step]，
#     并且预测值要经过一个封装好的scaler类的处理，不然无法计算loss，scaler通过参数的方式传入
#
#########################################################
import copy

import torch.optim as optim
from model import *
import util
import torch.nn as nn
import torch
from model import STGCN
from loss import VAE_loss,reset_loss
from torch_geometric_temporal import STConv
import numpy as np
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--saveA1',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA1\A1',help='save path')
parser.add_argument('--saveA2',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA2\A2',help='save path')
parser.add_argument('--saveA3',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA3\A3',help='save path')
parser.add_argument('--saveA4',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA4\A4',help='save path')
parser.add_argument('--saveA5',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA5\A5',help='save path')
parser.add_argument('--saveA6',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA6\A6',help='save path')
parser.add_argument('--saveA7',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA7\A7',help='save path')
parser.add_argument('--saveA8',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA8\A8',help='save path')
parser.add_argument('--saveA4s',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA4s\A4s',help='save path')
parser.add_argument('--saveA5s',type=str,default=r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\garageA5s\A5s',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
args = parser.parse_args()
class trainer():
    def __init__(self, scaler, in_dim, lrate, wdecay, device,batch):
        ## 调用自己的模型
        self.gloabmodel = STGCN(325, 2, 12, 12)
        self.modelA1 = STGCN(40, 2, 12, 12)
        self.modelA2 = STGCN(40, 2, 12, 12)
        self.modelA3 = STGCN(40, 2, 12, 12)
        self.modelA4 = STGCN(40, 2, 12, 12)
        self.modelA5 = STGCN(40, 2, 12, 12)
        self.modelA6 = STGCN(40, 2, 12, 12)
        self.modelA7 = STGCN(40, 2, 12, 12)
        self.modelA8 = STGCN(43, 2, 12, 12)
        #
        # self.modelA4s=STGCN(40, 2, 12, 12)
        # self.modelA5s = STGCN(40, 2, 12, 12)


        # model=Net(device, num_nodes,num_fea=in_dim,batch=batch)
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(model)
        # self.modelA2 = Net(device, num_fea=in_dim,batch=batch)
        # self.modelA2 = STGCN(70,2,12,12)
        # self.modelD3 = copy.deepcopy(self.modelA1)
        ##定义loss，以及优化器
        self.gloabmodel.to(device)
        self.modelA1.to(device)
        self.modelA2.to(device)
        self.modelA3.to(device)
        self.modelA4.to(device)
        self.modelA5.to(device)
        self.modelA6.to(device)
        self.modelA7.to(device)
        self.modelA8.to(device)
        #
        # self.modelA4s.to(device)
        # self.modelA5s.to(device)

        self.device=device
        self.optimizerA1 = optim.Adam(self.modelA1.parameters(), lr=lrate,weight_decay=wdecay)
        self.optimizerA2 = optim.Adam(self.modelA2.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizerA3 = optim.Adam(self.modelA3.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizerA4 = optim.Adam(self.modelA4.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizerA5 = optim.Adam(self.modelA5.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizerA6 = optim.Adam(self.modelA6.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizerA7 = optim.Adam(self.modelA7.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizerA8 = optim.Adam(self.modelA8.parameters(), lr=lrate, weight_decay=wdecay)
        #
        # self.optimizerA4s = optim.Adam(self.modelA4s.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizerA5s = optim.Adam(self.modelA5s.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = util.masked_mae
        ##scaler是封装好的用于转换预测结果的函数，
        self.scaler = scaler
        ##梯度截断的clip
        self.clip = 5
    def train_and_testA1(self, data_train,data_val,edge_index, param,id):
        self.modelA1.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA1.train()
                self.optimizerA1.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]

                output = self.modelA1(edge_index, trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_2##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA1.parameters(), self.clip)
                self.optimizerA1.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA1.eval()
            ##与train函数一致，对输出做转置，scaler

                output= self.modelA1(edge_index, val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA1.state_dict(),args.saveA1 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA1 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA2(self, data_train, data_val, edge_index, param, id):
        self.modelA2.load_state_dict(param)
        epoches = 5
        his_loss = []
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                # print(trainx.size())  ##[batch_size,2,node_num,12]
                trainx = trainx.transpose(1, 3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy = trainy.transpose(1, 3)
                trainy = trainy.permute(0, 2, 3, 1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA2.train()
                self.optimizerA2.zero_grad()
                ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
                ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]

                output = self.modelA2(edge_index, trainx)

                real = trainy[:, :, :, 0]  ##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
                ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

                # re=self.scaler.inverse_transform(re)

                ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
                # 预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1  #+0.001*loss_2##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA2.parameters(), self.clip)
                self.optimizerA2.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id, log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x = val_x.permute(0, 2, 3, 1)  ##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA2.eval()
                # ##与train函数一致，对输出做转置，scaler
                # output,mu,logvar= self.modelA2(edge_index, val_x)
                output = self.modelA2(edge_index, val_x)

                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id, log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id, log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse,
                                 (t2 - t1)),
                  flush=True)
            torch.save(self.modelA2.state_dict(),
                       args.saveA2 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params = torch.load(args.saveA2 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA3(self, data_train,data_val,edge_index, param,id):
        self.modelA3.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA3.train()
                self.optimizerA3.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output= self.modelA3(edge_index,trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA3.parameters(), self.clip)
                self.optimizerA3.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA3.eval()
            ##与train函数一致，对输出做转置，scaler
                output= self.modelA3(edge_index,val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA3.state_dict(),args.saveA3 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA3 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA4(self, data_train, data_val, edge_index, param, id):
        self.modelA4.load_state_dict(param)
        epoches = 5
        his_loss = []
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx = trainx.transpose(1, 3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy = trainy.transpose(1, 3)
                trainy = trainy.permute(0, 2, 3, 1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA4.train()
                self.optimizerA4.zero_grad()
                ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
                ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output = self.modelA4(edge_index, trainx)

                real = trainy[:, :, :, 0]  ##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
                ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

                # re=self.scaler.inverse_transform(re)

                ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
                # 预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1  # +0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA4.parameters(), self.clip)
                self.optimizerA4.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id, log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x = val_x.permute(0, 2, 3, 1)  ##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA4.eval()
                ##与train函数一致，对输出做转置，scaler
                output = self.modelA4(edge_index, val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id, log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id, log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse,
                                 (t2 - t1)),
                  flush=True)
            torch.save(self.modelA4.state_dict(),
                       args.saveA4 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params = torch.load(args.saveA4 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA5(self, data_train,data_val,edge_index, param,id):
        self.modelA5.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA5.train()
                self.optimizerA5.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output= self.modelA5(edge_index,trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA5.parameters(), self.clip)
                self.optimizerA5.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA5.eval()
            ##与train函数一致，对输出做转置，scaler
                output= self.modelA5(edge_index,val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA5.state_dict(),args.saveA5 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA5 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA6(self, data_train,data_val,edge_index, param,id):
        self.modelA6.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA6.train()
                self.optimizerA6.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output= self.modelA6(edge_index,trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA6.parameters(), self.clip)
                self.optimizerA6.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA6.eval()
            ##与train函数一致，对输出做转置，scaler
                output= self.modelA6(edge_index,val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA6.state_dict(),args.saveA6 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA6 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA7(self, data_train,data_val,edge_index, param,id):
        self.modelA7.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA7.train()
                self.optimizerA7.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output= self.modelA7(edge_index,trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA7.parameters(), self.clip)
                self.optimizerA1.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA7.eval()
            ##与train函数一致，对输出做转置，scaler
                output= self.modelA7(edge_index,val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA7.state_dict(),args.saveA7 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA7 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA8(self, data_train,data_val,edge_index, param,id):
        self.modelA8.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA8.train()
                self.optimizerA8.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output= self.modelA8(edge_index,trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA8.parameters(), self.clip)
                self.optimizerA8.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA8.eval()
            ##与train函数一致，对输出做转置，scaler
                output= self.modelA8(edge_index,val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA8.state_dict(),args.saveA8 + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA8 + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA4s(self, data_train,data_val,edge_index, param,id):
        self.modelA4s.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA4s.train()
                self.optimizerA4s.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output= self.modelA4s(edge_index,trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA4s.parameters(), self.clip)
                self.optimizerA4s.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA4s.eval()
            ##与train函数一致，对输出做转置，scaler
                output= self.modelA4s(edge_index,val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA4s.state_dict(),args.saveA4s + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA4s + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params

    def train_and_testA5s(self, data_train,data_val,edge_index, param,id):
        self.modelA5s.load_state_dict(param)
        epoches=5
        his_loss=[]
        val_time = []
        train_time = []
        # print("input.size   " + str(input.size()))  ##[64, 2, 207, 12]dawan
        for epoch in range(epoches):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1=time.time()
            for iter, data in enumerate(data_train.get_iterator()):
                (x, y) = data
                trainx = torch.Tensor(x).to(self.device)
                trainx=trainx.transpose(1,3)
                trainx = trainx.permute(0, 2, 3, 1)
                trainy = torch.Tensor(y).to(self.device)
                trainy=trainy.transpose(1,3)
                trainy=trainy.permute(0,2,3,1)
                # (batch_size, num_nodes, num_timesteps,
                #  num_features=in_channels).
                self.modelA5s.train()
                self.optimizerA5s.zero_grad()
        ##在输入的时间轴上做0填充，[64, 2, 207, 12]-->[64, 2, 207, 13]
                # input = nn.functional.pad(trainx,(1,0,0,0))    ##[64, 2, 207, 13]
        ##将调用初始化的模型做预测,输出数据size为[64, 12, 207, 1] [batch,time_step,num_nodes,node_feature]
                output= self.modelA5s(edge_index,trainx)

                real = trainy[:,:,:,0]##[64, 1, 207, 12] [batch,node_feature,num_nodes,time_step]
        ##调用用于转换数据的函数scaler，不然无法用于loss的计算
                predict = self.scaler.inverse_transform(output)

        # re=self.scaler.inverse_transform(re)

        ##计算loss的真实值的维数为[batch,node_feature,num_nodes,time_step]
        #预测值的维数为 [batch,node_feature,num_nodes,time_step]
                loss_1 = self.loss(predict, real, 0.0)
                # loss_2=VAE_loss(mu,logvar)
                # loss_5=reset_loss(real,re,self.scaler)
                loss = loss_1 #+0.001*loss_5##+0.5*loss_6
                loss.backward(retain_graph=True)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.modelA5s.parameters(), self.clip)
                self.optimizerA5s.step()

                mape = util.masked_mape(predict, real, 0.0).item()
                rmse = util.masked_rmse(predict, real, 0.0).item()
                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)
                if iter % 50 == 0:
                    log = ' Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(id,log.format(iter, loss.item(), mape, rmse), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, data in enumerate(data_val.get_iterator()):
                (x, y) = data
                val_x = torch.Tensor(x).to(self.device)
                val_x = val_x.transpose(1, 3)
                val_x=val_x.permute(0, 2, 3, 1)##[64, 12, 207, 2]
                val_y = torch.Tensor(y).to(self.device)
                val_y = val_y.transpose(1, 3)
                val_y = val_y.permute(0, 2, 3, 1)
                self.modelA5s.eval()
            ##与train函数一致，对输出做转置，scaler
                output= self.modelA5s(edge_index,val_x)
                real = val_y[:, :, :, 0]
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, real, 0.0)
                mape = util.masked_mape(predict,real,0.0).item()
                rmse = util.masked_rmse(predict,real,0.0).item()
                valid_loss.append(loss.item())
                valid_mape.append(mape)
                valid_rmse.append(rmse)
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(id,log.format(epoch, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(id,log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
            torch.save(self.modelA5s.state_dict(),args.saveA5s + "_epoch_" + str(epoch) + "_" + str(round(mvalid_loss, 4)) + ".pth")
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        bestid = np.argmin(his_loss)
        params= torch.load(args.saveA5s + "_epoch_" + str(bestid) + "_" + str(round(his_loss[bestid], 4)) + ".pth")

        return params