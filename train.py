#################################################
#训练使用这个文件，这个文件设置好了训练的epoch、batch_size,learning_rate等，训练完成之后会在训练集测试，
#按照训练集70%，验证集10%，测试集20%的比例划分
#本文件计算了多hop邻接矩阵，需要使用多hop的话只要吧注释部分消除即可
#本文件174行之后 写了训练完成之后的测试，如果模型输出的个数改变需要改一下这部分的输出个数，
##########################
import torch
import numpy as np
import argparse
import time
import util
import copy
import torch.nn as nn
import matplotlib;matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from engine import trainer
import seaborn as sns
import pandas as pd
import matplotlib.mlab as mlab

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='scalap',help='adj type')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=80,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
args = parser.parse_args()

def combine_params_cloud(para_A,para_B):
    w_cloud = []
    w_cloud.append(copy.deepcopy(para_A))
    w_cloud.append(copy.deepcopy(para_B))
    # w_cloud.append(copy.deepcopy(para_C))
    # w_cloud.append(copy.deepcopy(para_D))
    w_cloud_avg = copy.deepcopy(w_cloud[0])
    for k in w_cloud_avg.keys():
        for i in range(1, len(w_cloud)):
            w_cloud_avg[k] += w_cloud[i][k]
        w_cloud_avg[k] = torch.true_divide(w_cloud_avg[k], len(w_cloud))
    return w_cloud_avg
def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    print(adj_mx.shape)
    # for i in adj_mx:
    #     print(i)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(supports[0].shape)
    # mx=util.first_approx(adj_mx,207)
    # mx=torch.tensor(mx).to(torch.float).to(device)
    # adj_mx = np.array(adj_mx)
    # adj_mx = np.squeeze(adj_mx)
    # mx = adj_mx - np.diag(np.ones(adj_mx.shape[0]))
    # mx = mx / np.max(mx)
    # print(adj_mx.size()
    # fig,ax=plt.subplots()
    # im=ax.imshow(mx)
    # plt.show()

    # engine = trainer(scaler,  in_dim=args.in_dim, lrate=args.learning_rate,
    #                  wdecay=args.weight_decay, device=args.device, batch=args.batch_size)

    # # mask = np.ones_like(mx) - np.diag(np.ones(mx.shape[0]))
    # print("start training...",flush=True)
    # his_loss =[]
    # val_time = []
    # train_time = []
    # init_paramA=engine.modelA1.state_dict()
    # init_param=engine.modelA8.state_dict()
    # params_A_avg = init_paramA
    # params_B_avg = init_paramA
    # params_C_avg = init_param
    # # params_A1_avg = init_paramA
    # # params_B1_avg = init_paramA
    # # params_C_avg = init_paramA
    # # params_D_avg = init_paramA
    # # params_C_avg = engine.modelA8.state_dict()
    #
    # #edge server A
    # for i in range(1,21):
    #     # params_A1 = engine.train_and_testA1(dataloader,edge_index_a1, 'A1')
    #     params_A1 = engine.train_and_testA1(dataA1['train'], dataA1['val'], edge_index_a1, params_A_avg, 'A1')
    #     params_A2 = engine.train_and_testA2(dataA2['train'], dataA2['val'], edge_index_a2, params_A_avg,'A2')
    #     params_A3 = engine.train_and_testA3(dataA3['train'], dataA3['val'], edge_index_a3, params_A_avg, 'A3')
    #     params_A4 = engine.train_and_testA4(dataA4['train'], dataA4['val'], edge_index_a4, params_A_avg, 'A4')
    #     params_B5 = engine.train_and_testA5(dataA5['train'], dataA5['val'], edge_index_a5, params_B_avg, 'A5')
    #     params_B6 = engine.train_and_testA6(dataA6['train'], dataA6['val'], edge_index_a6, params_B_avg, 'A6')
    #     params_B7 = engine.train_and_testA7(dataA7['train'], dataA7['val'], edge_index_a7, params_B_avg, 'A7')
    #     params_B8 = engine.train_and_testA8(dataA8['train'], dataA8['val'], edge_index_a8, params_C_avg, 'A8')
    #     # if (i == 1):
    #     #     params_B4 = params_A4
    #     #     params_A5 = params_B5
    #     # if (i > 1):
    #     #     params_B4 = engine.train_and_testA4s(dataA4['train'], dataA4['val'], edge_index_a4, params_A_avg, 'A4')
    #     #     params_A5 = engine.train_and_testA5s(dataA5['train'], dataA5['val'], edge_index_a5, params_B_avg, 'A5')
    #     # params_A3 = engine.train_and_testA3(dataA3, dataA3_test, edge_index_a3, params_A_avg,'A3')
    #     params_A_avg=combine_paramsA(params_A1,params_A2,params_A3,params_A4)
    #     params_B_avg=combine_paramsB(params_B5,params_B6,params_B7,params_B8)
    #     params_C_avg=params_B_avg
    #     # params_A1_avg = params_A_avg
    #     # params_B1_avg = params_B_avg
    #     if i%2==0:
    #         params_A_avg = combine_params_cloud(params_A_avg,params_B_avg)
    #         params_B_avg = params_A_avg
    #         params_C_avg=params_B_avg
    #
    #     # params_C_avg = combine_paramsB(params_A5,params_A6)
    #     # params_D_avg = combine_paramsB(params_A7, params_A8)
    #     # params_E_avg = params_D_avg
    #     torch.save(params_A_avg,
    #                r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\A_avg\a' + "_epoch_" + str(i)+'_' ".pth")
    #     log="federated avg finished"
    #     print(i,log)
        # if i%2==0:
        #     params_A_avg= combine_params_cloud(params_A_avg,params_B_avg,params_C_avg,params_D_avg)
        #     params_B_avg = params_A_avg
        #     params_C_avg = params_A_avg
        #     params_D_avg = params_A_avg
        #     params_E_avg = params_A_avg
            # params_B1_avg = params_A_avg
            # params_C_avg = params_A_avg

    #
    # outputs = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy.transpose(1, 3).permute(0,2,3,1)[:, :, :, 0]
    # params_A_avg=torch.load(r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\A_avg\a_epoch_17_.pth')
    # engine.gloabmodel.load_state_dict(params_A_avg)
    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1, 3)
    #     testx = testx.permute(0,2,3,1)
    #     with torch.no_grad():
    #         preds = engine.gloabmodel(mx, testx)
    #
    #         # preds=preds.transpose(1,3)
    #     outputs.append(preds.squeeze())
    # yhat = torch.cat(outputs, dim=0)
    # yhat = yhat[:realy.size(0), ...]
    # pred = scaler.inverse_transform(yhat[3000:3288, 20, 0])
    #
    # outputs = []
    # params_A_avg = torch.load(r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\A_avg\a_epoch__.pth')
    # engine.gloabmodel.load_state_dict(params_A_avg)
    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1, 3)
    #     testx = testx.permute(0, 2, 3, 1)
    #     with torch.no_grad():
    #         preds = engine.gloabmodel(mx, testx)
    #         # preds=preds.transpose(1,3)
    #     outputs.append(preds.squeeze())
    # yhat1 = torch.cat(outputs, dim=0)
    # yhat1 = yhat1[:realy.size(0), ...]
    # pred1 = scaler.inverse_transform(yhat1[3000:3288, 20, 0])
    #
    # outputs = []
    # params_A_avg = torch.load(r'C:\Users\Liu Lei\Desktop\spatial-temporal paper\A_avg\a_epoch_4_.pth')
    # engine.gloabmodel.load_state_dict(params_A_avg)
    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1, 3)
    #     testx = testx.permute(0, 2, 3, 1)
    #     with torch.no_grad():
    #         preds = engine.gloabmodel(mx, testx)
    #         # preds=preds.transpose(1,3)
    #     outputs.append(preds.squeeze())
    # yhat2 = torch.cat(outputs, dim=0)
    # yhat2= yhat2[:realy.size(0), ...]
    # pred2= scaler.inverse_transform(yhat2[3000:3288, 20, 0])
    #
    # real = realy[3000:3288, 20, 0]
    # pred =pred.reshape([288]).cpu().numpy()
    # pred1 = pred1.reshape([288]).cpu().numpy()
    # pred2 = pred2.reshape([288]).cpu().numpy()
    # real = real.reshape([288]).cpu().numpy()
    # index= np.arange(0,288,1)
    # plt.plot(index, pred, color='red', linewidth=1,label='MFVSTGNN(STGCN)')
    # plt.plot(index, pred1, color='yellow', linewidth=1, label='MFL+STGCN')
    # plt.plot(index, pred2, color='green', linewidth=1, label='Cloud-FL+STGCN')
    # plt.plot(index, real, color='blue', linewidth=1,label='Ground truth')
    # plt.legend()
    # plt.show()
    # amae = []
    # amape = []
    # armse = []
    # for i in range(12):
    #     pred = scaler.inverse_transform(yhat[:, :, i])
    #     real = realy[:, :, i]
    #     metrics = util.metric(pred, real)
    #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
    #     amae.append(metrics[0])
    #     amape.append(metrics[1])
    #     armse.append(metrics[2])
    #
    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    # torch.save(engine.model.state_dict(),
    #            args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")


#     # #testing
#     # bestid = np.argmin(his_loss)
#     # engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
#     #
#     #
#     # outputs = []
#     # realy = torch.Tensor(dataloader['y_test']).to(device)
#     # realy = realy.transpose(1,3)[:,0,:,:]
#     #
#     # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
#     #     testx = torch.Tensor(x).to(device)
#     #     testx = testx.transpose(1,3)
#     #     testx=nn.functional.pad(testx, (1, 0, 0, 0))
#     #
#     #     with torch.no_grad():
#     #         preds,_,_, _= engine.model(testx,edge_index)
#     #         preds=preds.transpose(1,3)
#     #     outputs.append(preds.squeeze())
#     #
#     # yhat = torch.cat(outputs,dim=0)
#     # yhat = yhat[:realy.size(0),...]
#     #
#     #
#     # print("Training finished")
#     # print("The valid loss on best model is", str(round(his_loss[bestid],4)))
#     #
#     #
#     # amae = []
#     # amape = []
#     # armse = []
#     # for i in range(12):
#     #     pred = scaler.inverse_transform(yhat[:,:,i])
#     #     real = realy[:,:,i]
#     #     metrics = util.metric(pred,real)
#     #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#     #     print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
#     #     amae.append(metrics[0])
#     #     amape.append(metrics[1])
#     #     armse.append(metrics[2])
#     #
#     # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#     # print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
#     # torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
#     #
#
if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
