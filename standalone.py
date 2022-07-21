from argparse import ArgumentParser
from multiprocessing import cpu_count
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import to_dense_adj
from torch.utils.data import TensorDataset



import numpy as np


def pad_with_last_sample( xs, ys, batch_size):
    """
    :param xs:
    :param ys:
    :param batch_size:
    :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
    """



    num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
    x_padding = np.repeat(xs[-1:], num_padding, axis=0)
    y_padding = np.repeat(ys[-1:], num_padding, axis=0)
    xs = np.concatenate([xs, x_padding], axis=0)

    ys = np.concatenate([ys, y_padding], axis=0)

    return xs,ys



def unscaled_metrics(y_pred, y, scaler, null_val=np.nan):
    y = scaler.inverse_transform(y.detach().cpu())
    # y=y.detach().cpu()
    y_pred = scaler.inverse_transform(y_pred.detach().cpu())
    # mse
    if np.isnan(null_val):
        mask = ~torch.isnan(y)
    else:
        mask = (y != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    mse = ((y_pred - y) ** 2).mean()

    mae = torch.abs(y_pred - y).mean()
    # MAPE
    rmse=torch.sqrt(mse)
    # y=torch.clip(y,2,None)
    # mape = torch.abs(y_pred - y) / torch.clip(torch.abs(y), 1, None)
    mape = torch.abs((y_pred - y))/ (y+1)
    mape = mape * mask
    mape = torch.mean(torch.where(torch.isnan(mape), torch.zeros_like(mape), mape))

    return {
        'mse': mse.detach(), 'rmse': rmse.detach(), 'mae': mae.detach(),  'mape': mape.detach()
    }
    # return {
    #     '{}/mse'.format(name): mse.detach(),
    #     # '{}/rmse'.format(name): rmse.detach(),
    #     '{}/mae'.format(name): mae.detach(),
    #     '{}/mape'.format(name): mape.detach()
    # }

