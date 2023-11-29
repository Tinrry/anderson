from json import load
from typing import Any
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# FIXME nor 增加 y_origin
class AndersonDataset(Dataset):
    # anderson and chevbyshev datasets

    def __init__(self, h5_file, l=6, n=255, transform=None):
        super(AndersonDataset, self).__init__()
        self.L = l
        self.n = n
        dataset = h5py.File(h5_file, 'r')
        t_data = lambda x: torch.tensor(np.array(x, dtype=np.float64), dtype=torch.float64)

        self.anderson = torch.tensor([])
        self.chebyshev = torch.tensor([])
        self.Greens = torch.tensor([])
        self.chebyshev_origin = torch.tensor([])

        self.anderson = t_data(dataset['anderson'])
        # 与plot param 数据集兼容。
        if 'chebyshev' in dataset.keys():
            self.chebyshev = t_data(dataset['chebyshev'])
            if 'chebyshev_origin' in dataset.keys():
                # FIXME 在norm格式的数据集里，保存原始的Chebyshev系数，为了查看loss。
                self.chebyshev_origin = t_data(dataset['chebyshev_origin'])
            else:
                self.chebyshev_origin = t_data(dataset['chebyshev'])
        if 'Greens' in dataset.keys():
            self.Greens = t_data(dataset['Greens'])
                
        self.transform = transform  

    def __len__(self):
        return len(self.anderson)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if len(self.anderson) != 0:
            anderson = self.anderson[index, :, :, :]
        else:
            anderson = torch.tensor([])
        if len(self.chebyshev) != 0:
            chebyshev = self.chebyshev[index, :, :, :]
        else:
            chebyshev = torch.tensor([])
        if len(self.Greens) != 0:
            Greens = self.Greens[index, :, :, :]
        else:
            Greens = torch.tensor([])
        if len(self.chebyshev_origin) != 0:
                chebyshev_origin = self.chebyshev_origin[index, :, :, :]
        else:
            chebyshev_origin = torch.tensor([])

        sample = (anderson, chebyshev, Greens, chebyshev_origin)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    

class Normalize(object):
    """anderson input feature normalize"""

    def __call__(self, sample):
        anderson = sample[0]
        chebyshev = sample[1]
        Greens = sample[2]
        chebyshev_origin = sample[3]

        # todo, mean and std should read from 4000_mean_std.h5
        # 对一个图片进行处理 (c, h, w)
        # n_anderson = transforms.Normalize(mean=mean[0], std=std[0])(anderson)
        # n_chebyshev = transforms.Normalize(mean=mean[1], std=std[1])(chebyshev)

        return (anderson, chebyshev, Greens, chebyshev_origin)     


# 公共函数
def plot_spectrum(spectrum_filename, nn_alpha, n_pic=1):
    # plot Greens
    # plot chebyshev, TF, by alphas--labels
    # plot chebyshev, TF, by alphas--nn-predict
    hf = h5py.File(spectrum_filename, 'r')
    # meta_len = config["N"] + 1
    omegas = hf['omegas'][:]
    T_pred = hf['T_pred'][:]
    x_grid = hf['x_grid'][:]
    Tfs = hf['Tfs'][:]
    Greens = hf['Greens'][:]

    # compute spectrum by alpha
    nn_Tfs = np.array([])
    for idx in range(len(Tfs)):
        nn_a = nn_alpha[idx]
        # compute chebyshev function
        nn_Tf = T_pred @ nn_a
        nn_Tfs = np.row_stack((nn_Tfs, nn_Tf)) if nn_Tfs.size else nn_Tf
        if len(nn_Tfs.shape) == 1:
            nn_Tfs = np.expand_dims(nn_Tfs, axis=0)

    fig = plt.figure()
    omegas = np.squeeze(omegas, axis=0)
    x_grid = np.squeeze(x_grid, axis=0)
    for i in range(n_pic):
        axs = fig.add_subplot(1, n_pic, i+1)
        axs.plot(omegas, Greens[i], color='r')
        axs.plot(x_grid, Tfs[i], color='g')
        axs.plot(x_grid, nn_Tfs[i], color='b')
    fig.suptitle('Greens, cheby_Tfs, nn_Tfs, spectrum plot')
    plt.show()


def plot_loss(train_loss, valitate_loss, test_loss: int):
    steps = np.array(range(len(train_loss)))
    plt.plot(steps+1, train_loss, '-o', label='train loss')
    plt.plot(steps+1, valitate_loss, '-o', label='validate loss')
    plt.plot(steps[-1]+1, test_loss, '1', label='test loss')
    plt.legend()
    plt.show()

def plot_loss_from_h5(filename,ncols=5, nrows=4):
    h5 = h5py.File(filename, 'r')
    fig = plt.figure()
    idx = 0
    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
                ax_i = fig.add_subplot(nrows, ncols, idx+1)
                train = h5[f'model_{idx:03}']['train'][:]
                validate = h5[f'model_{idx:03}']['validate'][:]
                test = h5[f'model_{idx:03}']['test'][:]
                ax_i.plot(np.array(range(len(train)))+1, train, '-o', label='train loss')
                ax_i.plot(np.array(range(len(validate)))+1, validate, '-o', label='validate loss')
                ax_i.plot(len(train), test[0], '1', label='test loss')
                idx += 1
                ax_i.legend()
    plt.show()
    h5.close()


def plot_loss_scale(filename, chebyshev_i=0):
    h5 = h5py.File(filename, 'r')
    fig = plt.figure()
    idx = 0
    loss_scale = np.array([0, 10, 20, 30])         # 前面的loss太高了
    
    for begin in loss_scale:
        if begin >= h5[f'model_{chebyshev_i:03}']['train'].shape[0]:
            break
        ax_i = fig.add_subplot(1, len(loss_scale), idx+1)
        train = h5[f'model_{chebyshev_i:03}']['train'][begin:]
        validate = h5[f'model_{chebyshev_i:03}']['validate'][begin:]
        # test = h5[f'model_{chebyshev_i:03}']['test'][:]
        ax_i.plot(np.array(range(len(train)))+1+begin, train, '-o', label='train loss')
        ax_i.plot(np.array(range(len(validate)))+1+begin, validate, '-o', label='validate loss')
        # ax_i.plot(len(train)+begin, test[0], '1', label='test loss')
        idx += 1
    fig.legend()
    savename = filename.split('.')[0]
    fig.savefig(savename + '.png')
    plt.show()
    h5.close()


def plot_retrain_loss_scale(filename, chebyshev_i=0):
    h5 = h5py.File(filename, 'r')
    fig = plt.figure()
    idx = 0
    loss_scale = np.array([0, 5, 10, 15, 20])         # 前面的loss太高了
    
    for begin in loss_scale:
        if begin >= h5[f'model_{chebyshev_i:03}']['retrain'].shape[0]:
            break
        ax_i = fig.add_subplot(1, len(loss_scale), idx+1)
        train = h5[f'model_{chebyshev_i:03}']['retrain'][begin:]
        validate = h5[f'model_{chebyshev_i:03}']['revalidate'][begin:]
        ax_i.plot(np.array(range(len(train)))+1+begin, train, '-o', label='train loss')
        ax_i.plot(np.array(range(len(validate)))+1+begin, validate, '-o', label='validate loss')
        idx += 1
    fig.legend()
    savename = filename.split('.')[0]
    fig.savefig(savename + '.png')
    plt.show()
    h5.close()


def load_config(config_name):
    with open(config_name) as f:
        config = load(f)
    return config