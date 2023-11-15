from json import load
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


def load_config(config_name):
    with open(config_name) as f:
        config = load(f)
    return config

class AndersonChebyshevDataset(Dataset):
    # anderson and chevbyshev datasets

    def __init__(self, csv_file, L=6, n=255, transform=None):
        super(AndersonChebyshevDataset, self).__init__()
        self.L = L
        self.n = n
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        anderson = self.data.iloc[index, :self.L+2]
        chebyshev = self.data.iloc[index, self.L+2: self.L + 2 + self.n + 1]
        Greens = self.data.iloc[index, self.L + 2 + self.n + 1: ]
        anderson = np.array([anderson])
        chebyshev = np.array([chebyshev])
        Greens = np.array([Greens])
        sample = (anderson, chebyshev, Greens)

        if self.transform:
            sample = self.transform(sample)

        return sample

class AndersonParas(Dataset):
    def __init__(self, csv_file, L=6):
        super(AndersonParas, self).__init__()
        self.data = pd.read_csv(csv_file, delimiter=',', index_col=0, header=None)
        self.L = L
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        paras = self.data.iloc[index, :]
        paras = torch.DoubleTensor(torch.from_numpy(np.array([paras])))
        sample = (paras)
        
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        anderson_np, chebyshev_np, Greens_np = sample
        anderson = torch.DoubleTensor(torch.from_numpy(anderson_np))
        chebyshev = torch.DoubleTensor(torch.from_numpy(chebyshev_np))
        Greens = torch.DoubleTensor(torch.from_numpy(Greens_np))

        return (anderson, chebyshev, Greens)



# 公共函数
def plot_spectrum(spectrum_filename, nn_alpha, nrows=8, ncols=4):
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

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True)
    omegas = np.squeeze(omegas, axis=0)
    x_grid = np.squeeze(x_grid, axis=0)
    for i in range(nrows):
        for j in range(ncols):
            axs[i, j].plot(omegas, Greens[i * ncols + j], color='r')
            axs[i, j].plot(x_grid, Tfs[i * ncols + j], color='g')
            axs[i, j].plot(x_grid, nn_Tfs[i * ncols + j], color='b')
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
