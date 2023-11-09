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
