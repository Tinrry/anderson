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
        self.L = L
        self.n = n
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anderson = self.data.iloc[idx, :self.L+2]
        chebyshev = self.data.iloc[idx, self.L+2: self.L + 2 + self.n + 1]
        Greens = self.data.iloc[idx, self.L + 2 + self.n + 1: ]
        anderson = np.array([anderson])
        chebyshev = np.array([chebyshev])
        Greens = np.array([Greens])
        sample = (anderson, chebyshev, Greens)

        if self.transform:
            sample = self.transform(sample)

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

    nn_alpha = np.array([])

    # compute spectrum by alpha
    nn_Tfs = np.array([])
    for idx in range(len(Tfs)):
        nn_a = nn_alpha[idx]
        # compute chebyshev function
        nn_Tf = T_pred @ nn_a
        nn_Tfs = np.row_stack((nn_Tfs, nn_Tf)) if nn_Tfs.size else nn_Tf
        if len(nn_Tfs.shape) == 1:
            nn_Tfs = np.expand_dims(nn_Tfs, axis=0)

    fig, axs = plt.subplot(nrows, ncols, sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            axs[i, j].plot(omegas, Greens[i * ncols + j], color='r')
            axs[i, j].plot(x_grid, Tfs[i * ncols + j], color='g')
            axs[i, j].plot(x_grid, nn_Tfs[i * ncols + j], color='b')
    fig.suptitle('Greens, cheby_Tfs, nn_Tfs, spectrum plot')
    plt.show()
