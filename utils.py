import numpy as np
import h5py
from json import load
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

def load_config(config_name):
    with open(config_name) as f:
        config = load(f)
    return config


class AndersonDataset(Dataset):
    # anderson and chevbyshev datasets

    def __init__(self, h5_file, device, l=6, n=255, transform=None):
        super(AndersonDataset, self).__init__()
        self.L = l
        self.n = n
        self.device = device
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

        sample = (anderson.to(self.device), 
                  chebyshev.to(self.device), 
                  Greens.to(self.device), 
                  chebyshev_origin.to(self.device))
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


