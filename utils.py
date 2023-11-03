from json import load
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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
        print(f"{anderson[0,0]:.15f}", '    ', anderson.dtype)

        return (anderson, chebyshev, Greens)
