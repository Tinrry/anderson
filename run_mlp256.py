import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils import AndersonDataset
from utils import plot_spectrum, load_config, plot_loss_scale, plot_retrain_loss_scale
from mlp256 import train, test, retrain
from plot_spectrum import get_alpha
from nn_models import MyMLP_7


torch.manual_seed(123)

if __name__ == "__main__":    

    # hyper-parameters
    config = load_config('config_debug.json')
    L, N = config["L"], config["N"]

    model_range = np.arange(int(config['model_order']))
    n_epoch = config["n_epoch"]
    lr = config["lr"]
    batch_size = config['batch_size']
    training_file = config['training_file']

    input_d = L + 2
    train_set = AndersonDataset(
        h5_file=training_file, l=L, n=N)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # 改成相对误差平均
    criterious = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    model = MyMLP_7(input_d)

    # if train_model:
    train(model,
          train_loader, 
          n_epoch=n_epoch,
          criterion=criterious,  
          lr=lr, 
          device=device,  
          model_range=model_range,
          config=config)
