import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from utils import AndersonDataset
from utils import load_config
from nn_models import MyMLP_14 as MyMLP
from mlp import MultiLayerP


torch.manual_seed(123)

if __name__ == "__main__":    
    # hyper-parameters
    config = load_config('config_debug.json')

    L, N = config["L"], config["N"]
    model_range = np.arange(int(config['model_order']))
    epochs = config["n_epoch"]
    lr = config["lr"]
    batch_size = config['batch_size']
    training_file = config['training_file']
    testing_file = config['testing_file']
    step_size = config['step_size']
    gamma = config['gamma']
    hdf5_filename = config['loss_file']

    input_d = L + 2
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    train_set = AndersonDataset(h5_file=training_file, device=device, l=L, n=N)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_set, val_set = random_split(AndersonDataset(h5_file=testing_file, device=device, l=L, n=N), [500, 500])

    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)

    # TODO 改成相对误差平均
    loss_function = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    network = MyMLP(input_d)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scehduler= StepLR(optimizer, step_size=step_size, gamma=gamma)
    network.to(device).double()

    model = MultiLayerP(network, 
                        loss_function,
                        chebyshev_model_range=model_range,
                        scheduler=scehduler, 
                        save_hdf5=hdf5_filename)

    train_log_dict = model.train(optimizer=optimizer, 
                           epochs=epochs, 
                           train_loader=train_loader, 
                           val_loader=val_loader)
    test_log_dict = model.test(test_loader)
