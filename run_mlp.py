import os
import numpy as np
import logging
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from utils import AndersonDataset
from utils import load_config
from mlp import MultiLayerP

torch.manual_seed(1)

# log some messages
# debug, info, warning, error, critical

# create a logger object
logger = logging.getLogger(
    " train-mlp"
)
# set the logging level
logger.setLevel(logging.DEBUG)

# create a console handler and set its level
console_handler = logging.StreamHandler()
# console_handler.setLevel(logLevel)

# create formatter and add it to the console handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def train_model(network, chebyshev_i, layer_num, loss_temp_file):
    torch.manual_seed(1)
    L, N = config["L"], config["N"]
    epochs = config["n_epoch"]
    lr = config["lr"]
    batch_size = config['batch_size']
    training_file = config['training_file']
    testing_file = config['testing_file']
    step_size = config['step_size']
    gamma = config['gamma']
    save_model = config['save_model']
    if len(save_model) == 0:
        save_model = None           # 不保存模型
    

    input_d = L + 2
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    train_set = AndersonDataset(h5_file=training_file, device=device, l=L, n=N)
    test_set = AndersonDataset(h5_file=testing_file, device=device, l=L, n=N)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_set, val_set = random_split(test_set, [500, 500])

    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)

    loss_function = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    network = network(input_d, layer_num=layer_num)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scehduler= StepLR(optimizer, step_size=step_size, gamma=gamma)
    network.to(device).double()
    model = MultiLayerP(network, 
                        loss_function,
                        chebyshev_i=chebyshev_i,
                        scheduler=scehduler, 
                        save_log=loss_temp_file,
                        save_model=save_model,
                        logger = logger)

    model.train(optimizer=optimizer, 
                           epochs=epochs, 
                           train_loader=train_loader, 
                           val_loader=val_loader)
    model.test(test_loader)

def main(network, layer_num=7, loss_file=None, logger=logger):  
    
    # 创建一个文件，保存当前的训练结果，成功后再归档到对应的网络架构资料中。
    loss_temp_file = 'loss.h5'
    if os.path.exists(loss_temp_file):
        os.remove(loss_temp_file)

    model_list = re.findall(r"\d+", config['model_range'])
    if len(model_list) == 2:
        begin, end = int(model_list[0]), int(model_list[1])
    else:
        begin, end = 0, int(model_list[0])
    model_range = np.arange(begin, end)

    for chebyshev_i in model_range:
        train_model(network, chebyshev_i, layer_num, loss_temp_file)
    

    # 避免训练一半，把原来的文件给删除了。
    if loss_file is None:
        loss_file = config['loss_file']

    if os.path.exists(loss_file):
        os.remove(loss_file)
    os.rename(src=loss_temp_file, dst=loss_file)


if __name__ == '__main__':
    # from nn_models import MyMLP_14_Batchnorm 
    from nn_models import MyMLP
    # hyper-parameters
    config = load_config('config_model0_cnorm.json') 
    main(network=MyMLP, layer_num=7)