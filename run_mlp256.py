import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils import AndersonChebyshevDataset
from utils import ToTensor
from utils import plot_spectrum, load_config, plot_loss_scale, plot_retrain_loss_scale
from utils import AndersonParas
from mlp256 import MyMLP
from mlp256 import train, test, retrain
from plot_spectrum import get_alpha


if __name__ == "__main__":
    torch.manual_seed(123)
    train_model = True
    model_range = np.arange(1)

    # hyper-parameters
    # config = load_config('config_L6_14.json')
    config = load_config('config_debug.json')
    finetune = config['finetune']
    L, N = config["L"], config["N"]
    SIZE = config["SIZE"]
    RATIO = config["RATIO"]
    N_EPOCHS = config["N_EPOCHS"]
    LR = config["LR"]
    batch_size = config['batch_size']
    pre_name = config["pre_name"]
    training_file = os.path.join('datasets', f"L{L}N{N}_10000.csv")
    testing_file = os.path.join('datasets', f"L{L}N{N}_testing_1000.csv")
    val_size = 500              # we use 500 in validate , 500 in test

    input_d = L + 2
    transform = ToTensor()
    # transform = None
    train_set = AndersonChebyshevDataset(
        csv_file=training_file, L=L, n=N, transform=transform)

    dataset = AndersonChebyshevDataset(
        csv_file=testing_file, L=L, n=N, transform=transform)
    test_size = len(dataset) - val_size
    test_ds, val_ds = random_split(dataset, [val_size, test_size])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size)
    validate_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # 改成相对误差平均
    criterious = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    model = MyMLP(input_d, RATIO)

    if train_model:
        if not finetune:
            train(train_loader, 
                validate_loader,
                input_d=input_d, 
                ratio=RATIO, 
                n_epoch=N_EPOCHS,
                criterion=criterious, 
                lr=LR,
                device=device, 
                model_range=model_range,
                config=config)
            test(model, test_loader, model_range, criterion=criterious, device=device, config=config)
            
            plot_loss_scale(config['config_loss'])
        else:
            retrain(train_loader, 
                validate_loader,
                input_d=input_d, 
                ratio=RATIO, 
                n_epoch=N_EPOCHS,
                criterion=criterious, 
                relr=5e-9,
                device=device, 
                model_range=model_range,
                config=config)
            test(model, test_loader, model_range, criterion=criterious, device=device, config=config)
            plot_retrain_loss_scale(config['re_loss'])
    

    if len(model_range) == 256:
        # plot anderson parameters save in paras.csv
        paras = config['paras']
        plot_set = AndersonParas(csv_file=paras, L=L)
        plot_loader = DataLoader(plot_set, shuffle=False, batch_size=32)        # just one batch
        spectrum_f = config['spectrum_paras']
        alphas_f = config["config_alphas"]
        alphas = get_alpha(model, 
                            plot_loader=plot_loader,
                            num_models=256, 
                            device=device,
                            alpha_filename=alphas_f, 
                            config=config)
        plot_spectrum(spectrum_filename=spectrum_f, nn_alpha=alphas)
