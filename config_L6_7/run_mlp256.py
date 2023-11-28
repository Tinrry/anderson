import os, sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

project_dir = os.path.join('/home/hhzheng/project/anderson')
sys.path.append(project_dir)
sys.path.append(os.getcwd())
from utils import AndersonChebyshevDataset
from utils import ToTensor
from utils import plot_spectrum, load_config, plot_loss_scale, plot_retrain_loss_scale
from utils import AndersonParas
from mlp256 import train, test, retrain
from plot_spectrum import get_alpha


class MyMLP(nn.Module):
    def __init__(self, input_d, ratio=2) -> None:
        super(MyMLP, self).__init__()
        self.input_d = input_d
        self.ratio = ratio

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_d, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d),
            nn.ReLU(),
            nn.Linear(self.input_d, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x_1 = self.linear_relu_stack(x)
        return x_1


if __name__ == "__main__":
    torch.manual_seed(123)
    train_model = False
    model_range = np.arange(256)

    # hyper-parameters
    config = load_config(os.path.join(project_dir, 'config_L6_7/config_L6_7.json'))
    finetune = config['finetune']
    L, N = config["L"], config["N"]
    SIZE = config["SIZE"]
    RATIO = config["RATIO"]
    N_EPOCHS = config["N_EPOCHS"]
    LR = config["LR"]
    batch_size = config['batch_size']
    pre_name = config["pre_name"]
    training_file = os.path.join('datasets', f"L{L}N{N}_training_4000.csv")
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
            train(model,
                  train_loader, 
                  validate_loader,
                  n_epoch=N_EPOCHS,
                  criterion=criterious, 
                  lr=LR,
                  device=device, 
                  model_range=model_range,
                  config=config)
            test(model, test_loader, model_range, criterion=criterious, device=device, model_checkpoint=N_EPOCHS, config=config)
            
            plot_loss_scale(config['config_loss'])
        else:
            retrain(model,
                    train_loader, 
                validate_loader,
                n_epoch=N_EPOCHS,
                criterion=criterious, 
                relr=5e-9,
                device=device, 
                model_range=model_range,
                config=config)
            test(model, test_loader, model_range, criterion=criterious, device=device, model_checkpoint=N_EPOCHS, config=config)
            plot_retrain_loss_scale(config['re_loss'])
    else:
        test(model, test_loader, model_range[:1], criterion=criterious, device=device, model_checkpoint=N_EPOCHS, config=config)

    if len(model_range) == 256:
        # plot anderson parameters save in paras.csv
        csv_file = os.path.join(project_dir, config['paras'])
        plot_set = AndersonParas(csv_file=csv_file, L=L)
        plot_loader = DataLoader(plot_set, shuffle=False, batch_size=32)        # just one batch
        spectrum_f = os.path.join(project_dir, config['spectrum_paras'])
        alphas_f = os.path.join(project_dir, config["config_alphas"])
        alphas = get_alpha(model, 
                            plot_loader=plot_loader,
                            num_models=256, 
                            device=device,
                            alpha_filename=alphas_f, 
                            config=config)
        # 设置画几张图片的频谱，不用全部32张都画出来
        plot_spectrum(spectrum_filename=spectrum_f, nn_alpha=alphas, n_pic=1)
