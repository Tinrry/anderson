import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from utils import AndersonDataset
from utils import AndersonParas
from utils import Normalize
from utils import plot_spectrum, load_config, plot_loss_scale, plot_retrain_loss_scale
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
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
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
    train_model = True
    model_range = np.arange(1)

    # hyper-parameters
    # config = load_config('config_L6_14.json')
    config = load_config('config_debug.json')
    finetune = config['finetune']
    L, N = config["L"], config["N"]
    SIZE = config["SIZE"]
    RATIO = config["RATIO"]
    n_epoch = config["n_epoch"]
    LR = config["LR"]
    batch_size = config['batch_size']
    pre_name = config["pre_name"]
    print_loss_file = config['config_loss']
    print_reloss_file = config['re_loss']
    training_file = os.path.join('datasets', f"L{L}N{N}_norm_4000.h5")
    testing_file = os.path.join('datasets', f"L{L}N{N}_norm_testing_1000.h5")
    val_size = 500              # we use 500 in validate , 500 in test

    input_d = L + 2
    # this transform is each item, so we can not normalize whole set
    # item_norm = Normalize()         # 使用规范化的数据集，就不需要Normalize了。
    train_set = AndersonDataset(
        h5_file=training_file, L=L, n=N)

    test_set = AndersonDataset(
        h5_file=testing_file, L=L, n=N)

    test_size = len(test_set) - val_size
    test_ds, val_ds = random_split(test_set, [val_size, test_size])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
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
                  n_epoch=n_epoch,
                  criterion=criterious, 
                  lr=LR,
                  device=device, 
                  model_range=model_range,
                  config=config)
            test(model, 
                 test_loader, 
                 model_range[:5], 
                 criterion=criterious, 
                 device=device,
                 model_checkpoint=n_epoch,
                 config=config)
            # plot_loss_scale(print_loss_file)
        else:
            retrain(model,
                    train_loader, 
                    validate_loader,
                    n_epoch=n_epoch,
                    criterion=criterious, 
                    relr=5e-9,
                    device=device, 
                    model_range=model_range,
                    config=config)
            test(model, 
                 test_loader, 
                 model_range[:5], 
                 criterion=criterious,
                 device=device, 
                 model_checkpoint=n_epoch,
                 config=config)
            # plot_retrain_loss_scale(print_reloss_file)
    else:
        test(model, 
             test_loader, 
             model_range[:1], 
             criterion=criterious, 
             device=device, 
             model_checkpoint=n_epoch, 
             config=config)
    
    plot_loss_scale(print_loss_file)
    
# 可以考虑成为一个类
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
