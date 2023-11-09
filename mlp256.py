import os
import copy
import json
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils import AndersonChebyshevDataset
from utils import ToTensor


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


def train(train_loader, validate_loader, input_d, ratio, n_epoch, criterion, lr, device, num_models):
    for chebyshev_i in range(num_models):
        # init model
        model = MyMLP(input_d, ratio=ratio).to(device=device)
        # 第 i 个 model 预测 第i个chebyshev 的系数
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.double()

        for epoch in trange(n_epoch, desc='Training'):
            train_loss = 0.0
            validate_loss = 0.0
            total_sample = 0

            for x_batch, y_batch, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                y_batch = torch.squeeze(y_batch)[:, chebyshev_i]
                loss = criterion(y_pred, y_batch)
                # backward
                opt.zero_grad()
                loss.backward()
                # update parameters
                opt.step()

                if 1:
                    # record loss and accuracy
                    train_loss += loss.detach().cpu().item()
                    total_sample += len(x_batch)
            print(f" epoch : {epoch+1}/{n_epoch}  train loss: {train_loss / total_sample:.10f}, train sample: {total_sample}")
            
            # validate loop
            validate_sample = 0
            for x_v, y_v, _ in tqdm(validate_loader, desc=f'epoch {epoch+1} in validating', leave=False):
                x_v = x_v.to(device)
                y_v = y_v.to(device)
                yv_pred = model(x_v)
                y_v = torch.squeeze(y_v)[:, chebyshev_i]
                loss_v = criterion(yv_pred, y_v)

                # TODO plot
                if 1: 
                    validate_loss += loss_v.detach().cpu().item()
                    validate_sample += len(x_v)

            print(f" epoch : {epoch+1}/{n_epoch}  validate loss: {validate_loss / validate_sample:.10f}, validate sample : {validate_sample}")

        # save model
        torch.save(model.state_dict(),
                   f'./mlp_models/chebyshev_{chebyshev_i}.pt')

import h5py

def get_alpha(model, plot_loader,  num_models, device=None, spectrum_filename=None):
    alphas = np.array([])
    if spectrum_filename:
        hf = h5py.File(spectrum_filename, 'a')
        if 'mlp_alphas' not in hf.keys():
            alphas = _predict_alpha(model, plot_loader, num_models, device)
            hf.create_dataset(name='mlp_alphas', data=alphas, dtype='float64')
        hf.close()

        # read data
        hf = h5py.File(spectrum_filename, 'r')
        alphas = hf['mlp_alphas'][:]
        hf.close()
    else:
        alphas = _predict_alpha(model, plot_loader, num_models, device)

    return alphas


def _predict_alpha(model, plot_loader,  num_models, device=None):
    alphas = np.array([])

    # compute mlp alphas
    for chebyshev_i in range(num_models):
        model_i = get_model(
            pre_name=config["pre_name"], model_skeleton=model, order=chebyshev_i)
        model_i = model_i.to(device)
        model_i.double()

        n_alphas = np.array([])
        test_sample = 0
        with torch.no_grad():
            for para  in tqdm(plot_loader, desc=f'predict paras nn alphas', leave=False):
                para = para.to(device)
                # TODO  we should modify paras as dataloader to user predict data.
                y_pred = model_i(para)           # (batch_n, 1)
                y_np = y_pred.cpu().numpy()
                n_alphas = np.row_stack(
                    (n_alphas, y_np)) if n_alphas.size else y_np

        alphas = np.column_stack(
            (alphas, n_alphas)) if alphas.size else n_alphas

    return alphas


def test(model, plot_loader,  num_models, criterion=None, device=None):
    for chebyshev_i in range(num_models):
        model_i = get_model(
            pre_name=config["pre_name"], model_skeleton=model, order=chebyshev_i)
        model_i = model_i.to(device)
        model_i.double()

        test_loss = 0.0
        test_sample = 0
        with torch.no_grad():
            for x_batch, y_batch, _ in tqdm(plot_loader, desc=f'testing', leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model_i(x_batch)           # (batch_n, 1)
                y_batch = torch.squeeze(y_batch, dim=1)[:, chebyshev_i]
                loss = criterion(y_pred, y_batch)
                test_loss += loss.detach().cpu().item()
                test_sample += len(x_batch)
            print(f"for {chebyshev_i}th order, test loss : {test_loss / test_sample:.10f}, test sample: {test_sample}")


def get_model(pre_name, model_skeleton, order=0):
    # this method is specific in each model
    model_name = f'./mlp_models/{pre_name}_{order}.pt'
    model_skeleton.load_state_dict(torch.load(model_name))
    return model_skeleton


def get_models(pre_name, model_skeleton, num_models):
    # this method is specific in each model
    models = np.array([])
    for chebyshev_i in range(num_models):
        model_i = copy.deepcopy(model_skeleton)
        model_name = f'./mlp_models/{pre_name}_{chebyshev_i}.pt'
        model_i.load_state_dict(torch.load(model_name))
        models = np.row_stack(
            (models, model_i)) if models.size else np.array([[model_i]])

    return models


torch.manual_seed(123)
train_model = False

from utils import plot_spectrum, load_config
from utils import AndersonParas


if __name__ == "__main__":
    # hyper-parameters
    config = load_config('config_L6_1.json')
    L, N = config["L"], config["N"]
    SIZE = config["SIZE"]
    RATIO = config["RATIO"]
    N_EPOCHS = config["N_EPOCHS"]
    LR = config["LR"]
    batch_size = config['batch_size']

    pre_name = config["pre_name"]
    # num_models 是 N+1, (1, x, 2-x, 3-x, N-x)
    num_models = N+1

    training_size = int(config["SIZE"] * 0.8)       # training: testing = 8: 2
    training_file = os.path.join('datasets', f"L{L}N{N}_training_{training_size}.csv")
    testing_file = os.path.join('datasets', f"L{L}N{N}_testing_{SIZE - training_size}.csv")
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
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size*2)
    validate_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    criterious = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    if train_model:
        train(train_loader, 
              validate_loader,
              input_d=input_d, 
              ratio=RATIO, 
              n_epoch=N_EPOCHS,
              criterion=criterious, 
              lr=LR,
              device=device, 
              num_models=num_models)
    model = MyMLP(input_d, RATIO)
    # test(model, test_loader, num_models, criterion=criterious, device=device)
    # plot anderson parameters save in paras.csv
    paras = config['paras']
    plot_set = AndersonParas(csv_file=paras, L=L)
    plot_loader = DataLoader(plot_set, shuffle=False, batch_size=32)        # just one batch
    spectrum_f = config['spectrum_paras']
    alphas = get_alpha(model, plot_loader=plot_loader, num_models=256, device=device,spectrum_filename=spectrum_f)
    plot_spectrum(spectrum_filename=spectrum_f, nn_alpha=alphas)
