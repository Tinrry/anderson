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


def train(train_loader, input_d, ratio, n_epoch, criterion, lr, device, num_models):
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


def predict_alpha(model, plot_loader,  num_models, criterion=None, device=None, test=False):
    alphas = np.array([])
    for chebyshev_i in range(num_models):
        model_i = get_model(
            pre_name=config["pre_name"], model_skeleton=model, order=chebyshev_i)
        model_i = model_i.to(device)
        model_i.double()

        n_alphas = np.array([])
        test_loss = 0.0
        test_sample = 0
        with torch.no_grad():
            for x_batch, y_batch, _ in tqdm(plot_loader, desc=f'testing', leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model_i(x_batch)           # (batch_n, 1)
                y_np = y_pred.cpu().numpy()
                n_alphas = np.row_stack(
                    (n_alphas, y_np)) if n_alphas.size else y_np

                if criterion and test:
                    y_batch = torch.squeeze(y_batch)[:, chebyshev_i]
                    loss = criterion(y_pred, y_batch)
                    test_loss += loss.detach().cpu().item()
                    test_sample += len(x_batch)
        if test:
            print(f"for {chebyshev_i}th order, test loss : {test_loss / test_sample:.10f}, test sample: {test_sample}")
        alphas = np.column_stack(
            (alphas, n_alphas)) if alphas.size else n_alphas
    
    assert alphas.shape == (test_sample, num_models), f'{alphas.shape} != ({len(plot_loader)}, {num_models})'
    return alphas


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
# everytime notice
with open("config_L6_1.json") as f:
    config = json.load(f)
train_model = True
num_models = 1


if __name__ == "__main__":
    # hyper-parameters
    L, N = config["L"], config["N"]
    SIZE = config["SIZE"]
    RATIO = config["RATIO"]
    N_EPOCHS = config["N_EPOCHS"]
    LR = config["LR"]
    batch_size = config['batch_size']

    pre_name = config["pre_name"]
    # num_models 是 N+1, (1, x, 2-x, 3-x, N-x)

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
    validate_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model_skeleton = MyMLP(input_d=input_d, ratio=RATIO)
    criterious = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    if train_model:
        train(train_loader, 
              input_d=input_d, 
              ratio=RATIO, 
              n_epoch=N_EPOCHS,
              criterion=criterious, 
              lr=LR, device=device, 
              num_models=num_models)
    # todo 当前函数没有数据集，还未测试，有没有bug
    model = MyMLP(input_d, RATIO)
    _ = predict_alpha(model=model, 
                      plot_loader=validate_loader,
                      num_models=num_models,
                      device=device, 
                      criterion=criterious, 
                      test=True)
    # plot_spectrum(plot_loader=plot_loader, model=get_models, omegas=omegas, T_pred=T_pred, x_grid=x_grid)
