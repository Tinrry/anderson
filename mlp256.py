from tqdm import tqdm
from tqdm import trange

import torch
import torch.nn as nn

from encoder import ToTensor
from encoder import AndersonChebyshevDataset
from torch.utils.data import DataLoader

    

class MyMLP(nn.Module):
    def __init__(self, input_d, ratio=4) -> None:
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
    

def train(train_loader, input_d, ratio, n_epoch,criterion, lr, device, num_models):
    for chebyshev_i in range(num_models):
        # init model
        model = MyMLP(input_d, ratio=ratio)
        # 第 i 个 model 预测 第i个chebyshev 的系数
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model = model.to(device)

        train_acc = 0.0
        for epoch in trange(n_epoch, desc='Training'):
            train_loss = 0.0
            # TODO dataset load with header 可以分解。
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

            if (epoch+1) % 10 == 0:
            # record loss and accuracy
                train_loss += loss.detach().cpu().item() / len(train_loader)
                print(f" epoch : {epoch+1}/{n_epoch}  train loss: {train_loss:.3f}")

        # save model
        torch.save(model.state_dict(), f'./mlp_models/chebyshev_{chebyshev_i}.pt')

import numpy as np

def predict_alpha(model, plot_loader,  num_models,criterion=None, device=None, test=False):
    models = get_models(pre_name=config["pre_name"], model_skeleton=model, num_models=num_models)
    alphas = np.array([])
    for chebyshev_i in range(num_models):
        model_i = models[chebyshev_i]
        model_i = model_i.to(device)

        n_alphas = np.array([])
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, _ in tqdm(plot_loader, desc=f'testing', leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model_i(x_batch)           # scalar
                n_alphas = np.row_stack((n_alphas, y_pred)) if n_alphas.size else y_pred

                if criterion and test:
                    y_batch = torch.squeeze(y_batch)[chebyshev_i]
                    loss = criterion(y_pred, y_batch)
                    test_loss += loss.detach().cpu().item() / len(plot_loader)
        if test:
            print(f"for {chebyshev_i}th order, test loss : {test_loss:.2f}")
        alphas = np.column_stack((alphas, n_alphas)) if alphas.size else n_alphas
        return alphas

def get_models(pre_name, model_skeleton, num_models):
    models = np.array([])
    for chebyshev_i in range(num_models):
        model_name = f'./mlp_models/{pre_name}_{chebyshev_i}.pt'
        model_i = model_skeleton.load_state_dict(torch.load(model_name))
        models = np.row_stack((models, model_i)) if models.size else model_i
    return models

from encoder import plot_spectrum
import json

with open("config_L6.json") as f:
    config = json.load(f)

if __name__ == "__main__":
    # hyper-parameters
    L, N = config["L"], config["N"]
    SIZE = config["SIZE"]
    RATIO = config["RATIO"]
    N_EPOCHS = config["N_EPOCHS"]
    LR = config["LR"]

    pre_name = config["pre_name"]
    # num_models 是 N+1, (1, x, 2-x, 3-x, N-x)
    num_models = N + 1
    train_model = True

    training_size = int(config["SIZE"] * 0.8)       # training: testing = 8: 2
    training_file = f"L{L}N{N}_training_{training_size}.csv"
    testing_file = f"L{L}N{N}_testing_{SIZE - training_size}.csv"

    input_d = L + 2
    transform = ToTensor()
    # transform = None
    train_set = AndersonChebyshevDataset(csv_file=training_file, L=L, n=N, transform=transform)
    test_set =  AndersonChebyshevDataset(csv_file =testing_file, L=L, n=N, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    train_loader = DataLoader(train_set, shuffle=False, batch_size=128)
    plot_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model_skeleton = MyMLP(input_d=input_d, ratio=RATIO)
    criterious = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    if train_model:
        train(train_loader, input_d=input_d, ratio=RATIO, n_epoch=N_EPOCHS, criterion=criterious, lr=LR, device=device, num_models=num_models)
    # todo 当前函数没有数据集，还未测试，有没有bug
    # nn_alphas = predict_alpha(model_skeleton, plot_loader=plot_loader, criterion=criterious, device=device, num_models=num_models, test=False)
    # plot_spectrum(plot_loader=plot_loader, model=get_models, omegas=omegas, T_pred=T_pred, x_grid=x_grid)
