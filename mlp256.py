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
            for x_batch, y_batch in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                y_batch = torch.squeeze(y_batch)[chebyshev_i]
                loss = criterion(y_pred, y_batch)
                # backward
                opt.zero_grad()
                loss.backward()
                # update parameters
                opt.step()

            if (epoch+1) % 10 == 0:
            # record loss and accuracy
                train_loss += loss.detach().cpu().item() / len(train_loader)
                train_acc += (y_pred.max(1)[1] == y_batch).sum().item() / len(train_loader)
                print(f" epoch : {epoch+1}/{n_epoch}  train loss: {train_loss:.3f} train accuracy: {train_acc * 100:.3f}%")

        # save model
        torch.save(model.state_dict(), f'chebyshev_{chebyshev_i}.pt')

import numpy as np

def compose_chebyshev_alpha(plot_loader, criterion, device, num_models, test=False):
    models = get_models(pre_name='chebyshev', num_models=num_models)
    orders = np.array([])
    for chebyshev_i in range(num_models):
        model_i = models[chebyshev_i]
        model_i = model_i.to(device)

        nn_alphas = np.array([])
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(test_loader, desc=f'testing'):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model_i(x_batch)           # scalar
                nn_alphas = np.row_stack((nn_alphas, y_pred)) if nn_alphas.size else y_pred

                y_batch = torch.squeeze(y_batch)[chebyshev_i]
                loss = criterion(y_pred, y_batch)
                test_loss += loss.detach().cpu().item() / len(test_loader)
        if test:
            print(f"for {chebyshev_i}th order, test loss : {test_loss:.2f}")
        orders = np.column_stack((orders, nn_alphas)) if orders.size else nn_alphas
        return orders

def get_models(pre_name, num_models):
    models = np.array([])
    for chebyshev_i in range(num_models):
        model_name = f'{pre_name}_{chebyshev_i}.pt'
        model_i = torch.load_state_dict(torch.load(model_name))
        model_i = model_i.to(device)
        models = np.row_stack((models, model_i)) if models.size else model_i
    return models

from encoder import plot_spectrum


if __name__ == "__main__":
    # hyper-parameters
    L, N, SIZE = 6, 255 ,5000
    N_EPOCHS=10
    LR=0.000005
    RATIO = 2
    # num_models 是 N+1, (1, x, 2-x, 3-x, N-x)
    num_models = N + 1
    train_model = False

    training_size = int(SIZE * 0.8)       # training: testing = 8: 2
    training_file = f"L{L}N{N}_training{training_size}.csv"
    testing_file = f"L{L}N{N}_training{SIZE - training_size}.csv"

    input_d = L + 2
    transform = ToTensor()
    # transform = None
    train_set = AndersonChebyshevDataset(csv_file=training_file, L=L, n=N, transform=transform)
    test_set =  AndersonChebyshevDataset(csv_file =testing_file, L=L, n=N, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    criterious = nn.MSELoss()
    # every time we save model in train function, and load model in compose_chebyshev_alpha, 256 models
    # loss is too small
    if train_model:
        train(train_loader, input_d=input_d, ratio=RATIO, n_epoch=N_EPOCHS, criterion=criterious, lr=LR, device=device, num_models=num_models)
    # todo 当前函数没有数据集，还未测试，有没有bug
    nn_alphas = compose_chebyshev_alpha(plot_loader=test_loader, criterion=criterious, device=device, num_models=num_models, test=False)
    plot_spectrum(plot_loader=test_loader, models=get_models, omegas=omegas, T_pred=T_pred, x_grid=x_grid)
