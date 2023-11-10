import os
import copy
import numpy as np
from tqdm import tqdm, trange
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils import AndersonChebyshevDataset
from utils import ToTensor
from utils import plot_spectrum, load_config
from utils import AndersonParas

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
            # nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
            # nn.ReLU(),
            # nn.Linear(self.input_d * ratio**2, self.input_d * ratio**3),
            # nn.ReLU(),
            # nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            # nn.ReLU(),
            # nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            # nn.ReLU(),
            # nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            # nn.ReLU(),
            # nn.Linear(self.input_d * ratio**3, self.input_d * ratio**2),
            # nn.ReLU(),
            # nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
            # nn.ReLU(),
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


from torch.optim.lr_scheduler import StepLR

def train(train_loader, validate_loader, input_d, ratio, n_epoch, criterion, lr, device, num_models=1):
    for chebyshev_i in range(num_models):
        # init model
        model = MyMLP(input_d, ratio=ratio).to(device=device)
        # 第 i 个 model 预测 第i个chebyshev 的系数
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(opt, step_size=15, gamma=0.1)
        model.double()

        for epoch in trange(n_epoch, desc='Training'):
            train_loss = 0.0
            validate_loss = 0.0
            total_sample = 0

            once_batch = True
            for x_batch, y_batch, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                y_batch = torch.squeeze(y_batch, dim=1)[:, chebyshev_i]
                # turn MSE to RMSE
                loss = torch.sqrt(criterion(y_pred, y_batch))
                # backward
                opt.zero_grad()
                loss.backward()
                # update parameters
                opt.step()
                # record loss and accuracy
                train_loss += loss.detach().cpu().item()
                total_sample += len(x_batch)
                if once_batch:
                    # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
                    print(f" y_pred : {y_pred[:10].flatten()}")
                    print(f"y_batch: {y_batch[:10].flatten()}")
                    once_batch = False
            print(f" epoch : {epoch+1}/{n_epoch}  train RMSE loss: {train_loss / len(train_loader):.10f}, train sample: {total_sample}")
            # save every 10 epoch
            if (epoch + 1) % 5 == 0:
                save_pt = f'mlp_models/e{epoch+1}_{chebyshev_i}th.pt'
                torch.save(model.state_dict(), save_pt)

            # validate loop
            validate_sample = 0
            once_batch = True
            for x_v, y_v, _ in tqdm(validate_loader, desc=f'epoch {epoch+1} in validating', leave=False):
                x_v = x_v.to(device)
                y_v = y_v.to(device)
                yv_pred = model(x_v)
                y_v = torch.squeeze(y_v)[:, chebyshev_i]
                loss_v = torch.sqrt(criterion(yv_pred, y_v))

                # TODO plot
                validate_loss += loss_v.detach().cpu().item()
                validate_sample += len(x_v)
                if once_batch:
                   # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
                    print(f" y_pred : {yv_pred[:10].flatten()}")
                    print(f"y_v: {y_v[:10].flatten()}")
                    once_batch = False 
            print(f" epoch : {epoch+1}/{n_epoch}  validate RMSE loss: \
                  {validate_loss / len(validate_loader):.10f}, \
                  validate sample : {validate_sample}")
            scheduler.step()
            print(f'Epoch-{epoch+1} lr: ' + f"{opt.param_groups[0]['lr']}")

        # save model
        torch.save(model.state_dict(),
                   f'./mlp_models/chebyshev_{chebyshev_i}.pt')


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
                loss = torch.sqrt(criterion(y_pred, y_batch))
                test_loss += loss.detach().cpu().item()
                test_sample += len(x_batch)
                print(f'y_pred: {y_pred[:10].flatten()}')
                print(f'y_batch: {y_batch[:10].flatten()}')
                break
            # print(f"for {chebyshev_i}th order, test loss : {test_loss / test_sample:.10f}, test sample: {test_sample}")
            print(f"for {chebyshev_i}th order, test RMSE loss : {test_loss / len(test_loader):.10f}, test sample: {test_sample}")


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
train_model = True
num_models = 1


if __name__ == "__main__":
    # hyper-parameters
    config = load_config('config_L6_2.json')
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
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size)
    validate_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # 改成相对误差平均
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
    test(model, test_loader, num_models, criterion=criterious, device=device)
    # # plot anderson parameters save in paras.csv
    # paras = config['paras']
    # plot_set = AndersonParas(csv_file=paras, L=L)
    # plot_loader = DataLoader(plot_set, shuffle=False, batch_size=32)        # just one batch
    # spectrum_f = config['spectrum_paras']
    # alphas = get_alpha(model, plot_loader=plot_loader, num_models=256, device=device,spectrum_filename=spectrum_f)
    # plot_spectrum(spectrum_filename=spectrum_f, nn_alpha=alphas)
