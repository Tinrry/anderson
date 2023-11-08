import os
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from utils import AndersonChebyshevDataset
from utils import load_config, ToTensor


def patchify(images, n_patch, device):
    n, c, h = images.shape
    patch_size = h // n_patch

    patches = torch.zeros(n, n_patch, c * h // n_patch)

    for idx, image in enumerate(images):
        for i in range(n_patch):
            patch = image[:, i * patch_size: (i+1) * patch_size]
            patches[idx, i] = patch.flatten()
    return patches.to(device)


def position_embedding(pos, dim):
    pos_emb = torch.zeros(pos, dim)
    for p in range(pos):
        for i in range(dim):
            pos_emb[p, i] = np.sin(
                p / 10000 ** (2 * i / dim)) if i % 2 == 0 else np.cos(p / 10000 ** (2 * i / dim))
    return pos_emb


class VitBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio):
        super(VitBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        # norm
        self.norm1 = nn.LayerNorm(self.hidden_d)
        # multi-head attention
        self.mha = Mymha(hidden_d, n_heads)
        # residual concatenation
        # norm
        self.norm2 = nn.LayerNorm(self.hidden_d)
        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d*self.mlp_ratio),
            nn.GELU(),
            nn.Linear(self.hidden_d*self.mlp_ratio, self.hidden_d),
        )
        # residual concatenation

    def forward(self, x):
        out = self.mha(self.norm1(x)) + x
        out = self.mlp(self.norm2(out)) + out
        return out


class Mymha(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super(Mymha, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        assert self.hidden_d % self.n_heads == 0, "hidden_d can not equal split to n_heads"

        self.head_d = self.hidden_d // self.n_heads
        self.q_mapping = nn.ModuleList(
            [nn.Linear(self.head_d, self.head_d) for _ in range(self.n_heads)])
        self.k_mapping = nn.ModuleList(
            [nn.Linear(self.head_d, self.head_d) for _ in range(self.n_heads)])
        self.v_mapping = nn.ModuleList(
            [nn.Linear(self.head_d, self.head_d) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # sequences dimension is (N, pos, hidden_d)
        # split to (N, pos, n_heads, hidden_d//n_heads)
        # and result return to (N, pos, hidden_d) format
        results = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mapping[head]
                k_mapping = self.k_mapping[head]
                v_mapping = self.v_mapping[head]

                seq = sequence[:, head*self.head_d: (head+1)*self.head_d]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax((q @ k.T)/(self.head_d ** 0.5))
                seq_result.append(attention @ v)
            seq_result = torch.hstack(seq_result)
            results.append(seq_result)
        out = torch.cat([torch.unsqueeze(r, dim=0) for r in results])
        return out


class MyViT(nn.Module):
    def __init__(self, ch, n_patch, blocks, n_heads, hidden_d, out_d, mlp_ratio=4):
        super(MyViT, self).__init__()

        # attribute
        self.ch = ch
        self.n_patch = n_patch
        self.blocks = blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.out_d = out_d
        self.mlp_ratio = mlp_ratio

        assert self.ch[1] % n_patch == 0, "error, images must divide n_patch."
        patch_size = self.ch[1] // self.n_patch

        # Linear projection
        self.input_d = int(self.ch[0] * patch_size)
        self.linear = nn.Linear(in_features=self.input_d,
                                out_features=self.hidden_d)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        self.register_buffer('position_embedding', position_embedding(
            1+self.n_patch, self.hidden_d), persistent=True)
        self.blocks = nn.ModuleList(
            [VitBlock(self.hidden_d, self.n_heads, self.mlp_ratio) for _ in range(self.blocks)])
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images, device):
        n, _, _ = images.shape
        # patchify
        patches = patchify(images, self.n_patch, device)
        tokens = self.linear(patches)

        # add class token
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # position embedding
        out = tokens + self.position_embedding.repeat(n, 1, 1)
        for block in self.blocks:
            out = block(out)
        out = out[:, 0]
        out = self.mlp(out)
        return out


def train(model, train_loader, n_epoch, criterion, LR, device):

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in trange(n_epoch, desc='Training'):
        train_loss = 0.0
        for x_batch, y_batch, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
            y_batch = torch.squeeze(y_batch)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch, device)
            loss = criterion(y_pred, y_batch)
            # backward
            opt.zero_grad()
            loss.backward()
            # update parameters
            opt.step()

            train_loss += loss.detach().cpu().item() / len(train_loader)
        if (epoch+1) % 10 == 0:
            # record loss and accuracy
            print(
                f" epoch : {epoch+1}/{n_epoch}  train loss: {train_loss:.10f}")


def get_models(pre_name, model_skelton, num_models=1):
    model = model_skelton.load_state_dict(torch.load(pre_name))
    return model


def predict_alpha(model, plot_loader, criterion=None, device=None, num_models=1, test=True):
    model = get_models(pre_name=config["pre_name"],model_skelton=model, num_models=num_models)
    if device:
        model = model.to(device)

    test_loss = 0.0
    alphas = np.array([])
    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm(plot_loader, desc=f'testing', leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch, device)
            alphas = y_pred

            if criterion:
                y_batch = torch.squeeze(y_batch)
                loss = criterion(y_pred, y_batch)
                test_loss += loss.detach().cpu().item() / len(plot_loader)

    if test and criterion:
        print(f"test loss : {test_loss:.10f}")
    return alphas




np.random.seed(0)
torch.manual_seed(0)


if __name__ == "__main__":
    train_model = False
    config = load_config('config_L6.json')
    L, N = config["L"], config["N"]
    N_EPOCHS = config["N_EPOCHS"]
    SIZE = config["SIZE"]

    # plot parameters
    nrows=config['nrows']
    ncols=4['ncols']

    training_size = int(config["SIZE"] * 0.8)       # training: testing = 8: 2
    training_file = os.path.join('datasets', f"L{L}N{N}_training_{training_size}.csv")
    testing_file = os.path.join('datasets', f"L{L}N{N}_testing_{SIZE - training_size}.csv")

    input_d = L + 2
    transform = ToTensor()
    # transform = None
    train_set = AndersonChebyshevDataset(
        csv_file=training_file, L=L, n=N, transform=transform)
    test_set = AndersonChebyshevDataset(
        csv_file=testing_file, L=L, n=N, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, shuffle=False,
                             batch_size=config["batch_size"])

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = MyViT((1, input_d), 
                  n_patch=input_d,
                  blocks=config['blocks'], 
                  n_heads=config['n_heads'], 
                  hidden_d=config['hidden_d'], 
                  out_d=N+1).to(device)
    model.double()

    criterious = nn.MSELoss()
    if train_model:
        train(model, train_loader, n_epoch=N_EPOCHS, criterion=criterious, LR=0.005, device=device)
        # save model
        torch.save(model.state_dict(), config["model_name"])

    # load model
    model.load_state_dict(torch.load(config["model_name"]))
    # validate(model, test_loader, criterion=criterious, device=device)

    # in default plot batch = nrows * nclos
    plot_loader = DataLoader(test_set, shuffle=False, batch_size=nrows * ncols)
    plot_spectrum(plot_loader, 
                  model=model, 
                  omegas_csv=config["spectrum_paras"], 
                  nrows=nrows, 
                  ncols=ncols, 
                  model_flag = "transformer")
