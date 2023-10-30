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
    

def train(train_loader, n_epoch,criterion, LR, device, num_models):
    for chebyshev_i in range(num_models):
        # init model
        model = MyMLP(input_d=14, ratio=2)
        # 第 i 个 model 预测 第i个chebyshev 的系数
        opt = torch.optim.Adam(model.parameters(), lr=LR)
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


def test(test_loader, criterion, device, num_models):
    for chebyshev_i in range(num_models):
        model = torch.load_state_dict(torch.load(f'chebyshev_{chebyshev_i}.pt'))
        model = model.to(device)

        test_loss = 0.0
        correct = 0
        samples = 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(test_loader, desc=f'testing'):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                y_batch = torch.squeeze(y_batch)[chebyshev_i]
                loss = criterion(y_pred, y_batch)
                test_loss += loss.detach().cpu().item() / len(test_loader)
                correct += torch.sum(torch.argmax(y_pred, dim=1) == y_batch).detach().cpu().item()
                samples += len(y_batch)
        print(f"for {chebyshev_i}th order, test loss : {test_loss:.2f} test accuracy: {correct/samples * 100:.2f}%")

# 

if __name__ == "__main__":
    L, N = 3, 25
    N_EPOCHS=2

    input_d = 2 * L * 2 + 2
    transform = ToTensor()
    # transform = None
    train_set = AndersonChebyshevDataset(L=L, n=N, transform=transform)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    criterious = nn.MSELoss()
    train(train_loader, n_epoch=N_EPOCHS, criterion=criterious, LR=0.005, device=device, num_models=N+1)
    # test(test_loader=)