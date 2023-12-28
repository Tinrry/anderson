import torch.nn as nn
from torch import Tensor


class MyMLP_7(nn.Module):
    def __init__(self, input_d, ratio=2) -> None:
        super(MyMLP_7, self).__init__()
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
    

class MyMLP_14_Batchnorm(nn.Module):
    def __init__(self, input_d, layer_num=14, ratio=2) -> None:
        super(MyMLP_14_Batchnorm, self).__init__()
        self.input_d = input_d
        self.ratio = ratio

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(self.input_d, self.input_d * ratio),
            nn.BatchNorm1d(self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.BatchNorm1d(self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio * ratio),
            nn.BatchNorm1d(self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
            nn.BatchNorm1d(self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**3),
            nn.BatchNorm1d(self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            nn.BatchNorm1d(self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            nn.BatchNorm1d(self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            nn.BatchNorm1d(self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**2),
            nn.BatchNorm1d(self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
            nn.BatchNorm1d(self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio),
            nn.BatchNorm1d(self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.BatchNorm1d(self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d),
            nn.BatchNorm1d(self.input_d ),
            nn.ReLU(),
            nn.Linear(self.input_d, 1)
            # maxpool or avgpool
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out


class MyMLP_14(nn.Module):
    def __init__(self, input_d, ratio=2) -> None:
        super(MyMLP_14, self).__init__()
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
        out = self.linear_relu_stack(x)
        return out

# use in MyMLP
class BasicBlock(nn.Module):
    def __init__(self, input_d, output_d) -> None:
        super(BasicBlock, self).__init__()
        self.input_d = input_d
        self.output_d = output_d
        self.linear = nn.Linear(self.input_d, self.output_d)
        self.relu = nn.ReLU()
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.linear(x)
        out = self.relu(x)
        return out
    

# MyMLP layer genelization
class MyMLP(nn.Module):
    def __init__(self, input_d, ratio=2, block: nn.Module=BasicBlock, layer_num=7) -> None:
        super(MyMLP, self).__init__()
        self.input_d = input_d
        self.ratio = ratio

        self.flatten = nn.Flatten()
        self.block = block
        if layer_num == 7:
            self.layer_dim = [(self.input_d, self.input_d * self.ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio * ratio),
                         (self.input_d * ratio**2, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d)
                         ]
        elif layer_num == 14:
            self.layer_dim = [(self.input_d, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio * ratio),
                         (self.input_d * ratio**2, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d)
                         ]
        elif layer_num == 20:
            self.layer_dim = [(self.input_d, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio * ratio),
                         (self.input_d * ratio**2, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**3),
                         (self.input_d * ratio**3, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio**2),
                         (self.input_d * ratio**2, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d * ratio),
                         (self.input_d * ratio, self.input_d)
                         ]
        else:
            raise Exception(f'not implement {layer_num} configure')
        self.nn = self._make_layer(block)
        self.fc = nn.Linear(self.input_d, 1)

    def _make_layer(self, block) -> nn.Sequential:
        layers = []
        for input_d, output_d in self.layer_dim:
            layers.append(block(input_d, output_d))
        return nn.Sequential(*layers)
    
    def forward(self, x) -> Tensor:
        x = self.flatten(x)
        x = self.nn(x)
        out = self.fc(x)
        return out


class MyMLP_20(nn.Module):
    def __init__(self, input_d, ratio=2) -> None:
        super(MyMLP_20, self).__init__()
        self.input_d = input_d
        self.ratio = ratio

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(self.input_d, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
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
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**3),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**3, self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio**2),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio**2, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d * ratio),
            nn.ReLU(),
            nn.Linear(self.input_d * ratio, self.input_d),
            nn.ReLU(),
            nn.Linear(self.input_d, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out
    


class NNBatchnorm(nn.Module):
    def __init__(self, input_d, ratio=2) -> None:
        super(MyMLP_7, self).__init__()
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

import torch

if __name__ == '__main__':
    tensor = torch.rand([10, 8, 1, 1])
    model_auto = MyMLP(input_d=8, ratio=2, block=BasicBlock, layer_num=20)
    print(model_auto)
    model_20 = MyMLP_20(input_d=8, ratio=2)
    # print(model_20)

    total_params =  lambda model: sum(p.numel() for p in model.parameters())
    total_trainable_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model_auto {total_params(model_auto)} : total parameters.')
    print(f'model_20 {total_params(model_20)} : total parameters.')

    print(f'model_auto {total_trainable_params(model_auto)} : total trainable parameters.')
    print(f'model_20 {total_trainable_params(model_20)} : total trainable parameters.')
    
