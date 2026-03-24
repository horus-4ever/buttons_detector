from torch import nn, Tensor
import torch.nn.functional as F
import torch


class FFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_hidden: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        # network layers
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)
        # activation function
        self.activation = _get_activation_fn(activation)

    def forward(self, input):
        result = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return result


class AddNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.norm_layer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, add_target):
        add_target = self.dropout(add_target)
        return self.norm_layer(input + add_target)


import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x



def _get_activation_fn(name: str):
    match name:
        case "relu":
            return F.relu
        case "gelu":
            return F.gelu
        case _:
            return F.relu