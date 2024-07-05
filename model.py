import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, L):
    if L == 0:
        return x
    pe = []
    for i in range(L):
        pe.append(torch.sin(2.0**i * x * torch.pi))
        pe.append(torch.cos(2.0**i * x * torch.pi))
    return torch.cat(pe, dim=-1)


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=6, output_ch=4, L=10):
        super(NeRF, self).__init__()
        self.L = L
        self.input_ch = input_ch

        assert L >= 0, "L must be 0 or positive integer"

        if L == 0:
            encoded_ch = input_ch
        else:
            encoded_ch = input_ch * L * 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(encoded_ch, W))
        for i in range(D - 1):
            if i in [4]:
                self.layers.append(nn.Linear(W + encoded_ch, W))
            else:
                self.layers.append(nn.Linear(W, W))
        self.output_layer = nn.Linear(W, output_ch)

    def forward(self, x):
        encoded_x = positional_encoding(x, self.L)
        h = encoded_x
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            if i in [4]:
                h = torch.cat([encoded_x, h], -1)
        return self.output_layer(h)
