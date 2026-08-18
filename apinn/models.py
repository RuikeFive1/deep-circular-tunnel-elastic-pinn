
import torch
import torch.nn as nn

class MLP_APINN(nn.Module):
    def __init__(self, in_dim=2, hidden=40, depth=6, out_dim=6, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        layers = []
        last = in_dim
        for _ in range(depth):
            layer = nn.Linear(last, hidden)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers += [layer, nn.Tanh()]
            last = hidden
        out = nn.Linear(last, out_dim)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        self.net = nn.Sequential(*layers, out)
        self.to(dtype=dtype)

    def forward(self, X):
        return self.net(X)
