"""Residual Connection Module."""

import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, block: nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.block = block
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)
