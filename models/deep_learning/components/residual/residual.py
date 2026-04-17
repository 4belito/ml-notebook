"""Residual Connection Module.

Tensor dimensions:
    b: batch size
    c: channel/feature dimension
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class ResidualConnection(nn.Module):
    def __init__(self, block: nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.block = block
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Float[Tensor, "b c"]) -> Float[Tensor, "b c"]:
        return self.block(x) + self.skip(x)
