"""Dropout Module Implementation"""

import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        """
        p: drop probability (0 <= p < 1)
        """
        super().__init__()
        assert 0.0 <= p < 1.0, "p must be in [0, 1)"
        self.p = p
        self.keep_prob = 1.0 - self.p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If we are in eval mode, dropout does nothing
        if not self.training or self.p == 0.0:
            return x
        ## this match PyTorch's nn.Dropout random behavior
        mask = torch.empty_like(x, dtype=torch.float32)
        mask = mask.bernoulli_(self.keep_prob)
        return x * mask / self.keep_prob
