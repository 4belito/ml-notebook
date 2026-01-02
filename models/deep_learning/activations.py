"""Activation Functions implemented in PyTorch."""

import torch
import torch.nn as nn


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.negative_slope * x)


class PReLU(nn.Module):
    def __init__(self, init: float = 0.25):
        super().__init__()
        a = torch.empty(1)
        self.weight = nn.Parameter(a)
        nn.init.constant_(a, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.weight * x)


class ELU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))


class SELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))


class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
