"""Implementation of a linear (fully connected) layer."""

import math
import torch
from torch import Tensor, nn
from typing import Literal


class Linear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_activation: Literal["relu", "leaky_relu", "tanh", "sigmoid"] | None = None,
        init_a: float = 0.0,
    ):
        super().__init__()  # type: ignore
        W = torch.empty(output_dim, input_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(W)
        match init_activation:
            case None:
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            case "relu":
                nn.init.kaiming_uniform_(self.weight, a=0, nonlinearity="relu")
            case "leaky_relu":
                nn.init.kaiming_uniform_(self.weight, a=init_a, nonlinearity="leaky_relu")
            case "tanh":
                gain = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(self.weight, gain=gain)
            case "sigmoid":
                gain = nn.init.calculate_gain("sigmoid")
                nn.init.xavier_uniform_(self.weight, gain=gain)

        if bias:
            b = torch.empty(output_dim, device=device, dtype=dtype)
            self.bias = nn.Parameter(b)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # type: ignore
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias:
            out += self.bias
        return out
