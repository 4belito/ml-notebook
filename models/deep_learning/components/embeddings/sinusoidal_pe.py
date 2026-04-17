"""Sinusoidal Positional Encoding.

Tensor dimensions:
    n: sequence length
    d: embedding dimension
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn


class SinusoidalPE(nn.Module):
    """
    Transformer-style sinusoidal positional encoding.

    The embedding dimension must be even. This module implements the fixed
    sinusoidal encoding introduced in the original Transformer paper, and can be
    seen as a particular case of FourierPositionalEncoding with geometrically
    spaced frequencies.

    Unlike a general FourierPositionalEncoding, the positional encodings are
    fully precomputed at initialization time and stored as a buffer. This makes
    the implementation more efficient at runtime, as the forward pass only
    performs a slice and addition, avoiding repeated trigonometric computations
    and concatenations.
    """

    def __init__(self, d_model: int, max_len: int = 5000, base_freq: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(base_freq) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe: Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]
