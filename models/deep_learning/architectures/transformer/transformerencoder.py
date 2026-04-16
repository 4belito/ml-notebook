"""
Transformer Encoder Module.
Inspired by PyTorch nn.TransformerEncoderLayer
"""

import torch
from torch import Tensor, nn

from models.deep_learning.components import SelfAttention


# B: batch size
# H: number of heads
# S: sequence length
# D: head dimension
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")
        self.norm_first = norm_first
        selfatt = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.block1 = nn.Sequential(selfatt, nn.Dropout(dropout))

        self.block2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )

    def forward(self, x: Tensor):
        x = self._skkip_block(x, self.norm1, self.block1)
        x = self._skkip_block(x, self.norm2, self.block2)
        return x

    def _skkip_block(self, x: Tensor, norm: nn.LayerNorm, block: nn.Sequential):
        if self.norm_first:
            return block(norm(x)) + x
        else:
            return norm(block(x) + x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.LayerNorm | None,
    ):
        super().__init__()
        self.network = nn.Sequential(*[encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, x: Tensor) -> Tensor:
        x = self.network(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
