"""
Transformer Encoder Module.
Inspired by PyTorch nn.TransformerEncoderLayer

Tensor Dimension Conventions:
    b: batch size
    n: sequence length
    c: embedding dimension / channels (d_model)
"""

import copy

import torch
from jaxtyping import Float
from torch import Tensor, nn

from models.deep_learning.components import SelfAttention


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation_cls: type[nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.attn_block = nn.Sequential(self.self_attn, nn.Dropout(dropout))

        self.linear1 = nn.Linear(
            d_model, dim_feedforward, bias=bias, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, bias=bias, device=device, dtype=dtype
        )
        self.ff_block = nn.Sequential(
            self.linear1,
            activation_cls(),
            nn.Dropout(dropout),
            self.linear2,
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.ff_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )

    def load_weights_from_torch_encoder_layer(self, src: nn.TransformerEncoderLayer):
        """Load weights from an nn.TransformerEncoderLayer."""
        self.self_attn.load_weights_from_torch_mha(src.self_attn)
        with torch.no_grad():
            self.linear1.weight.copy_(src.linear1.weight)
            self.linear1.bias.copy_(src.linear1.bias)
            self.linear2.weight.copy_(src.linear2.weight)
            self.linear2.bias.copy_(src.linear2.bias)
            self.attn_norm.weight.copy_(src.norm1.weight)
            self.attn_norm.bias.copy_(src.norm1.bias)
            self.ff_norm.weight.copy_(src.norm2.weight)
            self.ff_norm.bias.copy_(src.norm2.bias)

    def forward(self, x: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        x = self._skkip_block(x, self.attn_norm, self.attn_block)
        x = self._skkip_block(x, self.ff_norm, self.ff_block)
        return x

    def _skkip_block(
        self, x: Float[Tensor, "b n c"], norm: nn.LayerNorm, block: nn.Sequential
    ) -> Float[Tensor, "b n c"]:
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
        self.network = nn.Sequential(
            *[copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, x: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        x = self.network(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
