"""
Transformer Encoder Module.
Inspired by PyTorch nn.TransformerEncoderLayer
"""

import torch

from torch import Tensor, nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        mbed_dim: int,
        num_heades: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            mbed_dim,
            num_heades,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ):
        return self.mha(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]


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
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)

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
        self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm: nn.LayerNorm | None
    ):
        super().__init__()
        self.network = nn.Sequential(*[encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, x: Tensor) -> Tensor:
        x = self.network(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
