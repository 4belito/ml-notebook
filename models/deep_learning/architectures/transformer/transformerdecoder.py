"""
Transformer Encoder Module.
Inspired by PyTorch nn.TransformerEncoderLayer

Tensor Dimension Conventions:
    b: batch size
    n: sequence length
    c: embedding dimension / channels (d_model)
"""

import torch
from jaxtyping import Float
from torch import Tensor, nn

from models.deep_learning.components import MultiheadAttention, SelfAttention


class TransformerDecoderLayer(nn.Module):
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
    ) -> None:
        self.norm_first = norm_first
        self.self_attn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        super().__init__()
        self.multihead_attn = MultiheadAttention.from_torch_mha(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        # Implementation of Feedforward model
        self.ff_block = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype),
            activation_cls(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )

        self.norm_first = norm_first
        # pyrefly: ignore [bad-argument-type]
        self.selfattn_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, dtype=dtype, device=device
        )
        # pyrefly: ignore [bad-argument-type]
        self.multiattn_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, dtype=dtype, device=device
        )
        # pyrefly: ignore [bad-argument-type]
        self.ff_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, dtype=dtype, device=device
        )

        self.selfattn_dropout = nn.Dropout(dropout)
        self.multiattn_dropout = nn.Dropout(dropout)
        self.activation = activation_cls()

    def forward(self, x: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:

        return x

    def _selfattn_block(
        self,
        x: Float[Tensor, "b n c"],
        key_padding_mask: Tensor | None = None,
    ) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            x = self.selfattn_norm(x)
            x_selfattn = self.self_attn(
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=True,
            )[0]
            x = self.selfattn_dropout(x_selfattn) + x
        if not self.norm_first:
            x_selfattn = self.self_attn(
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=True,
            )[0]
            x = self.selfattn_dropout(x_selfattn) + x
            x = self.selfattn_norm(x)
        return x

    def multiattn_block(
        self,
        x: Float[Tensor, "b n c"],
        memory: Float[Tensor, "b m c"],
        key_padding_mask: Tensor | None = None,
    ) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            x = self.multiattn_norm(x)
            x_multiattn = self.multihead_attn(
                x,
                memory,
                memory,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
            x = self.multiattn_dropout(x_multiattn) + x
        if not self.norm_first:
            x_multiattn = self.multihead_attn(
                x,
                memory,
                memory,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
            x = self.multiattn_dropout(x_multiattn) + x
            x = self.multiattn_norm(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.LayerNorm | None,
    ):
        super().__init__()
        self.network = nn.Sequential(*[decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, x: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        x = self.network(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
