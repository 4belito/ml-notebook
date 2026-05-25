"""
Transformer Decoder Module.
Inspired by PyTorch nn.TransformerDecoderLayer

Tensor Dimension Conventions:
    b: batch size
    n: sequence length
    c: embedding dimension / channels (d_model)
"""

import copy

import torch
from jaxtyping import Float
from torch import Tensor, nn

from models.deep_learning.components import MultiheadAttention, SelfAttention


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer."""

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

        # Block 1: Self-Attention
        self.selfattn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.selfattn_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.selfattn_dropout = nn.Dropout(dropout)

        # Block 2: Multi-Head Attention
        self.multiheadattn = MultiheadAttention.from_torch_mha(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.multiheadattn_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.multiheadattn_dropout = nn.Dropout(dropout)

        # Block 3: Feedforward Network
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, bias=bias, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, bias=bias, device=device, dtype=dtype
        )
        self.ffn_block = nn.Sequential(
            self.linear1,
            activation_cls(),
            nn.Dropout(dropout),
            self.linear2,
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )

    def _forward_selfattn(
        self,
        x: Float[Tensor, "b n c"],
        attn_mask: Float[Tensor, "n n"] | None = None,
    ) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            y = self.selfattn_norm(x)
            y = self.selfattn(y, attn_mask=attn_mask)
            y = self.selfattn_dropout(y)
            return y + x
        else:
            y = self.selfattn(x, attn_mask=attn_mask)
            y = self.selfattn_dropout(y)
        return self.selfattn_norm(y + x)

    def _forward_multiheadattn(
        self, x: Float[Tensor, "b n c"], memory: Float[Tensor, "b m c"]
    ) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            y = self.multiheadattn_norm(x)
            y = self.multiheadattn(y, memory, memory)
            y = self.multiheadattn_dropout(y)
            return y + x
        else:
            y = self.multiheadattn(x, memory, memory)
            y = self.multiheadattn_dropout(y)
        return self.multiheadattn_norm(y + x)

    def _forward_ffn(self, x: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            return self.ffn_block(self.ffn_norm(x)) + x
        return self.ffn_norm(self.ffn_block(x) + x)

    def forward(
        self,
        x: Float[Tensor, "b n c"],
        memory: Float[Tensor, "b m c"],
        tgt_mask: Float[Tensor, "n n"] | None = None,
    ) -> Float[Tensor, "b n c"]:
        x = self._forward_selfattn(x, attn_mask=tgt_mask)
        x = self._forward_multiheadattn(x, memory)
        x = self._forward_ffn(x)
        return x

    def load_weights_from_torch_decoder_layer(self, src: nn.TransformerDecoderLayer):
        """Load weights from an nn.TransformerDecoderLayer."""
        self.selfattn.load_weights_from_torch_mha(src.self_attn)
        self.multiheadattn.load_weights_from_torch_mha(src.multihead_attn)
        with torch.no_grad():
            self.linear1.weight.copy_(src.linear1.weight)
            self.linear1.bias.copy_(src.linear1.bias)
            self.linear2.weight.copy_(src.linear2.weight)
            self.linear2.bias.copy_(src.linear2.bias)
            self.selfattn_norm.weight.copy_(src.norm1.weight)
            self.selfattn_norm.bias.copy_(src.norm1.bias)
            self.multiheadattn_norm.weight.copy_(src.norm2.weight)
            self.multiheadattn_norm.bias.copy_(src.norm2.bias)
            self.ffn_norm.weight.copy_(src.norm3.weight)
            self.ffn_norm.bias.copy_(src.norm3.bias)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.LayerNorm | None,
    ):
        super().__init__()
        self.network = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(
        self,
        x: Float[Tensor, "b n c"],
        memory: Float[Tensor, "b m c"],
        tgt_mask: Float[Tensor, "n n"] | None = None,
    ) -> Float[Tensor, "b n c"]:
        for layer in self.network:
            x = layer(x, memory, tgt_mask=tgt_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
