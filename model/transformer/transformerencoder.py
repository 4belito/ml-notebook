"""
Transformer Encoder Module.
Inspired by PyTorch nn.TransformerEncoderLayer
"""

import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, mbed_dim, num_heades, **mha_kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(mbed_dim, num_heades, **mha_kwargs)

    def forward(self, x, attn_mask=None, key_padding_mask=None, is_causal=False):
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
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")
        to_kwarg = {"device": device, "dtype": dtype}
        self.norm_first = norm_first

        self.block1 = nn.Sequential(
            SelfAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **to_kwarg
            ),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias, **to_kwarg),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias, **to_kwarg),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **to_kwarg)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **to_kwarg)

    def forward(self, x):
        x = self._skkip_block(x, self.norm1, self.block1)
        x = self._skkip_block(x, self.norm2, self.block2)
        return x

    def _skkip_block(self, x, norm, block):
        if self.norm_first:
            return block(norm(x)) + x
        else:
            return norm(block(x) + x)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm):
        super().__init__()
        self.network = nn.Sequential(*[encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, x):
        x = self.network(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
