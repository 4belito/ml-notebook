"""
Educational Transformer Decoder: TransformerDecoderLayer + TransformerDecoder.

Goal: an explicit, readable decoder that exposes every sub-block as a named
attribute and makes the data flow easy to follow. Weights can be copied from
nn.TransformerDecoderLayer / nn.TransformerDecoder to verify correctness —
eval outputs match exactly; deviations from nn are documented below.

Similarities with nn.TransformerDecoderLayer:
    - Same three sub-blocks (masked self-attn + cross-attn + FFN) in the same order.
    - Same pre-norm / post-norm behaviour controlled by norm_first.
    - Dropout applied at identical positions after each sub-block output.

Discrepancies vs nn.TransformerDecoderLayer:
    - Always batch-first (b, n, c); nn default is seq-first (n, b, c).
    - activation_cls takes a class (nn.GELU); nn takes a string ("gelu").
    - Both attention blocks use separate q_proj/k_proj/v_proj; nn packs them
      into a single in_proj_weight per block. The packed single-GEMM backward
      differs in floating-point order from three separate GEMMs, producing
      ~1e-8 gradient differences that amplify to ~1e-3 after multi-layer
      propagation through updated weights.
    - Attention dropout goes through an explicit nn.Dropout. On CPU/MPS training
      outputs match because F.scaled_dot_product_attention uses the same global
      RNG (math backend). On CUDA with flash attention the Philox RNG diverges.
    - key_padding_mask is not passed per sub-block; it is merged into the
      attention mask upstream (Transformer._add_masks). Float (0/-inf) masks
      work correctly via addition; boolean masks are not supported.
    - is_causal is not supported.

Tensor dimension conventions:
    b: batch size
    m: target sequence length
    n: source (memory) sequence length
    c: embedding dimension (d_model)
    h: number of attention heads
"""

import copy

import torch
from jaxtyping import Float
from torch import Tensor, nn

from models.deep_learning.components import MultiheadAttention, SelfAttention


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer: masked self-attention + cross-attention + FFN.

    Supports both post-norm (norm_first=False, default) and pre-norm
    (norm_first=True) variants.

    Args:
        d_model: embedding dimension.
        nhead: number of attention heads.
        dim_feedforward: inner dimension of the FFN.
        dropout: dropout probability (applied after each sub-block).
        activation_cls: activation class for the FFN (e.g. nn.ReLU, nn.GELU).
        layer_norm_eps: epsilon for LayerNorm layers.
        norm_first: if True, apply LayerNorm before each sub-block (pre-norm).
        bias: whether linear layers include a bias term.
        device: target device.
        dtype: target dtype.
    """

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
        self.crossattn = MultiheadAttention.from_torch_mha(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.crossattn_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.crossattn_dropout = nn.Dropout(dropout)

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
        mask: Float[Tensor, "n n"] | None = None,
    ) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            y = self.selfattn_norm(x)
            y = self.selfattn(y, attn_mask=mask)[0]
            y = self.selfattn_dropout(y)
            return y + x
        else:
            y = self.selfattn(x, attn_mask=mask)[0]
            y = self.selfattn_dropout(y)
        return self.selfattn_norm(y + x)

    def _forward_crossattn(
        self,
        x: Float[Tensor, "b n c"],
        memory: Float[Tensor, "b m c"],
        mask: Float[Tensor, "n m"] | None = None,
    ) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            y = self.crossattn_norm(x)
            y = self.crossattn(y, memory, memory, attn_mask=mask)[0]
            y = self.crossattn_dropout(y)
            return y + x
        else:
            y = self.crossattn(x, memory, memory, attn_mask=mask)[0]
            y = self.crossattn_dropout(y)
        return self.crossattn_norm(y + x)

    def _forward_ffn(self, x: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        if self.norm_first:
            return self.ffn_block(self.ffn_norm(x)) + x
        return self.ffn_norm(self.ffn_block(x) + x)

    def forward(
        self,
        x: Float[Tensor, "b n c"],
        memory: Float[Tensor, "b m c"],
        memory_mask: Float[Tensor, "n m"] | None = None,
        tgt_mask: Float[Tensor, "n n"] | None = None,
    ) -> Float[Tensor, "b n c"]:
        """
        Args:
            x: target sequence (b, m, d_model).
            memory: encoder output (b, n, d_model).
            tgt_mask: additive mask for self-attention over x (m, m).
            memory_mask: additive mask for cross-attention (m, n).

        Returns:
            Decoded sequence (b, m, d_model).
        """
        x = self._forward_selfattn(x, mask=tgt_mask)
        x = self._forward_crossattn(x, memory, mask=memory_mask)
        x = self._forward_ffn(x)
        return x

    def load_weights_from_torch_decoder_layer(self, src: nn.TransformerDecoderLayer):
        """Load weights from an nn.TransformerDecoderLayer."""
        self.selfattn.load_weights_from_torch_mha(src.self_attn)
        self.crossattn.load_weights_from_torch_mha(src.multihead_attn)
        with torch.no_grad():
            self.linear1.weight.copy_(src.linear1.weight)
            self.linear1.bias.copy_(src.linear1.bias)
            self.linear2.weight.copy_(src.linear2.weight)
            self.linear2.bias.copy_(src.linear2.bias)
            self.selfattn_norm.weight.copy_(src.norm1.weight)
            self.selfattn_norm.bias.copy_(src.norm1.bias)
            self.crossattn_norm.weight.copy_(src.norm2.weight)
            self.crossattn_norm.bias.copy_(src.norm2.bias)
            self.ffn_norm.weight.copy_(src.norm3.weight)
            self.ffn_norm.bias.copy_(src.norm3.bias)


class TransformerDecoder(nn.Module):
    """Stack of N TransformerDecoderLayer instances with an optional final LayerNorm.

    Args:
        decoder_layer: layer template (deep-copied N times).
        num_layers: number of layers.
        norm: optional LayerNorm applied after the last layer (required to
            match nn.Transformer, which always adds one).
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.LayerNorm | None = None,
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
        memory_mask: Float[Tensor, "n m"] | None = None,
    ) -> Float[Tensor, "b n c"]:
        for layer in self.network:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def load_weights_from_torch_decoder(self, src: nn.TransformerDecoder):
        """Load weights from an nn.TransformerDecoder."""
        for layer, src_layer in zip(self.network, src.layers, strict=True):
            layer.load_weights_from_torch_decoder_layer(src_layer)  # type: ignore
        if self.norm is not None and isinstance(src.norm, nn.LayerNorm):
            with torch.no_grad():
                self.norm.weight.copy_(src.norm.weight)
                self.norm.bias.copy_(src.norm.bias)
