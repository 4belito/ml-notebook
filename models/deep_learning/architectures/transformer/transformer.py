"""
Educational full Transformer (encoder + decoder).

Goal: a simple, explicit implementation of the full Transformer that is easy
to read and trace. Correctness is verified by loading weights from an
nn.Transformer and comparing eval outputs — they match exactly.
Simplicity is preferred over production features (no is_causal, no nested
tensors, boolean padding masks not supported).

Architecture:
    src → TransformerEncoder (N layers + LayerNorm) → memory
    tgt + memory → TransformerDecoder (M layers + LayerNorm) → output

Similarities with nn.Transformer:
    - Same encoder-decoder structure with a final LayerNorm after each stack.
    - Same mask parameter names and shapes.
    - Eval outputs match exactly after load_weights_from_torch_transformer.

Discrepancies vs nn.Transformer:
    - activation_cls takes a class (nn.GELU); nn takes a string ("gelu").
    - is_causal / src_is_causal / tgt_is_causal are not supported.
    - Mask handling (see _add_masks): nn passes attention_mask and
      key_padding_mask separately, converting boolean padding masks to -inf
      internally. Here they are merged by addition before the forward call.
      Float (0/-inf) key_padding_masks work correctly via addition.
      Boolean key_padding_masks are NOT supported (True would add 1, not -inf).
    - Training outputs match on CPU/MPS; may diverge on CUDA with flash
      attention (see encoder/decoder module docstrings).
"""

import torch
import torch.nn as nn

import models.deep_learning.architectures.transformer as T


class Transformer(nn.Module):
    """Encoder-decoder Transformer equivalent to nn.Transformer (batch-first).

    Weights can be loaded directly from an nn.Transformer instance via
    load_weights_from_torch_transformer.

    Args:
        d_model: embedding dimension.
        nhead: number of attention heads.
        num_encoder_layers: number of encoder layers.
        num_decoder_layers: number of decoder layers.
        dim_feedforward: inner dimension of the FFN.
        dropout: dropout probability.
        activation_cls: activation class (e.g. nn.ReLU, nn.GELU).
        layer_norm_eps: epsilon for all LayerNorm layers.
        norm_first: if True, use pre-norm (norm before attention/FFN).
        bias: whether linear layers include a bias term.
        device: target device.
        dtype: target dtype.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
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
        self.encoder_layer = T.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_cls=activation_cls,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.decoder_layer = T.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_cls=activation_cls,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        encoder_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        decoder_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.encoder = T.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm
        )
        self.decoder = T.TransformerDecoder(
            self.decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm
        )

    def load_weights_from_torch_transformer(self, src: nn.Transformer):
        """Copy all weights (layers + final norms) from an nn.Transformer."""
        self.encoder.load_weights_from_torch_encoder(src.encoder)
        self.decoder.load_weights_from_torch_decoder(src.decoder)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: source sequence (b, n, d_model).
            tgt: target sequence (b, m, d_model).
            src_mask: additive mask for encoder self-attention (n, n).
            tgt_mask: additive mask for decoder self-attention (m, m).
            memory_mask: additive mask for decoder cross-attention (m, n).

        Returns:
            output tensor (b, m, d_model).
        """
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return output
