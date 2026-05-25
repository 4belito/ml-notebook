"""
Educational Transformer implementation.

Goal: a readable, explicit implementation of the Transformer architecture that
can be verified step-by-step against PyTorch's nn.Transformer. Correctness is
checked by loading weights from an nn.Transformer and comparing eval outputs.
Simplicity is preferred over performance or production flexibility.

Public API:
    TransformerEncoderLayer, TransformerEncoder  — encoder stack
    TransformerDecoderLayer, TransformerDecoder  — decoder stack
    Transformer                                  — full encoder-decoder model
"""

from .decoder import TransformerDecoder, TransformerDecoderLayer
from .encoder import TransformerEncoder, TransformerEncoderLayer
from .transformer import Transformer

__all__ = [
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "Transformer",
]
