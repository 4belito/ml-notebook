"""
This method initializes the InputEmbeddings class.
original code taken from:
ttps://medium.com/@sayedebad.777/building-a-transformer-from-scratch-a-step-by-step-guide-a3df0aeb7c9a
"""

import math

from torch import Tensor, nn


class InputEmbeddings(nn.Module):
    """Convert the original sentence into vectors of d_model dimensions"""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
