"""Custom Embedding Layer Implementation.

Tensor dimensions:
    b: batch size
    n: sequence length
    d: embedding dimension
"""

import torch
from jaxtyping import Float
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input_indices: Tensor) -> Float[Tensor, "... d"]:
        return self.weight[input_indices]
