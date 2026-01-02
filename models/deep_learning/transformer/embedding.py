"""Custom Embedding Layer Implementation."""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input_indices: torch.Tensor):
        return self.weight[input_indices]
