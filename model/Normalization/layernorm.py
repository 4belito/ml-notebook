"""Layer Normalization module."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    This version is designed for readability and to expose the internal logic of
    layer normalization.
    It behaves similarly to `torch.nn.LayerNorm` but may differ slightly in
    numerical values due to floating-point accumulation order and lack of low-level
    optimizations.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.normalized_dim = len(self.normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # Compute mean and variance over last normalized_dims
        dim = tuple(range(x.dim() - self.normalized_dim, x.dim()))
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm
