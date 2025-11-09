"""Batch Normalization Layer Implementation."""

import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    """
    This version is designed for readability and to expose the internal logic of batch normalization (mean/variance computation, normalization, and affine scaling).
    It behaves similarly to `torch.nn.BatchNorm1d` but may differ slightly in
    numerical values due to floating-point accumulation order and lack of low-level
    optimizations.

    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(
                torch.ones(num_features, device=device, dtype=dtype)
            )  # γ (scale)
            self.bias = nn.Parameter(
                torch.zeros(num_features, device=device, dtype=dtype)
            )  # β (shift)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # train/eval mode handling (initialized in the parent class nn.Module)
        if self.training:
            # Step 1: Compute batch statistics
            mean = x.mean(dim=0, keepdim=False)  # (C,)
            var = x.var(dim=0, unbiased=False, keepdim=False)  # (C,)

            with torch.no_grad():
                # Update running statistics using exponential moving average
                n = x.size(0)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                unbias_var = var * n / (n - 1) if n > 1 else var
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * unbias_var
        else:
            # Evaluation mode: use running statistics
            mean = self.running_mean
            var = self.running_var

        # Step 2: Normalize: (x - μ) / √(σ² + ε)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Step 3: Apply affine transformation if enabled
        if self.affine:
            output = self.weight * x_normalized + self.bias
        else:
            output = x_normalized

        return output
