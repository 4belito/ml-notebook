import math

import torch
from jaxtyping import Float
from torch import Tensor, nn

# B: batch size
# H: number of heads
# S: sequence length
# D: head dimension


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000, cache_max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.nax_len = cache_max_seq_len
        self.rope_mattices = rope_diagonal_stacked(
            dim, theta, self.nax_len, dtype=torch.float32
        )

    def rotate_queries_or_keys(
        self, x: Float[Tensor, "B H S D"]
    ) -> Float[Tensor, "B H S D"]:
        rope = self.rope_mattices[: x.shape[-2]]  # shape (S, D//2, 2, 2)
        x = x.reshape(*x.shape[:-1], -1, 1, 2)  # reshape to (B, H, S, D//2, 1, 2)
        x = rope[..., 0] * x[..., 0] + rope[..., 1] * x[..., 1]  # apply rotation
        x = x.reshape(*x.shape[:-2], -1)  # reshape back to (B, H, S, D)
        return x

    def forward(self, x: Float[Tensor, "B H S D"]) -> Float[Tensor, "B H S D"]:
        return self.rotate_queries_or_keys(x)


def trot2d_mat(
    alpha: float, dtype: torch.dtype = torch.float32
) -> Float[Tensor, "2 2"]:
    """Returns a 2D rotation matrix for a given angle alpha.
    output shape: (2, 2).
    """
    return torch.tensor(
        [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]],
        dtype=dtype,
    )


def rope_diagonal(
    dim: int, theta: float, pos: int, dtype: torch.dtype = torch.float32
) -> Float[Tensor, "dim/2 2 2"]:
    """Returns the diagonal of the ROPE rotation matrix for a given position.
    output shape: (dim/2,2,2).
    """
    alphas = [pos / (theta ** (2 * k / dim)) for k in range(dim // 2)]
    return torch.stack([trot2d_mat(a, dtype=dtype) for a in alphas], dim=0)


def rope_diagonal_stacked(
    dim: int, theta: float, seq_len: int, dtype: torch.dtype = torch.float32
) -> Float[Tensor, "seq_len dim/2 2 2"]:
    """Returns the stacked diagonals of the ROPE rotation matrices for all positions in
    the sequence.
    output shape: (seq_len, dim/2, 2, 2).
    """
    posed_rope_diags = [
        rope_diagonal(dim, theta, pos, dtype=dtype) for pos in range(seq_len)
    ]
    return torch.stack(posed_rope_diags, dim=0)
