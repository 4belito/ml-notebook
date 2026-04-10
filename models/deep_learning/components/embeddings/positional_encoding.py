import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, ndim: int = 1, base: float = 10000.0):
        super().__init__()
        assert embed_dim % (2 * ndim) == 0, (
            "Embedding dimension must be divisible by 2 * number of dimensions"
        )
        self.ndim = ndim
        half_dim = embed_dim // (2 * self.ndim)
        # Frequencies (Transformer-style scaling)
        # freq[i] = base^(-i/dim) for i in range(half_dim)
        freq = torch.exp(-math.log(base) * torch.arange(half_dim) / half_dim)
        self.freq: Tensor
        self.register_buffer("freq", freq)

    def forward(self, spatial_dimensions: tuple[int, ...], sampling_factor: int = 1) -> Tensor:
        """
        space_dim: (D1, D2, ..., DN)
        returns: (..., embed_dim)
        """
        assert len(spatial_dimensions) == self.ndim, (
            "Input spatial dimensions must match the initialized number of dimensions"
        )
        x = self.make_coords(
            spatial_dimensions, self.freq.device, sampling_factor=sampling_factor
        )  # (..., D)
        x = x[..., None, :]
        freq = self.freq[:, None]
        x = x * freq
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return emb.flatten(start_dim=-2)

    @staticmethod
    def make_coords(
        shape: tuple[int, ...],
        device: torch.device,
        sampling_factor: int = 1,
    ) -> Tensor:
        """
        shape: (D1, D2, ..., DN)
        sampling_factor: factor by which to increase sampling density
        """

        axes = [
            torch.linspace(
                0,
                s - 1,
                steps=s * sampling_factor,
                device=device,
            )
            for s in shape
        ]

        grids = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack(grids, dim=-1)

        return coords
