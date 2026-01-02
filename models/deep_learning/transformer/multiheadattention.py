"""Multi-Head Attention module with (general) parameter."""

import einops
import torch


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module with (general) parameters
    cq, ck, cv: input dimensions for Q, K, V
    dk, dv: dimensions for each head's Q, K and V
    do: output dimension
    h: number of heads.

    The initialization of the weights differs from PyTorch’s `nn.MultiheadAttention`.
    Here we use standard `nn.Linear` initialization (Xavier uniform for weights and
    zeros for biases) for clarity and simplicity.
    """

    def __init__(
        self, cq, ck, cv, dk, dv, do, h, bias=True, add_bias_kv=False, device=None, dtype=None
    ):
        super().__init__()
        assert dk % h == 0, "dk must be divisible by h"
        self.cq = cq
        self.ck = ck
        self.cv = cv
        self.dk = dk
        self.dv = dv
        self.do = do
        self.h = h
        self.add_bias_kv = add_bias_kv
        self.device = device
        self.dtype = dtype
        # Q -> QW_q+B_q
        self.q_proj = torch.nn.Linear(cq, dk * h, bias, self.device, self.dtype)
        # K -> KW_k+B_k
        self.k_proj = torch.nn.Linear(ck, dk * h, bias, self.device, self.dtype)
        # V -> VW_v+B_v
        self.v_proj = torch.nn.Linear(cv, dv * h, bias, self.device, self.dtype)

        self.out_proj = torch.nn.Linear(dv * h, do, bias, self.device, self.dtype)
        if self.add_bias_kv:
            self.bias_k = torch.nn.Parameter(
                torch.zeros(1, 1, dk * h, device=self.device, dtype=self.dtype)
            )
            self.bias_v = torch.nn.Parameter(
                torch.zeros(1, 1, dv * h, device=self.device, dtype=self.dtype)
            )

    def forward(self, Q, K, V):
        """Forward pass of the MHA module."""
        # Linear projections
        proj_q = self.q_proj(Q)  # Q=QW_q+B_q
        proj_k = self.k_proj(K)  # K=KW_k+B_k
        proj_v = self.v_proj(V)  # V=VW_v+B_v
        if self.add_bias_kv:
            # append bias to the key and value sequences
            batch_size = proj_k.shape[0]
            proj_k = torch.cat([proj_k, self.bias_k.repeat(batch_size, 1, 1)], dim=1)
            proj_v = torch.cat([proj_v, self.bias_v.repeat(batch_size, 1, 1)], dim=1)

        # Reshape for multi-head attention
        r_q = einops.rearrange(proj_q, "b m (h dk) -> b h m dk", h=self.h)
        r_k = einops.rearrange(proj_k, "b n (h dk) -> b h n dk", h=self.h)
        r_v = einops.rearrange(proj_v, "b n (h dv) -> b h n dv", h=self.h)

        # QK^T
        scores = torch.einsum("bhmd, bhnd -> bhmn", r_q, r_k)

        # softmax(QK^T/sqrt(dk))

        attn = torch.nn.functional.softmax(scores / (self.dk**0.5), dim=-1)

        # softmax(QK^T/sqrt(dk))V
        o = torch.einsum("bhmn, bhnv -> bhmv", attn, r_v)

        # Reshape back
        r_o = einops.rearrange(o, "b h m dv -> b m (h dv)")

        # Final linear projection
        proj_o = self.out_proj(r_o)
        return proj_o
