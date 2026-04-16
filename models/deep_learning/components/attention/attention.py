"""Multi-Head Attention module with (general) parameter."""

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn


class SelfAttention(nn.Module):
    """
    Multi-Head Attention module with (general) parameters
    b: Batch dimensioin
    n: sequence length
    c: input dimensions for Q, K, V
    dk, dv: dimensions for each head's Q, K and V
    do: output dimension
    h: number of heads.
    """

    def __init__(
        self,
        mbed_dim: int,
        num_heades: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            mbed_dim,
            num_heades,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, "b n c"],
        attn_mask: Float[Tensor, "n n"]
        | Float[Tensor, "(b*h) n n"]
        | Float[Tensor, "b h n n"]
        | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ):
        return self.mha(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]


class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention module with (general) parameters
    b: Batch dimensioin
    m: querry sequence length
    n: key-value sequence length
    cq, ck, cv: input dimensions for Q, K, V
    dk, dv: dimensions for each head's Q, K and V
    do: output dimension
    h: number of heads.

    The initialization of the weights differs from PyTorch’s `nn.MultiheadAttention`.
    Here we use standard `nn.Linear` initialization (Xavier uniform for weights and
    zeros for biases) for clarity and simplicity.
    """

    def __init__(
        self,
        cq: int,
        ck: int,
        cv: int,
        dk: int,
        dv: int,
        do: int,
        h: int,
        bias: bool = True,
        add_bias_kv: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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
        # Q -> QW_q+1_Mb^T_q
        self.q_proj = nn.Linear(cq, dk * h, bias, self.device, self.dtype)
        # K -> KW_k+1_Mb^T_k
        self.k_proj = nn.Linear(ck, dk * h, bias, self.device, self.dtype)
        # V -> VW_v+1_Mb^T_v
        self.v_proj = nn.Linear(cv, dv * h, bias, self.device, self.dtype)

        self.out_proj = nn.Linear(dv * h, do, bias, self.device, self.dtype)
        if self.add_bias_kv:
            self.bias_k = nn.Parameter(
                torch.zeros(1, 1, dk * h, device=self.device, dtype=self.dtype)
            )
            self.bias_v = nn.Parameter(
                torch.zeros(1, 1, dv * h, device=self.device, dtype=self.dtype)
            )

    def forward(
        self,
        Q: Float[Tensor, "b m cq"],
        K: Float[Tensor, "b n ck"],
        V: Float[Tensor, "b n cv"],
        attn_mask: Float[Tensor, "m n"]
        | Float[Tensor, "(b*h) m n"]
        | Float[Tensor, "b h m n"]
        | None = None,
    ) -> Float[Tensor, "b m do"]:
        """Forward pass of the MHA module."""
        # Linear projections
        proj_q = self.q_proj(Q)  # Q=QW_q+1_Mb^T_q
        proj_k = self.k_proj(K)  # K=KW_k+1_Mb^T_k
        proj_v = self.v_proj(V)  # V=VW_v+1_Mb^T_v
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
        if attn_mask is not None:
            match attn_mask.dim():
                case 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                case 3:
                    attn_mask = einops.rearrange(
                        attn_mask, "(b h) m n -> b h m n", h=self.h
                    )
                case 4:
                    pass
                case _:
                    raise ValueError("attn_mask has incorrect dimensions")
            scores += attn_mask

        # softmax(QK^T/sqrt(dk))
        attn = nn.functional.softmax(scores / (self.dk**0.5), dim=-1)

        # softmax(QK^T/sqrt(dk))V
        o = torch.einsum("bhmn, bhnv -> bhmv", attn, r_v)

        # Reshape back
        r_o = einops.rearrange(o, "b h m dv -> b m (h dv)")

        # Final linear projection
        proj_o = self.out_proj(r_o)
        return proj_o
