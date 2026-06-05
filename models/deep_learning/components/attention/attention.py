"""
Educational Multi-Head Attention: MultiheadAttention + SelfAttention.

Goal: an explicit, readable attention implementation that exposes every
projection and the score computation step-by-step. Correctness is verified
by loading weights from nn.MultiheadAttention and comparing outputs —
non-masked positions match exactly; deviations from nn are documented below.

Similarities with nn.MultiheadAttention:
    - Same scaled dot-product attention formula: softmax(QK^T / sqrt(dk)) V.
    - Same multi-head split/merge via reshape (einops rearrange).
    - Separate q_proj / k_proj / v_proj + out_proj structure.
    - Weights are identical after load_weights_from_torch_mha.

Discrepancies vs nn.MultiheadAttention:
    - Always batch-first (b, n, c); nn default is seq-first (n, b, c).
    - Weights split into separate q_proj / k_proj / v_proj Linear layers;
      nn packs them into a single in_proj_weight. Random init therefore
      differs at the same seed, but values match after weight loading.
    - Attention dropout uses an explicit nn.Dropout. On CPU/MPS,
      F.scaled_dot_product_attention uses the same global PyTorch RNG
      (math backend, no fused kernel), so training outputs match exactly.
      On CUDA with flash attention the kernel uses a separate Philox RNG
      not reset by torch.manual_seed, so dropout masks diverge there.
    - Fully-masked rows (all -inf in attn_mask): we apply nan_to_num and
      return 0, matching F.scaled_dot_product_attention (the path used
      by nn.TransformerEncoderLayer internally with need_weights=False).
      Calling nn.MultiheadAttention directly with need_weights=True (the
      default) instead returns nan for those rows.
    - attn_mask accepts bool (True=block, matches nn) or additive float
      (0=attend, -inf=block); key_padding_mask and is_causal are not supported.
    - Attention weights (b, h, m, n) are always returned alongside the output.
"""

from __future__ import annotations

import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


class SelfAttention(nn.Module):
    """Thin wrapper around MultiheadAttention for the self-attention case (Q=K=V=x).

    Used by TransformerEncoderLayer and TransformerDecoderLayer so they work
    with a single input tensor instead of separate Q, K, V tensors.

    Tensor dimension conventions:
        b: batch size
        n: sequence length
        d: embedding dimension (mbed_dim); Q=K=V=x so input and output share it.
        h: number of attention heads

    Args:
        mbed_dim: embedding dimension (d_model).
        num_heades: number of attention heads.
        dropout: dropout probability on attention weights.
        bias: whether projection layers include a bias term.
        add_bias_kv: if True, append learnable bias to K and V sequences.
        kdim: key input dimension (defaults to mbed_dim).
        vdim: value input dimension (defaults to mbed_dim).
        device: target device.
        dtype: target dtype.
    """

    def __init__(
        self,
        mbed_dim: int,
        num_heades: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.mha = MultiheadAttention.from_torch_mha(
            mbed_dim,
            num_heades,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            kdim=kdim,
            vdim=vdim,
            device=device,
            dtype=dtype,
        )

    def load_weights_from_torch_mha(self, src: nn.MultiheadAttention):
        """Load weights from an nn.MultiheadAttention."""
        self.mha.load_weights_from_torch_mha(src)

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        attn_mask: Bool[Tensor, "n n"]
        | Bool[Tensor, "(b*h) n n"]
        | Bool[Tensor, "b h n n"]
        | Float[Tensor, "n n"]
        | Float[Tensor, "(b*h) n n"]
        | Float[Tensor, "b h n n"]
        | None = None,
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b h n n"]]:
        return self.mha(x, x, x, attn_mask=attn_mask)


class MultiheadAttention(nn.Module):
    """General Multi-Head Attention with explicit separate projections.

    Supports different Q / K / V input dimensions (cq, ck, cv) and head
    dimensions (dk, dv), making the full generality of MHA visible.
    Use from_torch_mha for the standard case (cq=ck=cv=d_model, dk=dv=d_model/h).

    All deviations from nn.MultiheadAttention are listed in the module docstring.

    Tensor dimension conventions:
        b:  batch size
        m:  query sequence length
        n:  key/value sequence length
        cq, ck, cv: input dims for Q, K, V
        dk: per-head key/query dimension
        dv: per-head value dimension
        do: output dimension
        h:  number of heads

    Args:
        cq: query input dimension.
        ck: key input dimension.
        cv: value input dimension.
        dk: per-head key/query dimension.
        dv: per-head value dimension.
        do: output dimension.
        h: number of attention heads.
        bias: whether projection layers include a bias term.
        dropout: dropout probability applied to attention weights.
        add_bias_kv: if True, append learnable bias vectors to K and V.
        device: target device.
        dtype: target dtype.
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
        dropout: float = 0.0,
        add_bias_kv: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
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
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
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

    def load_weights_from_torch_mha(self, src: nn.MultiheadAttention):
        """Load weights from an nn.MultiheadAttention."""
        with torch.no_grad():
            if src.kdim == src.embed_dim and src.vdim == src.embed_dim:
                d = src.in_proj_weight.shape[0] // 3
                self.q_proj.weight.copy_(src.in_proj_weight[0:d])
                self.k_proj.weight.copy_(src.in_proj_weight[d : 2 * d])
                self.v_proj.weight.copy_(src.in_proj_weight[2 * d :])
            else:
                self.q_proj.weight.copy_(src.q_proj_weight)
                self.k_proj.weight.copy_(src.k_proj_weight)
                self.v_proj.weight.copy_(src.v_proj_weight)

            if self.bias:
                d = src.in_proj_bias.shape[0] // 3
                self.q_proj.bias.copy_(src.in_proj_bias[0:d])
                self.k_proj.bias.copy_(src.in_proj_bias[d : 2 * d])
                self.v_proj.bias.copy_(src.in_proj_bias[2 * d :])

            if self.add_bias_kv and src.bias_k is not None and src.bias_v is not None:
                self.bias_k.copy_(src.bias_k.squeeze(0))
                self.bias_v.copy_(src.bias_v.squeeze(0))

            self.out_proj.weight.copy_(src.out_proj.weight)
            if self.bias:
                self.out_proj.bias.copy_(src.out_proj.bias)

    @classmethod
    def from_torch_mha(
        cls: type[MultiheadAttention],
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        add_bias_kv: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> MultiheadAttention:
        """Construct an instance with the standard nn.MultiheadAttention shape.

        Sets dk = dv = embed_dim // num_heads and do = embed_dim,
        matching PyTorch's default MHA configuration.
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        dh = embed_dim // num_heads
        return cls(
            cq=embed_dim,
            ck=kdim if kdim is not None else embed_dim,
            cv=vdim if vdim is not None else embed_dim,
            dk=dh,
            dv=dh,
            do=embed_dim,
            h=num_heads,
            bias=bias,
            dropout=dropout,
            add_bias_kv=add_bias_kv,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        Q: Float[Tensor, "b m cq"],
        K: Float[Tensor, "b n ck"],
        V: Float[Tensor, "b n cv"],
        attn_mask: Bool[Tensor, "m n"]
        | Bool[Tensor, "(b*h) m n"]
        | Bool[Tensor, "b h m n"]
        | Float[Tensor, "m n"]
        | Float[Tensor, "(b*h) m n"]
        | Float[Tensor, "b h m n"]
        | None = None,
    ) -> tuple[Float[Tensor, "b m do"], Float[Tensor, "b h m n"]]:
        """Compute multi-head attention output.

        Args:
            Q: query tensor (b, m, cq).
            K: key tensor   (b, n, ck).
            V: value tensor (b, n, cv).
            attn_mask: mask applied to scores before softmax.
                Bool (True=block, False=attend) or additive float (0=attend, -inf=block)
                Matches nn.MultiheadAttention convention.
                Shape (m, n), (b*h, m, n), or (b, h, m, n).
                Fully-masked rows output a zero vector.

        Returns:
            Attention output (b, m, do) and attention weights (b, h, m, n).
        """
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
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))  # True=block
            else:
                scores = scores + attn_mask  # additive float: 0=attend, -inf=block

        # Numerically stable softmax(QK^T / sqrt(dk)).
        # Standard F.softmax fails for fully-masked rows: softmax([-inf,...])
        # produces nan, and even though nan_to_num zeros the forward output,
        # the softmax Jacobian contains nan so nan * 0 = nan gradients
        # propagate back and corrupt q_proj / k_proj weights during training.
        #
        # Fix: subtract the row max before exp (standard log-sum-exp trick).
        # For fully-masked rows the max is -inf; clamp it to 0 so that
        # (-inf) - 0 = -inf and exp(-inf) = 0, avoiding (-inf) - (-inf) = nan.
        # Clamp the denominator to avoid 0/0 for fully-masked rows → output 0.
        # This matches F.scaled_dot_product_attention (need_weights=False path).
        scores_scaled = scores / (self.dk**0.5)

        # softmax(QK^T/sqrt(dk))V
        # attn = scores_scaled.softmax(dim=-1).nan_to_num(neginf=0.0)
        attn = self._softmax_scores(scores_scaled)
        attn = self.dropout(attn)
        o = torch.einsum("bhmn, bhnv -> bhmv", attn, r_v)

        # Reshape back
        r_o = einops.rearrange(o, "b h m dv -> b m (h dv)")

        # Final linear projection
        proj_o = self.out_proj(r_o)  #
        return proj_o, attn.mean(dim=1)  # average attention weights over heads

    def _softmax_scores(self, scores: Tensor) -> Tensor:
        """Compute softmax(QK^T / sqrt(dk)) with numerical stability fixes."""
        row_max = scores.amax(dim=-1, keepdim=True).nan_to_num(neginf=0.0)
        exp_s = (scores - row_max).exp()
        attn = exp_s / exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return attn


def create_causal_mask(
    shape: tuple[int, int], device: torch.device | None = None
) -> Bool[Tensor, "m n"]:
    """Bool causal mask: True = block future positions."""
    return torch.triu(torch.ones(shape, dtype=torch.bool, device=device), diagonal=1)


def create_random_mask(
    shape: tuple[int, int], device: torch.device | None = None
) -> Bool[Tensor, "m n"]:
    """Bool mask with ~50% blocked positions. True = block."""
    return torch.rand(shape, device=device) > 0.5
