"""A 2D convolution layer implementation in PyTorch.

Tensor dimensions:
    b: batch size
    c_in: input channels
    c_out: output channels
    h, w: height, width
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Parameter


class Conv2d(nn.Module):
    """
    This version is designed for readability and to expose the internal logic of
    the convolution operation. It behaves similarly to `torch.nn.Conv2d` but may differ slightly in numerical results and performance due to the use of explicit Python loops and the absence of low-levelo ptimizations such as vectorized kernels or memory‐efficient stride operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # type: ignore

        # Normalize tuple inputs
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias: Parameter | None
        # Learnable parameters
        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = nn.Parameter(
            torch.empty(weight_shape, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # Kaiming initialization (no activvation function assumed)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # bias initialization (fan_in needs to be computed manually because bias is broadcasted in conv computation)
        if self.bias is not None:
            fan_in = in_channels * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: Float[Tensor, "b c_in h w"]
    ) -> Float[Tensor, "b c_out h_out w_out"]:
        N, _, H, W = x.shape
        h, w = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        Hpad, Wpad = H + 2 * pH, W + 2 * pW
        H_out = (Hpad - h) // sH + 1
        W_out = (Wpad - w) // sW + 1

        if pH > 0 or pW > 0:
            x = torch.nn.functional.pad(x, (pW, pW, pH, pH))

        out = x.new_zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):  # batch
            for oc in range(self.out_channels):  # each filter
                for i in range(H_out):
                    for j in range(W_out):
                        # top-left corner of the sliding window in the input
                        h_start = i * sH
                        w_start = j * sW
                        # get the input patch: (C_in, h, w)
                        x_patch = x[n, :, h_start : h_start + h, w_start : w_start + w]
                        # corresponding filter: (C_in, kH, kW)
                        w_filter = self.weight[oc]
                        # elementwise mul + sum
                        val = (x_patch * w_filter).sum()
                        if self.bias is not None:
                            val = val + self.bias[oc]
                        out[n, oc, i, j] = val
        return out

    @torch.no_grad()  # type: ignore
    def to_matrix(self, height: int, width: int) -> tuple[Tensor, Tensor]:
        """Dense (block-Toeplitz) matrix form of the convolution for a given input size.

        For an input of spatial size ``(height, width)`` the affine map is
        ``conv(x) = W @ x + b``. Flattening conventions (both row-major):

        * input  ``x``: ``(c_in, height, width)`` -> ``reshape(-1)``,
          i.e. column index ``((l * height) + r) * width + c``;
        * output ``y``: ``(h_out, w_out, c_out)`` -> ``reshape(-1)``,
          i.e. spatial position outer, channel inner. Reshape the result as
          ``(W @ x + b).reshape(h_out, w_out, c_out).permute(2, 0, 1)`` to recover
          the ``(c_out, h_out, w_out)`` tensor returned by ``forward``.

        Returns the pair ``(W, b)`` with ``W`` of shape ``(dy, dx)`` and ``b`` of
        shape ``(dy,)``, where ``dx = c_in * height * width`` and
        ``dy = c_out * h_out * w_out``.
        """
        h, w = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        H_out = (height + 2 * pH - h) // sH + 1
        W_out = (width + 2 * pW - w) // sW + 1

        d_x = self.in_channels * height * width
        d_y = self.out_channels * H_out * W_out
        weight = self.weight.detach()
        bias = self.bias.detach() if self.bias is not None else None

        mat = weight.new_zeros((d_y, d_x))
        vec = weight.new_zeros(d_y)
        for j in range(H_out):
            for k in range(W_out):
                for oc in range(self.out_channels):
                    row = (j * W_out + k) * self.out_channels + oc
                    if bias is not None:
                        vec[row] = bias[oc]
                    for ic in range(self.in_channels):
                        for m in range(h):
                            for n in range(w):
                                r = j * sH + m - pH  # map padded coord back to x
                                c = k * sW + n - pW
                                if 0 <= r < height and 0 <= c < width:
                                    col = (ic * height + r) * width + c
                                    mat[row, col] = weight[oc, ic, m, n]
        return mat, vec
