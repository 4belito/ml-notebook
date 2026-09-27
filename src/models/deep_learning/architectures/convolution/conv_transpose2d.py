"""A 2D transposed convolution ("deconvolution") layer implementation in PyTorch.

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


class ConvTranspose2d(nn.Module):
    """
    Readable transposed-convolution (a.k.a. deconvolution) layer.

    It mirrors `torch.nn.ConvTranspose2d`: the learnable weight has shape
    ``(in_channels, out_channels, kH, kW)`` and each input pixel scatters a
    scaled copy of the kernel onto the output grid. Explicit Python loops make
    the scatter logic visible at the cost of speed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
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
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias: Parameter | None
        # Learnable parameters: note the transposed channel order (in, out, ...)
        weight_shape = (in_channels, out_channels, *kernel_size)
        self.weight = nn.Parameter(
            torch.empty(weight_shape, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # Match torch.nn.ConvTranspose2d initialization exactly.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # fan_in is computed from the weight (out_channels * kH * kW here).
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(  # type: ignore
                self.weight
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: Float[Tensor, "b c_in h w"]
    ) -> Float[Tensor, "b c_out h_out w_out"]:
        N, _, H, W = x.shape
        h, w = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        opH, opW = self.output_padding
        H_out = (H - 1) * sH - 2 * pH + h + opH
        W_out = (W - 1) * sW - 2 * pW + w + opW

        out = x.new_zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):  # batch
            for ic in range(self.in_channels):  # input channel that scatters
                for i in range(H):  # input row
                    for j in range(W):  # input column
                        val = x[n, ic, i, j]
                        for oc in range(self.out_channels):
                            for m in range(h):  # kernel row
                                for k in range(w):  # kernel column
                                    # scatter position, then crop the padding
                                    a = i * sH + m - pH
                                    b = j * sW + k - pW
                                    if 0 <= a < H_out and 0 <= b < W_out:
                                        out[n, oc, a, b] += (
                                            val * self.weight[ic, oc, m, k]
                                        )
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
