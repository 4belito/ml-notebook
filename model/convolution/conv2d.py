"""A 2D convolution layer implementation in PyTorch."""

import math

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    """
    This version is designed for readability and to expose the internal logic of
    the convolution operation. It behaves similarly to `torch.nn.Conv2d` but may differ slightly in numerical results and performance due to the use of explicit Python loops and the absence of low-levelo ptimizations such as vectorized kernels or memory‐efficient stride operations.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

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

        # Learnable parameters
        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Kaiming initialization (no activvation function assumed)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # bias initialization (fan_in needs to be computed manually because bias is broadcasted in conv computation)
        if self.bias is not None:
            fan_in = in_channels * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
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
