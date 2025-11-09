"""Kaiming/He initialization methods for neural network weights."""

import math

import torch


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor):
    """Calculates the fan-in and fan-out for a given tensor."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in/out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # convolutional weights e.g. [out_channels, in_channels, kH, kW, ...]
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def calculate_gain(nonlinearity: str, a: float = 0.0):
    nonlinearity = nonlinearity.lower()
    match nonlinearity:
        case "sigmoid" | "linear" | "conv1d" | "conv2d" | "conv3d":
            return 1.0
        case "tanh":
            return 5.0 / 3  # example value
        case "relu":
            return math.sqrt(2.0)
        case "leaky_relu":
            return math.sqrt(2.0 / (1 + a * a))
        case _:
            raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0):
    """
    Fills the input `tensor` with values drawn from U(-bound, bound)
    according to Xavier/Glorot uniform initialization.

    Reference:
    Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks".
    """

    # 1) Compute fan_in and fan_out
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    # 2) Compute standard deviation and uniform bound
    bound = gain * math.sqrt(6 / (fan_in + fan_out))

    # 3) Fill tensor
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0):
    """
    Fills the input `tensor` with values drawn from N(0, std^2)
    according to Xavier/Glorot normal initialization.

    Reference:
    Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks".
    """

    # 1) Compute fan_in and fan_out
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    # 2) Compute standard deviation
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    # 3) Fill tensor
    with torch.no_grad():
        return tensor.normal_(0.0, std)


def kaiming_uniform_(
    tensor: torch.Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
    """
    Fills the input `tensor` with values drawn from U(-bound, bound)
    according to Kaiming/He initialization.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError("mode should be 'fan_in' or 'fan_out'")

    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0 / fan) * gain

    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
