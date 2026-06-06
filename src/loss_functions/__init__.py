from collections.abc import Callable

from torch import Tensor

from .loss_function import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

Loss = Callable[[Tensor, Tensor], Tensor]

__all__ = ["MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "Loss"]
