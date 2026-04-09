from .loss_function import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from typing import Callable
from torch import Tensor


Loss = Callable[[Tensor, Tensor], Tensor]

__all__ = ["MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "Loss"]
