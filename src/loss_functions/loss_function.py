import torch
from torch import Tensor, nn


class MSELoss:
    def __call__(self, pred_y: Tensor, y: Tensor) -> Tensor:
        return ((pred_y - y) ** 2).mean()


class BCEWithLogitsLoss:
    def __call__(self, pred_y: Tensor, y: Tensor) -> Tensor:
        p = torch.sigmoid(pred_y)
        return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


class CrossEntropyLoss:
    def __call__(self, pred_y: Tensor, y: Tensor) -> Tensor:
        probs = nn.functional.softmax(pred_y, dim=1)
        p_yi = probs[torch.arange(y.shape[0]), y]
        log_p_yi = torch.log(p_yi)
        return -log_p_yi.mean()
