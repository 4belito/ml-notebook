import torch
from torch.optim.optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        defaults = {
            "lr": lr,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        # Parent class creates param_groups and self.state
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single AdaGrad update step."""
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Optional L2 weight decay: grad <- grad + λ θ
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Initialize state for accumulator parameter s_0 = 0
                state = self.state[p]
                if "accumulator" not in state:
                    state["accumulator"] = torch.zeros_like(p)

                s = state["accumulator"]

                # s_{t+1} = s_t + g_t^2 (elementwise square)
                s.add_(grad.pow(2))

                # Denominator: sqrt(s_{t+1}) + eps
                denom = s.sqrt().add_(eps)

                # Parameter update:
                # θ <- θ - lr * g / denom
                p.addcdiv_(grad, denom, value=-lr)
