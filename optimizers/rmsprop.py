import torch
from torch.optim.optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        defaults = {
            "lr": lr,
            "eps": eps,
            "alpha": alpha,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        # Parent class creates param_groups and self.state
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single AdaGrad update step."""
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                # Optional L2 weight decay: grad <- grad + λ θ
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Initialize state for accumulator parameter s_0 = 0
                state = self.state[p]
                if "square_avg" not in state:
                    state["square_avg"] = torch.zeros_like(p)
                s = state["square_avg"]

                # Exponential moving average of squared gradients: s_t
                # s_{t+1} = α s_t + (1 - α) g_t^2  (elementwise)
                s.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # Denominator: sqrt(s_{t+1}) + eps
                denom = s.sqrt().add_(eps)

                if momentum > 0.0:
                    # Initialize state for accumulator parameter v_0 = 0
                    if "velocity" not in state:
                        state["velocity"] = torch.zeros_like(p)
                    v = state["velocity"]
                    # With momentum: v_{t+1} = μ v_t + g_t / denom
                    v.mul_(momentum).addcdiv_(grad, denom)
                    # θ <- θ - lr * v_{t+1}
                    p.add_(v, alpha=-lr)
                else:
                    # Plain RMSProp: θ <- θ - lr * g_t / denom
                    p.addcdiv_(grad, denom, value=-lr)
