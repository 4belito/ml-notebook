import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Basic Adam optimizer (Kingma & Ba, 2015) with bias correction.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not (0.0 < beta1 < 1.0 and 0.0 < beta2 < 1.0):
            raise ValueError(f"Invalid beta values: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single Adam update step."""
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                # Apply L2 weight decay (same style as torch.optim.Adam)
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)  # first moment
                    state["v"] = torch.zeros_like(p)  # second moment

                m = state["m"]
                v = state["v"]
                state["step"] += 1
                t = state["step"]

                # 1) Update biased first and second moments
                # m_t = β1 m_{t-1} + (1 - β1) g_t      # first moment (momentum-like)
                # v_t = β2 v_{t-1} + (1 - β2) g_t^2    # second moment (variance-like)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 2) Compute bias-corrected moments
                # m̂_t = m_t / (1 - β1^t)
                # v̂_t = v_t / (1 - β2^t)
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # 3) Parameter update
                # θ_{t+1} = θ_t - lr * m̂_t / (sqrt(v̂_t) + ε)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
