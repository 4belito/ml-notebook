"""Stochastic Gradient Descent (SGD) Optimizer Implementation."""

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer

Params = torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, Iterable[torch.Tensor]]]


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if nesterov and momentum == 0.0:
            raise ValueError("Nesterov requires momentum > 0.")
        if nesterov and dampening != 0.0:
            # this is also what PyTorch enforces
            raise ValueError("Nesterov is used with dampening = 0.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]  # η
            mu = group["momentum"]  # μ
            damp = group["dampening"]  # δ
            wd = group["weight_decay"]  # λ
            nesterov = group["nesterov"]

            for p in group["params"]:
                grad = p.grad  # g_t
                if grad is None:
                    continue

                # ---- 1) Weight decay: g_t = g_t + λ θ_t ----
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                # ---- 2) Momentum + dampening on g̃_t ----
                if mu != 0.0:
                    state = self.state[p]

                    # v_0 = 0 the first time we see this parameter
                    if "velocity" not in state:
                        state["velocity"] = torch.zeros_like(p)

                    v = state["velocity"]

                    # v_{t+1} = μ v_t + (1 - δ) g̃_t
                    v.mul_(mu).add_(grad, alpha=1.0 - damp)
                    state["velocity"] = v

                    # ---- 3) Compute direction d_t ----
                    if nesterov:
                        # d_t = g_t + μ v_{t+1}
                        d_p = grad.add(v, alpha=mu)
                    else:
                        # d_t = v_{t+1}
                        d_p = v
                else:
                    # No momentum: direction is just (optionally decayed) gradient
                    d_p = grad

                # ---- 4) θ_{t+1} = θ_t - η d_t ----
                p.add_(d_p, alpha=-lr)
