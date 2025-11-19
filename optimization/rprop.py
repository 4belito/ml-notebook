import torch
from torch.optim import Optimizer


class RProp(Optimizer):
    r"""
    RProp (Resilient Backpropagation), in the style of torch.optim.Rprop.

    Per coordinate i:

        q_{t,i} = g_{t-1,i} * g_{t,i}

        Δ_{t,i} =
            min(η_plus  * Δ_{t-1,i}, Δ_max)   if q_{t,i} > 0  (same sign)
            max(η_minus * Δ_{t-1,i}, Δ_min)   if q_{t,i} < 0  (sign flip)
            Δ_{t-1,i}                         otherwise

        θ_{t+1,i} = θ_{t,i} - sign(g_{t,i}) * Δ_{t,i}

    Here lr is only used to set the initial Δ_0 = lr.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50.0),
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        eta_minus, eta_plus = etas
        if not (0.0 < eta_minus < 1.0 < eta_plus):
            raise ValueError(f"Invalid etas: {etas}")
        step_min, step_max = step_sizes
        if not (step_min > 0.0 and step_max >= step_min):
            raise ValueError(f"Invalid step_sizes: {step_sizes}")

        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Perform a single RProp update (no closure, returns None)."""
        for group in self.param_groups:
            lr = group["lr"]
            eta_minus, eta_plus = group["etas"]
            step_min, step_max = group["step_sizes"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]

                # Initialize state once
                if len(state) == 0:
                    state["prev_grad"] = torch.zeros_like(p)
                    # step_size_0 = lr (PyTorch uses lr this way)
                    state["step_size"] = torch.full_like(p, lr)

                prev_grad = state["prev_grad"]
                step_size = state["step_size"]

                # q_t = g_{t-1} * g_t, tells us if signs agree
                q = prev_grad * grad

                same_mask = q > 0  # same sign
                flip_mask = q < 0  # sign flip
                # q == 0 -> keep step_size

                # Build multiplicative factor for Δ_t: {η_plus, η_minus, 1}
                factor = torch.ones_like(grad)
                factor[same_mask] = eta_plus
                factor[flip_mask] = eta_minus

                # Update and clamp step sizes
                step_size.mul_(factor)
                step_size.clamp_(step_min, step_max)

                # When sign flips, ignore current gradient for that coordinate
                grad_eff = grad.clone()
                grad_eff[flip_mask] = 0.0

                # RProp update: θ_{t+1} = θ_t − sign(g_eff) * Δ_t
                p.add_(torch.sign(grad_eff) * step_size, alpha=-1.0)

                # Store effective gradient for next step
                prev_grad.copy_(grad_eff)

        return None
