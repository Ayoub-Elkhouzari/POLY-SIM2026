"""
Multi-Scale Momentum Muon (M3) Optimizer
==========================================
Based on: "Nested Learning" (Behrouz et al.), Algorithm 1.
Reference implementation: github.com/kmccleary3301/nested_learning

Core idea: two momentum buffers at different timescales, both orthogonalized
via Newton-Schulz before being combined into the update step.

  m1  — fast momentum, updated every step:
          m1 += β1 * g

  m2  — slow momentum, updated every `slow_chunk` steps:
          slow_buffer accumulates raw gradients each step
          every slow_chunk steps:  m2 += β3 * slow_buffer;  slow_buffer = 0

  o1, o2  — Newton-Schulz orthogonalization of m1 and m2 (2D params only)

  v   — second-moment estimate (updated every step, like Adam)

  update = (o1 + α * o2) / (√v + ε)
  θ      = θ − lr * update

Orthogonalization (Newton-Schulz, `ns_steps` iterations):
  Normalizes the update matrix so that the effective step size is
  well-conditioned regardless of gradient scale — key for large 2D
  weight matrices (attention, projection layers).
  1D tensors (biases, norms) are left unchanged.

Default slow_chunk tuning for POLY-SIM:
  ~3231 EN + ~7443 UR samples, batch_size=32 → ~202 optimizer steps/epoch
  × ~58 epochs (early stop) = ~11 700 total steps.
  slow_chunk=500  →  slow buffer updates ~23 times across training  ✓
  Sweep range: [100, 250, 500, 1000] — see scripts/sweep_m3.sh.

Weight decay is decoupled (AdamW-style) for consistency with the rest of
the codebase.
"""

import torch
from torch.optim import Optimizer


# ── Newton-Schulz orthogonalization ─────────────────────────────────────────

def _newton_schulz(matrix: torch.Tensor, steps: int, eps: float = 1e-6) -> torch.Tensor:
    """Approximate orthogonalization of a 2D matrix via Newton-Schulz iterations.

    Normalizes `matrix` so its singular values are close to 1, giving a
    well-conditioned update direction.  `steps=3` is sufficient in practice.
    """
    dtype = matrix.dtype
    device = matrix.device
    _, n = matrix.shape
    x = matrix / (torch.linalg.norm(matrix) + eps)
    eye = torch.eye(n, device=device, dtype=dtype)
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * eye - x.T @ x)
    return x


def _orthogonalize(tensor: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """Orthogonalize a tensor if it is 2D; pass 1D tensors through unchanged."""
    if tensor.ndim < 2:
        return tensor
    mat = tensor.reshape(tensor.shape[0], -1)
    return _newton_schulz(mat, steps=steps, eps=eps).reshape_as(tensor)


# ── M3 optimizer ─────────────────────────────────────────────────────────────

class M3(Optimizer):
    """Multi-Scale Momentum Muon (M3) optimizer.

    Args:
        params:       model parameters or param groups
        lr:           learning rate (default: 1e-3)
        beta1:        fast-momentum scaling factor (default: 0.9)
        beta2:        second-moment decay, like Adam β₂ (default: 0.999)
        beta3:        slow-momentum scaling factor (default: 0.9)
        alpha:        weight of the slow momentum term in the update
                      (default: 1.0 — equal weight to fast and slow)
        eps:          numerical stability for the denominator (default: 1e-8)
        ns_steps:     Newton-Schulz iterations for orthogonalization
                      (default: 3 — sufficient for practical convergence)
        slow_chunk:   number of optimizer steps between slow-buffer flushes
                      (default: 10 — tuned for ~240 total steps in this repo;
                       increase for longer training runs)
        weight_decay: decoupled weight decay, AdamW-style (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9,
        alpha: float = 1.0,
        eps: float = 1e-8,
        ns_steps: int = 3,
        slow_chunk: int = 10,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3: {beta3}")
        if slow_chunk < 1:
            raise ValueError(f"slow_chunk must be >= 1, got {slow_chunk}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            ns_steps=ns_steps,
            slow_chunk=slow_chunk,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr         = group["lr"]
            beta1      = group["beta1"]
            beta2      = group["beta2"]
            beta3      = group["beta3"]
            alpha      = group["alpha"]
            eps        = group["eps"]
            ns_steps   = group["ns_steps"]
            slow_chunk = group["slow_chunk"]
            wd         = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("M3 does not support sparse gradients")

                state = self.state[p]

                # ── Initialise state on first step ───────────────────────────
                if not state:
                    state["step"]         = 0
                    state["m1"]           = torch.zeros_like(p)   # fast momentum
                    state["m2"]           = torch.zeros_like(p)   # slow momentum
                    state["v"]            = torch.zeros_like(p)   # second moment
                    state["slow_buffer"]  = torch.zeros_like(p)   # gradient accumulator
                    state["o2"]           = torch.zeros_like(p)   # cached slow ortho

                state["step"] += 1
                t = state["step"]

                # ── Decoupled weight decay (AdamW-style) ─────────────────────
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ── Fast momentum (updated every step) ───────────────────────
                m1 = state["m1"]
                m1.add_(grad, alpha=beta1)

                # ── Second moment (updated every step) ───────────────────────
                v = state["v"]
                v.addcmul_(grad, grad, value=beta2)

                # ── Slow buffer accumulation ─────────────────────────────────
                state["slow_buffer"].add_(grad)

                # ── Orthogonalize fast momentum ──────────────────────────────
                o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)
                o2 = state["o2"]

                # ── Parameter update ─────────────────────────────────────────
                denom  = v.sqrt().add_(eps)
                update = o1.add(o2, alpha=alpha)
                p.addcdiv_(update, denom, value=-lr)

                # ── Slow momentum flush (every slow_chunk steps) ─────────────
                if slow_chunk > 0 and t % slow_chunk == 0:
                    m2 = state["m2"]
                    m2.add_(state["slow_buffer"], alpha=beta3)
                    state["slow_buffer"].zero_()
                    state["o2"] = _orthogonalize(m2, steps=ns_steps, eps=eps)

        return loss
