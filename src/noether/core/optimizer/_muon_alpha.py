#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Muon with an α-interpolated update.

Extends :class:`torch.optim.Muon` with a scalar knob ``α ∈ [0, 1]`` that blends between
the Frobenius-normalized raw momentum and the Newton–Schulz orthogonalized update::

    update = (1 - α) · √min(m, n) · M_hat  +  α · NS(M_hat)

where ``M_hat = M / ‖M‖_F``. The ``√min(m, n)`` factor Frobenius-matches the left
branch to the right branch (NS output has singular values ≈ 1 → ‖NS‖_F ≈ √min(m, n)),
so at α = 1 the expression collapses exactly to torch's Muon update and at α = 0 we
get rescaled raw momentum with no NS compute.

Derived from torch.optim._muon (torch 2.11). We vendor rather than monkey-patch
because torch's ``_zeropower_via_newtonschulz`` performs the Frobenius normalization
internally and only returns the NS output — we need ``M_hat`` separately.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _normalize(grad: Tensor, eps: float) -> Tensor:
    """Frobenius-normalize ``grad`` to have unit norm (in bf16, matching torch.optim.Muon)."""
    out = grad.bfloat16()
    out.div_(out.norm().clamp(min=eps))
    return out


def _newton_schulz(normalized_grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int) -> Tensor:
    """Newton–Schulz iteration applied to an already Frobenius-normalized matrix.

    Identical to the iteration in torch.optim._muon._zeropower_via_newtonschulz but without
    the internal normalization step (we've split that out into :func:`_normalize`).
    """
    if ns_steps >= 100:
        raise ValueError("Number of NS steps must be less than 100")
    if normalized_grad.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    a, b, c = ns_coefficients
    ortho = normalized_grad
    transposed = ortho.size(0) > ortho.size(1)
    if transposed:
        ortho = ortho.T
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.T
    return ortho


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Per-matrix LR adjustment, identical to torch.optim._muon._adjust_lr."""
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class _MuonAlpha(Optimizer):
    """Muon optimizer with an α-interpolated update.

    Drop-in replacement for :class:`torch.optim.Muon` that accepts an additional ``alpha``
    hyperparameter. When ``alpha == 1.0`` the update is bit-identical to torch.optim.Muon
    (same NS iteration, same LR adjustment). When ``alpha == 0.0`` the NS iteration is
    skipped entirely and the update is ``√min(m, n) · M / ‖M‖_F``. Intermediate values
    produce a convex blend of the two (Frobenius-matched, so the total update norm stays
    roughly constant across α).

    ``alpha`` is stored per param group, so external schedulers can mutate it each step.

    After each :meth:`step`, the optimizer exposes three diagnostic attributes populated
    from Muon-managed params in the most recent step:

    * ``last_alpha``: mean α across param groups.
    * ``last_update_frob_mean``: mean Frobenius norm of the blended update.
    * ``last_cos_mhat_ns``: mean cosine similarity between ``M_hat`` and ``NS(M_hat)``.

    These are intended to be consumed by a logging callback.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        alpha: float = 1.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate must be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Momentum must be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay must be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"adjust_lr_fn {adjust_lr_fn!r} is not supported")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1] but is: {alpha}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "alpha": alpha,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"_MuonAlpha only supports 2D parameters but got shape {tuple(p.size())}")

        self.last_alpha: float = alpha
        self.last_update_frob_mean: float = 0.0
        self.last_cos_mhat_ns: float = 0.0

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
    ) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("_MuonAlpha does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("_MuonAlpha does not support sparse gradients")
            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
            momentum_bufs.append(state["momentum_buffer"])

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # diagnostics accumulators (Python floats to keep them off-device)
        alpha_sum = 0.0
        alpha_count = 0
        update_frob_sum = 0.0
        update_frob_count = 0
        cos_sum = 0.0
        cos_count = 0

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            alpha = float(group["alpha"])
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"alpha must be in [0, 1] but is: {alpha}")
            alpha_sum += alpha
            alpha_count += 1

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []
            self._init_group(group, params_with_grad, grads, momentum_bufs)

            for param, grad, buf in zip(params_with_grad, grads, momentum_bufs, strict=True):
                if grad.ndim != 2:
                    raise ValueError("Param gradient must be 2D")

                # Momentum buffer update (matches torch.optim.Muon exactly)
                buf.lerp_(grad, 1 - momentum)
                raw = grad.lerp(buf, momentum) if nesterov else buf

                m_hat = _normalize(raw, eps)

                if alpha == 1.0:
                    # Fast path: bit-identical to torch.optim.Muon.
                    update = _newton_schulz(m_hat, ns_coefficients, ns_steps)
                    # cos(M_hat, NS(M_hat)) is still informative here.
                    cos_sum += _cosine_sim(m_hat, update)
                    cos_count += 1
                elif alpha == 0.0:
                    # Skip NS entirely: update is √min(m,n) · M_hat.
                    scale = math.sqrt(min(m_hat.shape[0], m_hat.shape[1]))
                    update = m_hat * scale
                    # No NS output, so skip cosine diagnostic for this param.
                else:
                    ns_out = _newton_schulz(m_hat, ns_coefficients, ns_steps)
                    scale = math.sqrt(min(m_hat.shape[0], m_hat.shape[1]))
                    update = (1.0 - alpha) * scale * m_hat + alpha * ns_out
                    cos_sum += _cosine_sim(m_hat, ns_out)
                    cos_count += 1

                update_frob_sum += float(update.norm().item())
                update_frob_count += 1

                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
                param.mul_(1 - lr * weight_decay)
                param.add_(update, alpha=-adjusted_lr)

        self.last_alpha = alpha_sum / alpha_count if alpha_count else 0.0
        self.last_update_frob_mean = update_frob_sum / update_frob_count if update_frob_count else 0.0
        self.last_cos_mhat_ns = cos_sum / cos_count if cos_count else 0.0

        return loss


def _cosine_sim(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two matrices, treated as flat vectors."""
    af = a.flatten().float()
    bf = b.flatten().float()
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return 0.0
    return float((af @ bf / denom).item())
