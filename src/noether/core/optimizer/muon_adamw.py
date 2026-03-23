#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Muon Optimizer - MomentUm Orthogonalized by Newton-schulz

Paper: https://kellerjordan.github.io/posts/muon/
Original Implementation: https://github.com/KellerJordan/Muon

"""

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute zeroth power / orthogonalization of G using Newton-Schulz iteration.

    Uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    This produces an approximation to UV^T where USV^T = G is the SVD, but with S' diagonal
    having S_{ii}' ~ Uniform(0.5, 1.5), which empirically doesn't hurt model performance.

    Args:
        G: Input tensor of shape (..., m, n) where m and n are the last two dimensions
        steps: Number of Newton-Schulz iterations to perform (default: 5)

    Returns:
        Orthogonalized tensor of the same shape as G
    """
    assert G.ndim >= 2, "Input tensor must have at least 2 dimensions"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    result: torch.Tensor = X
    return result


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute Muon update for a gradient tensor.

    Args:
        grad: Gradient tensor
        momentum: Momentum buffer
        beta: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        nesterov: Whether to use Nesterov momentum (default: True)

    Returns:
        Update tensor to be applied to parameters (may be reshaped, caller should reshape back)
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # For 1D parameters (like biases), just return momentum update without orthogonalization
    if update.ndim < 2:
        return update

    # Reshape 4D conv filters to 2D
    if update.ndim == 4:
        update = update.view(len(update), -1)

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Compute Adam update for a gradient tensor.

    Args:
        grad: Gradient tensor
        exp_avg: Exponential moving average of gradients
        exp_avg_sq: Exponential moving average of squared gradients
        step: Current step number
        betas: Beta coefficients for Adam
        eps: Epsilon for numerical stability

    Returns:
        Update tensor to be applied to parameters
    """
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    exp_avg_corrected = exp_avg / (1 - betas[0] ** step)
    exp_avg_sq_corrected = exp_avg_sq / (1 - betas[1] ** step)
    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


class Muon(Optimizer):
    """Muon optimizer for distributed training.

    Muon uses momentum with orthogonalization via Newton-Schulz iteration.
    This variant is designed for distributed settings using torch.distributed.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.02)
        weight_decay: Weight decay coefficient (default: 0.0)
        momentum: Momentum coefficient (default: 0.95)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
    ):
        """Initialize the Muon optimizer."""
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)

        # Sort parameters by size for better distribution across devices
        if isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter):
            params = sorted(params, key=lambda x: x.size(), reverse=True)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            world_size = dist.get_world_size()
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)

            for base_i in range(len(params))[::world_size]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

                dist.all_gather(params_pad[base_i : base_i + world_size], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(Optimizer):
    """Muon optimizer for single-device (non-distributed) training.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.02)
        weight_decay: Weight decay coefficient (default: 0.0)
        momentum: Momentum coefficient (default: 0.95)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        **kwargs,
    ):
        """Initialize the SingleDeviceMuon optimizer."""
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class MuonWithAuxAdam(Optimizer):
    """Muon optimizer with auxiliary Adam for incompatible parameters (distributed).

    This optimizer runs Muon for high-dimensional (ndim >= 2) parameters and Adam for
    low-dimensional parameters (biases, gains, embeddings). Users must specify which
    parameters use Muon vs Adam via the 'use_muon' flag in param_groups.

    Example:
        >>> hidden_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
        >>> scalar_params = [p for p in model.parameters() if p.ndim < 2]
        >>> param_groups = [
        ...     dict(params=hidden_params, lr=0.05, momentum=0.95, use_muon=True),
        ...     dict(params=scalar_params, lr=3e-4, betas=(0.9, 0.95), use_muon=False),
        ... ]
        >>> optimizer = MuonWithAuxAdam(param_groups)

    Args:
        param_groups: List of parameter groups with 'use_muon' flag
    """

    def __init__(self, param_groups: list[dict], **kwargs):
        """Initialize the MuonWithAuxAdam optimizer."""
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each param_group must have 'use_muon' flag")

            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                expected = {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
                expected = {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}

            if not set(group.keys()).issubset(expected | {"name", "lr_scale"}):
                raise ValueError(f"Invalid keys in param_group: {set(group.keys()) - expected}")

        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                world_size = dist.get_world_size()
                params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)

                for base_i in range(len(params))[::world_size]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)

                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                    dist.all_gather(params_pad[base_i : base_i + world_size], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(Optimizer):
    """Muon optimizer with auxiliary Adam for incompatible parameters (single-device).

    Non-distributed variant of MuonWithAuxAdam. See MuonWithAuxAdam for usage details.

    Args:
        param_groups: List of parameter groups with 'use_muon' flag
    """

    def __init__(
        self,
        param_groups: list[dict],
        lr_muon: float | None = None,
        lr_adamw: float | None = None,
        momentum: float | None = None,
        weight_decay: float | None = None,
        **kwargs,
    ):
        """Initialize the SingleDeviceMuonWithAuxAdam optimizer."""
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each param_group must have 'use_muon' flag")

            if group["use_muon"]:
                group.setdefault("lr", lr_muon)
                group.setdefault("momentum", momentum)
                group.setdefault("weight_decay", weight_decay)
                expected = {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group.setdefault("lr", lr_adamw)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("momentum", momentum)
                group.setdefault("weight_decay", weight_decay)
                expected = {"params", "lr", "betas", "eps", "momentum", "weight_decay", "use_muon"}

            if not set(group.keys()).issubset(expected | {"name", "lr_scale"}):
                raise ValueError(f"Invalid keys in param_group: {set(group.keys()) - expected}")

        # Use the maximum base LR as default (should be Muon's LR)
        max_lr = max(g["lr"] for g in param_groups)
        default_weight_decay = param_groups[0]["weight_decay"] if param_groups else 0.0
        defaults = dict(lr=max_lr, weight_decay=default_weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


def create_muon_optimizer(
    params,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    adam_lr: float = 3e-4,
    adam_betas: tuple[float, float] = (0.8, 0.95),
    adam_weight_decay: float = 0,
    model: torch.nn.Module | None = None,
    **kwargs,
) -> SingleDeviceMuonWithAuxAdam:
    """Create Muon optimizer preserving weight decay groups while splitting by dimensionality.

    Creates up to 4 groups:
    1. High-dim (ndim>=2) WITH weight decay → Muon + weight_decay
    2. High-dim (ndim>=2) WITHOUT weight decay → Muon, no weight_decay
    3. Low-dim (ndim<2) WITH weight decay → Adam + weight_decay
    4. Low-dim (ndim<2) WITHOUT weight decay → Adam, no weight_decay

    Args:
        params: Model parameters or parameter groups
        muon_lr: Learning rate for Muon optimizer (default: 0.02)
        muon_momentum: Momentum value for Muon optimizer (default: 0.95)
        muon_weight_decay: Weight decay for Muon optimizer (default: 0.01)
        adam_lr: Learning rate for Adam optimizer (default: 3e-4)
        adam_betas: Beta coefficients for Adam optimizer (default: (0.9, 0.95))
        adam_weight_decay: Weight decay for Adam optimizer (default: 0.01)
        model: Optional model reference
        **kwargs: Additional arguments (ignored)

    Returns:
        SingleDeviceMuonWithAuxAdam optimizer instance

    Raises:
        AssertionError: If muon_lr < adam_lr (required for proper LR scheduling)
    """
    # Assert that Muon LR is greater than or equal to Adam LR for proper scheduling
    assert muon_lr >= adam_lr, f"muon_lr ({muon_lr}) must be >= adam_lr ({adam_lr}) for proper LR scheduling"

    # Separate by original weight decay setting AND dimensionality
    high_dim_with_wd = []
    high_dim_no_wd = []
    low_dim_with_wd = []
    low_dim_no_wd = []

    if isinstance(params, list):
        for group in params:
            if isinstance(group, dict) and "params" in group:
                # Check if this group has weight decay
                has_wd = "weight_decay" in group and group["weight_decay"] > 0

                # Separate by dimensionality
                for p in group["params"]:
                    if p.ndim >= 2:
                        if has_wd:
                            high_dim_with_wd.append(p)
                        else:
                            high_dim_no_wd.append(p)
                    else:
                        if has_wd:
                            low_dim_with_wd.append(p)
                        else:
                            low_dim_no_wd.append(p)
            else:
                # Fallback: treat as params without weight decay
                for p in group if isinstance(group, list | tuple) else [group]:
                    if p.ndim >= 2:
                        high_dim_no_wd.append(p)
                    else:
                        low_dim_no_wd.append(p)

    print("=" * 30)
    print(f"[Muon] High-dim WITH wd: {len(high_dim_with_wd)}")
    print(f"[Muon] High-dim NO wd: {len(high_dim_no_wd)}")
    print(f"[Muon] Low-dim WITH wd: {len(low_dim_with_wd)}")
    print(f"[Muon] Low-dim NO wd: {len(low_dim_no_wd)}")
    print("=" * 30)

    # Calculate lr_scale for Adam groups (relative to Muon)
    lr_scale_adam = adam_lr / muon_lr

    # Create parameter groups (skip empty ones)
    param_groups = []

    if high_dim_with_wd:
        param_groups.append(
            dict(
                params=high_dim_with_wd,
                use_muon=True,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                lr_scale=1.0,
                name="muon_wd",
            )
        )

    if high_dim_no_wd:
        param_groups.append(
            dict(
                params=high_dim_no_wd,
                use_muon=True,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=0.0,
                lr_scale=1.0,
                name="muon",
            )
        )

    if low_dim_with_wd:
        param_groups.append(
            dict(
                params=low_dim_with_wd,
                use_muon=False,
                lr=adam_lr,
                betas=adam_betas,
                weight_decay=adam_weight_decay,
                lr_scale=lr_scale_adam,
                name="adam_wd",
            )
        )

    if low_dim_no_wd:
        param_groups.append(
            dict(
                params=low_dim_no_wd,
                use_muon=False,
                lr=adam_lr,
                betas=adam_betas,
                weight_decay=0.0,
                lr_scale=lr_scale_adam,
                name="adam",
            )
        )

    return SingleDeviceMuonWithAuxAdam(param_groups)
