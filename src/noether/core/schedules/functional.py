#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math


def linear(step: int, total_steps: int) -> float:
    """linearly increasing from [0 to 1]"""
    return step / max(1, total_steps - 1)


def cosine(step: int, total_steps: int) -> float:
    """cosine schedule from [0 to 1]"""
    progress = step / max(1, total_steps - 1)
    return 1 - (1 + math.cos(math.pi * progress)) / 2


def polynomial(step: int, total_steps: int, power: float) -> float:
    """polynomial schedule from [0 to 1]
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html"""
    progress = step / max(1, total_steps - 1)
    return float(1 - (1 - progress) ** power)


def sqrt_decay(step: int, total_steps: int) -> float:
    """sqrt schedule from [0 to 1]; rapid initial change, slow tail.

    Used as the cooldown phase of the WSD schedule (Hägele et al. 2024)."""
    progress = step / max(1, total_steps - 1)
    return math.sqrt(progress)
