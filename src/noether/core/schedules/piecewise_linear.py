#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Piecewise-linear schedule defined by a list of (fraction, value) control points.

Useful for non-monotonic shapes such as a U-curve, which none of the monotonic
``ProgressSchedule`` subclasses can express directly.
"""

from noether.core.schedules.base import ScheduleBase
from noether.core.schemas.schedules import PiecewiseLinearScheduleConfig


class PiecewiseLinearSchedule(ScheduleBase):
    """A scheduler that linearly interpolates between user-defined control points.

    Control points are given as ``(fraction, value)`` pairs, where ``fraction ∈ [0, 1]``
    is the fraction of ``total_steps``. Fractions must be non-decreasing and must cover
    ``0.0`` and ``1.0``. Duplicate fractions encode step-function jumps, with the later
    value winning at the tie point (right-continuous).

    Example:

        .. code-block:: yaml

            schedule_config:
                kind: noether.core.schedules.PiecewiseLinearSchedule
                control_points:
                    - [0.0, 1.0]
                    - [0.3, 0.3]
                    - [0.7, 0.3]
                    - [1.0, 1.0]
    """

    def __init__(self, config: PiecewiseLinearScheduleConfig):
        super().__init__(overhang_percent=config.overhang_percent, overhang_steps=config.overhang_steps)
        self.control_points: list[tuple[float, float]] = [(float(f), float(v)) for f, v in config.control_points]

    def _get_value(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return self.control_points[0][1]
        frac = step / total_steps
        pts = self.control_points
        if frac <= pts[0][0]:
            return pts[0][1]
        if frac >= pts[-1][0]:
            return pts[-1][1]
        # Largest i such that pts[i][0] <= frac. Scanning from the right makes duplicate
        # fractions (step jumps) right-continuous: at a tie, the later pair wins.
        i = len(pts) - 1
        while pts[i][0] > frac:
            i -= 1
        f0, v0 = pts[i]
        f1, v1 = pts[i + 1]
        if f1 == f0:
            return v0
        t = (frac - f0) / (f1 - f0)
        return v0 + t * (v1 - v0)

    def __str__(self):
        return f"{type(self).__name__}(points={self.control_points})"
