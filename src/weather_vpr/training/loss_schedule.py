"""Update-based weights for optional auxiliary descriptor regularization."""
from __future__ import annotations

import math


def auxiliary_weight_multiplier(schedule: str, optimizer_step: int, total_steps: int,
                                final_ratio: float = 0.0) -> float:
    """Use one weight throughout an accumulation window; decay on updates only."""
    if schedule not in {"constant", "linear"}:
        raise ValueError("Auxiliary schedule must be constant or linear")
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if type(optimizer_step) is not int or not 0 <= optimizer_step < total_steps:
        raise ValueError("optimizer_step must be in [0, total_steps)")
    if (isinstance(final_ratio, bool) or not isinstance(final_ratio, (int, float))
            or not math.isfinite(final_ratio) or not 0 <= final_ratio <= 1):
        raise ValueError("final_ratio must be finite in [0,1]")
    if schedule == "constant" or total_steps == 1:
        return 1.0
    return 1.0 - (1.0 - final_ratio) * optimizer_step / (total_steps - 1)
