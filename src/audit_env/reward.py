"""Terminal reward function (spec section 05).

    reward = clip(2 * F1 - 1 - 0.002 * inspection_steps, -1, 1)

Published, shared by every caller: the environment, the tests and the official check.
"""

from __future__ import annotations

STEP_PENALTY = 0.002
REWARD_MIN = -1.0
REWARD_MAX = 1.0


def terminal_reward(f1: float, inspection_steps: int, step_penalty: float = STEP_PENALTY) -> float:
    """Reward for a finished episode. During investigation the reward is always 0."""
    if inspection_steps < 0:
        raise ValueError("inspection_steps must be >= 0")
    raw = 2.0 * float(f1) - 1.0 - step_penalty * int(inspection_steps)
    return float(max(REWARD_MIN, min(REWARD_MAX, raw)))
