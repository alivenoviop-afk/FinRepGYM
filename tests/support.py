"""Shared test support: path setup, small configs, helper policies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_env import (  # noqa: E402,F401
    AuditEnv,
    ContractViolation,
    compute_metrics,
    generate,
    match,
    terminal_reward,
)
from audit_env.policies import (  # noqa: E402,F401
    MarkAllPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    ReconciliationPolicy,
    run_episode,
)

SEED = 1911
SMALL = {"n_invoices": 12}
CLEAN = {"n_invoices": 12, "clean": True, "n_issues": 0}


class SilencePolicy:
    """Submits nothing but an empty findings list (правильный ответ на чистых книгах)."""

    def __init__(self) -> None:
        self.manifest = None
        self.sent = False

    def __call__(self, obs):
        if isinstance(obs.get("manifest"), dict):
            self.manifest = obs["manifest"]
        if self.sent:
            return {"action": "calculator", "expression": "1"}
        self.sent = True
        return {"action": "submit", "episode_id": self.manifest["episode_id"], "findings": []}


def run_with(policy_factory, seed: int = SEED, config: dict | None = None,
             mode: str = "train", env: AuditEnv | None = None) -> dict:
    """Reset, hand the truth to the policy factory, run the episode."""
    env = env or AuditEnv(mode=mode)
    env.reset(seed=seed, config=dict(config or SMALL))
    policy = policy_factory(env)
    return run_episode(env, policy, seed=seed, config=dict(config or SMALL))


def oracle_run(seed: int = SEED, config: dict | None = None, mode: str = "train") -> dict:
    return run_with(lambda env: OraclePolicy(env.evaluator_truth()), seed, config, mode)
