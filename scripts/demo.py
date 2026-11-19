#!/usr/bin/env python3
"""Demo: one scripted policy, one episode, printed diagnostics.

    python scripts/demo.py [--seed N] [--clean] [--policy reconciliation|oracle|random|mark_all]

The scripted reconciliation policy is enough for the mandatory demo (spec 07);
no model, no GPU, no network.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_env import AuditEnv  # noqa: E402
from audit_env.policies import (  # noqa: E402
    MarkAllPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    ReconciliationPolicy,
    run_episode,
)

SMALL = {"n_invoices": 40}


def build_policy(name: str, env: AuditEnv):
    if name == "reconciliation":
        return ReconciliationPolicy()
    if name == "oracle":
        return OraclePolicy(env.evaluator_truth())
    if name == "random":
        return RandomExplorerPolicy(seed=7)
    if name == "mark_all":
        return MarkAllPolicy()
    raise SystemExit(f"unknown policy: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1911)
    ap.add_argument("--policy", default="reconciliation",
                    choices=["reconciliation", "oracle", "random", "mark_all"])
    ap.add_argument("--clean", action="store_true", help="чистая компания, ошибок нет")
    ap.add_argument("--issues", default=None,
                    help="типы ошибок через запятую (по умолчанию: 2 базовых)")
    ap.add_argument("--n-issues", type=int, default=None, help="сколько ошибок вшить")
    ap.add_argument("--mode", default="train", choices=["train", "eval"])
    args = ap.parse_args()

    config = dict(SMALL)
    if args.clean:
        config.update({"clean": True, "n_issues": 0})
    if args.issues:
        config["issue_types"] = [t.strip() for t in args.issues.split(",") if t.strip()]
    if args.n_issues is not None:
        config["n_issues"] = args.n_issues

    env = AuditEnv(mode=args.mode)
    # policy is built AFTER reset for the oracle, so reset first via a dry run
    obs, _ = env.reset(seed=args.seed, config=config)
    policy = build_policy(args.policy, env)

    started = time.monotonic()
    result = run_episode(env, policy, seed=args.seed, config=config)
    elapsed = time.monotonic() - started

    report = result["report"]
    metrics = report["metrics"] or {}

    print(f"episode      : {result['episode_id']}")
    print(f"policy       : {args.policy}  (mode={args.mode})")
    print(f"steps        : {result['steps']} / {env.max_steps}")
    print(f"time         : {elapsed:.2f}s (limit {env.time_limit_s:.0f}s)")
    print(f"truth issues : {len(env.evaluator_truth())}")
    print(f"metrics      : P={metrics.get('precision', 0):.3f} "
          f"R={metrics.get('recall', 0):.3f} F1={metrics.get('f1', 0):.3f}")
    print(f"counts       : TP={metrics.get('tp')} FP={metrics.get('fp')} FN={metrics.get('fn')}")
    print(f"reward(true) : {report['true_reward']:+.4f}")
    print(f"log          : {report['log']}")


if __name__ == "__main__":
    main()
