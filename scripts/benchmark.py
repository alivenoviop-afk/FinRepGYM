#!/usr/bin/env python3
"""Бенчмарк: ИЗМЕРЕННЫЕ запасы до лимитов регламента.

    python scripts/benchmark.py

Спека 07: 60 секунд на эпизод, 100 шагов, 2 CPU-ядра, 4 ГБ RAM, 1 ГБ на решение,
Python 3.11, без сети и GPU. Скрипт меряет реальное время генерации и прогона,
пик памяти и размер решения и печатает запас по каждому лимиту.

Результат: таблица в stdout + artifacts/benchmark.csv.
"""

from __future__ import annotations

import csv
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_env import AuditEnv, generate  # noqa: E402
from audit_env.policies import (  # noqa: E402
    OraclePolicy,
    ReconciliationPolicy,
    run_episode,
)
from jury_marathon import memory_mb, pin_to_two_cores  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
SIZES = [50, 100, 150, 200]          # n_invoices: от малого до запаса
REPS = 3                              # минимум из трёх замеров (стабильнее)
FOUR_TYPES = ["duplicate_entry", "period_cutoff", "partial_allocation", "fx_rate"]

# Лимиты регламента (спека 07)
EPISODE_LIMIT_S = 60.0
STEP_LIMIT = 100
RAM_LIMIT_MB = 4096.0
SOLUTION_LIMIT_MB = 1024.0


def best_of(reps: int, fn):
    """(минимум, среднее, последний результат) — минимум устойчивее к шуму ОС."""
    times: list[float] = []
    out = None
    for _ in range(reps):
        started = time.monotonic()
        out = fn()
        times.append(time.monotonic() - started)
    return min(times), sum(times) / len(times), out


def run_policy(name: str, n_invoices: int, seed: int) -> dict:
    cfg = {"n_invoices": n_invoices, "issue_types": list(FOUR_TYPES), "n_issues": 4}
    env = AuditEnv()
    env.reset(seed=seed, config=cfg)
    policy = (OraclePolicy(env.evaluator_truth()) if name == "oracle"
              else ReconciliationPolicy())
    return run_episode(env, policy, seed=seed, config=cfg)


def solution_size_mb() -> float:
    total = 0
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            total += path.stat().st_size
    return total / 2 ** 20


def main() -> int:
    pin = pin_to_two_cores()
    rows: list[dict] = []
    worst_episode = 0.0
    worst_steps = 0

    print("=" * 78)
    print("  FinRepGYM — бенчмарк запасов до лимитов регламента")
    print(f"  платформа : {platform.system()} {platform.release()}, "
          f"Python {platform.python_version()}")
    print(f"  ядра      : {pin} (лимит регламента: 2)")
    print("=" * 78)
    print()
    print("| n_invoices | проводок | генерация, с | сверка, с | оракул, с | шаги | запас до 60 с |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    for n in SIZES:
        gen_best, gen_avg, files = best_of(
            REPS, lambda n=n: generate(20260000 + n, {"n_invoices": n}))
        rows_ledger = files["open"]["ledger.csv"].strip().count("\n")

        rec_best, rec_avg, rec = best_of(
            REPS, lambda n=n: run_policy("reconciliation", n, 31000 + n))
        orc_best, orc_avg, orc = best_of(
            REPS, lambda n=n: run_policy("oracle", n, 31000 + n))

        episode_worst = max(rec_avg, orc_avg)
        steps_worst = max(rec["steps"], orc["steps"])
        worst_episode = max(worst_episode, episode_worst)
        worst_steps = max(worst_steps, steps_worst)
        margin = EPISODE_LIMIT_S / episode_worst if episode_worst else float("inf")

        rows.append({
            "n_invoices": n,
            "ledger_rows": rows_ledger,
            "generate_best_s": round(gen_best, 4),
            "generate_avg_s": round(gen_avg, 4),
            "reconciliation_best_s": round(rec_best, 4),
            "reconciliation_avg_s": round(rec_avg, 4),
            "oracle_best_s": round(orc_best, 4),
            "oracle_avg_s": round(orc_avg, 4),
            "steps_max": steps_worst,
            "oracle_f1": round(orc["report"]["metrics"]["f1"], 4),
            "recon_f1": round(rec["report"]["metrics"]["f1"], 4),
            "margin_to_60s": round(margin, 1),
        })
        print(f"| {n} | {rows_ledger} | {gen_best:.3f} | {rec_best:.3f} | "
              f"{orc_best:.3f} | {steps_worst} | x{margin:.0f} |")

    peak_mb, current_mb = memory_mb()
    size_mb = solution_size_mb()

    print()
    print("Запасы до лимитов регламента (измерено, не заявлено):")
    print("-" * 78)
    print(f"  время на эпизод   : {worst_episode:.3f} с из {EPISODE_LIMIT_S:.0f} с"
          f"   -> запас x{EPISODE_LIMIT_S / worst_episode:.0f}")
    print(f"  шаги на эпизод    : {worst_steps} из {STEP_LIMIT}"
          f"          -> запас x{STEP_LIMIT / max(worst_steps, 1):.1f}")
    if peak_mb:
        print(f"  пик памяти        : {peak_mb:.0f} МБ из {RAM_LIMIT_MB:.0f} МБ"
              f"   -> запас x{RAM_LIMIT_MB / peak_mb:.0f}")
    else:
        print("  пик памяти        : недоступен на этой платформе")
    if current_mb:
        print(f"  текущая память    : {current_mb:.0f} МБ")
    print(f"  размер решения    : {size_mb:.0f} МБ из {SOLUTION_LIMIT_MB:.0f} МБ"
          f"   -> запас x{SOLUTION_LIMIT_MB / size_mb:.0f}")
    print(f"  сеть / GPU        : не используются (проверка `python -S` в CI)")
    print("-" * 78)

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / "benchmark.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Пороги: бенчмарк обязан ПАДАТЬ, если лимиты регламента нарушены
    failures = []
    if worst_episode >= EPISODE_LIMIT_S:
        failures.append(f"эпизод {worst_episode:.2f} с >= лимита {EPISODE_LIMIT_S}")
    if worst_steps > STEP_LIMIT:
        failures.append(f"шаги {worst_steps} > лимита {STEP_LIMIT}")
    if peak_mb and peak_mb >= RAM_LIMIT_MB:
        failures.append(f"память {peak_mb:.0f} МБ >= лимита {RAM_LIMIT_MB:.0f} МБ")
    if size_mb >= SOLUTION_LIMIT_MB:
        failures.append(f"размер {size_mb:.0f} МБ >= лимита {SOLUTION_LIMIT_MB:.0f} МБ")
    for row in rows:
        if row["oracle_f1"] < 1.0:
            failures.append(f"оракул не достиг F1=1 на n_invoices={row['n_invoices']}")
        if row["recon_f1"] < 1.0:
            failures.append(f"сверка не достигла F1=1 на n_invoices={row['n_invoices']}")

    print(f"artifacts/benchmark.csv : {len(rows)} строк")
    if failures:
        for item in failures:
            print(f"  [X] {item}")
        return 1
    print("  [OK] все лимиты регламента соблюдены с запасом; F1=1 у оракула и сверки")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
