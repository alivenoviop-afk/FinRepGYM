#!/usr/bin/env python3
"""Репетиция ЗАКРЫТОЙ проверки жюри (спека, раздел 06).

    python scripts/organizer_check.py
    python scripts/organizer_check.py --seeds 11,12,13,...          # 20 своих seed
    python scripts/organizer_check.py --seeds-file seeds.txt        # по числу в строке

Что делает (ровно протокол регламента):
  1. Собирает 20 эпизодов: 5 чистых + 5 одиночных + 5 с двумя типами
     + 5 с взаимодействующими ошибками;
  2. Гоняет на ОДНИХ И ТЕХ ЖЕ эпизодах четыре политики организаторов:
     случайный исследователь, простой алгоритм сверки, атака «пометить всё», оракул;
  3. Проверяет: оракул достигает максимума, сверка без FP/FN, каждый прогон
     укладывается в 60 секунд, в наблюдениях нет закрытого эталона;
  4. Печатает сводку и пишет artifacts/organizer_report.md + organizer_runs.csv.

Свои seed жюри подставляются аргументами --seeds / --seeds-file:
каждый seed применяется к своему месту в фиксированном составе 5/5/5/5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_env import AuditEnv  # noqa: E402
from audit_env.policies import (  # noqa: E402
    MarkAllPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    ReconciliationPolicy,
    run_episode,
)

ARTIFACTS = ROOT / "artifacts"
REPORT_MD = ARTIFACTS / "organizer_report.md"
REPORT_CSV = ARTIFACTS / "organizer_runs.csv"
EPISODE_TIME_LIMIT_S = 60.0

DUP = "duplicate_entry"
CUT = "period_cutoff"
PART = "partial_allocation"
FX = "fx_rate"

FORBIDDEN_IN_OBSERVATION = ["iss_", "issue_id", "expected_amount_minor", "provenance",
                            "causal_change", "invariants_", "truth.json", "seed",
                            "statement"]


def clean(n: int, **extra) -> dict:
    return {"n_invoices": n, "clean": True, "n_issues": 0, "scenarios": [], **extra}


def single(n: int, kind: str, **extra) -> dict:
    return {"n_invoices": n, "issue_types": [kind], "n_issues": 1, **extra}


def pair(n: int, a: str, b: str, **extra) -> dict:
    return {"n_invoices": n, "issue_types": [a, b], "n_issues": 2, **extra}


def scen(n: int, names: list[str], issues: list[str] | None = None, **extra) -> dict:
    return {"n_invoices": n, "issue_types": issues or [], "n_issues": len(issues or []),
            "scenarios": names, **extra}


# Фиксированный состав: 5 чистых / 5 одиночных / 5 парных / 5 взаимодействий
COMPOSITION: list[tuple[str, dict]] = [
    # -- 5 чистых ------------------------------------------------------- #
    ("clean", clean(60)),
    ("clean", clean(70, utc_offset="+03:00")),
    ("clean", clean(80)),
    ("clean", clean(55, utc_offset="-07:00")),
    ("clean", clean(65)),
    # -- 5 одиночных (все 4 типа + один повтор) -------------------------- #
    ("single", single(60, DUP)),
    ("single", single(65, PART)),
    ("single", single(70, FX)),
    ("single", single(75, CUT)),
    ("single", single(55, DUP)),
    # -- 5 с двумя типами ------------------------------------------------ #
    ("pair", pair(60, DUP, CUT)),
    ("pair", pair(65, PART, FX)),
    ("pair", pair(70, DUP, PART)),
    ("pair", pair(60, FX, CUT)),
    ("pair", pair(65, DUP, FX)),
    # -- 5 с взаимодействующими ошибками --------------------------------- #
    ("interacting", scen(70, ["S1"])),
    ("interacting", scen(75, ["S2"])),
    ("interacting", scen(60, ["S1", "S2"])),
    ("interacting", scen(65, ["S1"], issues=[DUP])),
    ("interacting", scen(70, ["S2"], issues=[FX])),
]

DEFAULT_SEEDS = [3101 + i for i in range(len(COMPOSITION))]

POLICIES = [
    ("reconciliation", lambda env: ReconciliationPolicy()),
    ("oracle", lambda env: OraclePolicy(env.evaluator_truth())),
    ("mark_all", lambda env: MarkAllPolicy()),
    ("random", lambda env: RandomExplorerPolicy(seed=7)),
]


def run_one(seed: int, cfg: dict, factory) -> tuple[dict, float, list[str]]:
    env = AuditEnv(mode="train")
    env.reset(seed=seed, config=cfg)
    started = time.monotonic()
    result = run_episode(env, factory(env), seed=seed, config=cfg)
    elapsed = time.monotonic() - started

    leaks = []
    for rec in result["trajectory"]:
        blob = json.dumps(rec["observation"], ensure_ascii=False)
        leaks.extend(t for t in FORBIDDEN_IN_OBSERVATION if t in blob)
    return result, elapsed, sorted(set(leaks))


def parse_seeds(args) -> list[int]:
    if args.seeds_file:
        seeds = [int(line.strip()) for line in Path(args.seeds_file).read_text().splitlines()
                 if line.strip() and not line.strip().startswith("#")]
    elif args.seeds:
        seeds = [int(s) for s in args.seeds.replace(" ", "").split(",") if s]
    else:
        seeds = list(DEFAULT_SEEDS)
    if len(seeds) < len(COMPOSITION):
        base = list(seeds) or [1]
        seeds = [base[i % len(base)] * 1000 + i for i in range(len(COMPOSITION))]
        print(f"  ! seed'ов меньше 20 — дополнены детерминированно: {seeds[:4]}...")
    return seeds[: len(COMPOSITION)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", help="20 seed через запятую")
    ap.add_argument("--seeds-file", help="файл: одно число в строке")
    args = ap.parse_args()

    started = time.monotonic()
    seeds = parse_seeds(args)
    ARTIFACTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("  ЗАКРЫТАЯ ПРОВЕРКА (протокол раздела 06): 20 эпизодов x 4 политики")
    print("=" * 72)

    rows: list[dict] = []
    group_counter: dict[str, int] = {"clean": 0, "single": 0, "pair": 0, "interacting": 0}
    problems: list[str] = []
    all_leaks: list[str] = []
    max_episode_time = 0.0

    for (group, cfg), seed in zip(COMPOSITION, seeds):
        group_counter[group] += 1
        label = (f"{group} {group_counter[group]}/5")

        env = AuditEnv(mode="train")
        env.reset(seed=seed, config=cfg)
        n_issues = len(env.evaluator_truth())
        env_types = sorted({i["type"] for i in env.evaluator_truth()})

        parts = []
        for policy_name, factory in POLICIES:
            result, elapsed, leaks = run_one(seed, cfg, factory)
            max_episode_time = max(max_episode_time, elapsed)
            all_leaks.extend(leaks)
            m = result["report"]["metrics"] or {"precision": 0, "recall": 0, "f1": 0,
                                                "tp": 0, "fp": 0, "fn": 0}
            rows.append({
                "group": group, "seed": seed, "policy": policy_name,
                "n_issues": n_issues, "issue_types": ",".join(env_types),
                "precision": round(m["precision"], 6), "recall": round(m["recall"], 6),
                "f1": round(m["f1"], 6),
                "return": round(result["report"]["true_reward"], 6),
                "steps": result["steps"], "elapsed_s": round(elapsed, 4),
            })
            parts.append(f"{policy_name}: F1={m['f1']:.3f} {result['report']['true_reward']:+.3f}")

            if policy_name == "oracle" and m["f1"] != 1.0:
                problems.append(f"seed={seed} [{label}]: оракул F1={m['f1']} != 1.0")
            if policy_name == "reconciliation" and (m["fp"] or m["fn"]):
                problems.append(f"seed={seed} [{label}]: сверка FP={m['fp']} FN={m['fn']}")
            if elapsed >= EPISODE_TIME_LIMIT_S:
                problems.append(f"seed={seed} [{label}] {policy_name}: {elapsed:.1f}s >= 60s")

        print(f"  [{label:14s}] seed={seed} issues={n_issues} "
              f"({','.join(env_types) or 'clean'})")
        print(f"      " + " | ".join(parts))

    # ---- проверки ------------------------------------------------------- #
    composition_ok = all(v == 5 for v in group_counter.values())
    leak_ok = not all_leaks
    checks = [
        ("состав 5/5/5/5 (чистые, одиночные, парные, взаимодействия)", composition_ok,
         str(group_counter)),
        ("оракул достигает максимума на всех эпизодах",
         not any("оракул" in p for p in problems), "F1 = 1.0"),
        ("сверка без ложных и без пропущенных на всех эпизодах",
         not any("сверка" in p for p in problems), "FP = 0, FN = 0"),
        ("каждый прогон укладывается в 60 секунд",
         not any("60s" in p for p in problems), f"max = {max_episode_time:.3f} с"),
        ("в наблюдениях нет закрытого эталона", leak_ok,
         "чисто" if leak_ok else f"утечки: {sorted(set(all_leaks))[:3]}"),
    ]

    # ---- агрегат по политикам ------------------------------------------- #
    def agg(policy: str) -> dict:
        sel = [r for r in rows if r["policy"] == policy]
        n = len(sel)
        return {
            "precision": sum(r["precision"] for r in sel) / n,
            "recall": sum(r["recall"] for r in sel) / n,
            "f1": sum(r["f1"] for r in sel) / n,
            "ret": sum(r["return"] for r in sel) / n,
            "steps": sum(r["steps"] for r in sel) / n,
            "time": sum(r["elapsed_s"] for r in sel) / n,
        }

    total = time.monotonic() - started
    passed = sum(1 for _, ok, _ in checks if ok)

    # ---- отчёты ---------------------------------------------------------- #
    md = [
        "# Отчёт «закрытой проверки» (репетиция протокола раздела 06)",
        "",
        f"- Эпизодов: **20** (5 чистых / 5 одиночных / 5 парных / 5 взаимодействий)",
        f"- Политики: **4** на одних и тех же эпизодах (всего прогонов: {len(rows)})",
        f"- Seed'ы: `{', '.join(str(s) for s in seeds)}` (свои — аргументы `--seeds` / `--seeds-file`)",
        f"- Проверок: **{passed}/{len(checks)}**, время: {total:.1f} с, "
        f"самый долгий прогон: {max_episode_time:.3f} с (лимит 60 с)",
        "",
        "## Сводка по политикам (среднее по 20 эпизодам)",
        "",
        "| Политика | P | R | F1 | Возврат | Шаги | Время, с |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("reconciliation", "oracle", "mark_all", "random"):
        a = agg(name)
        md.append(f"| {name} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | "
                  f"{a['ret']:+.3f} | {a['steps']:.1f} | {a['time']:.3f} |")
    md += ["", "## Проверки", ""]
    for name, ok, detail in checks:
        md.append(f"- {'[PASS]' if ok else '[FAIL]'} {name} — {detail}")
    md += ["", "Сырые значения каждого прогона: `artifacts/organizer_runs.csv`.", ""]
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 72)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    print(f"\n  ИТОГ: {passed}/{len(checks)} групп проверок, {len(rows)} прогонов, "
          f"{total:.1f} с")
    print(f"  Отчет: {REPORT_MD.relative_to(ROOT)} и {REPORT_CSV.relative_to(ROOT)}")
    print("=" * 72)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
