#!/usr/bin/env python3
"""Единая команда проверки: python scripts/check.py

Прогоняет всё, что возможно проверить локально без сети:

  1. Юнит-тесты (контракт, данные, награда, атаки, воспроизводимость).
  2. Свип по seed'ам и конфигурациям: чистые / одиночные / парные / взаимодействия.
  3. Оракул: максимальная награда достижима на каждом эпизоде.
  4. Сверка: 0 ложных и 0 пропущенных на всех эпизодах свипа.
  5. Атаки: mark-all и случайная политика не получают выгоды.
  6. Детерминизм: повтор seed+config даёт байт-в-байт те же файлы.
  7. Утечки: в траекториях нет закрытого эталона.
  8. Размеры и форматы: эпизод в границах спеки.

Выход: таблица PASS/FAIL и итог; код выхода 0 только при полном успехе.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from audit_env import AuditEnv, generate  # noqa: E402
from audit_env.policies import (  # noqa: E402
    MarkAllPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    ReconciliationPolicy,
    run_episode,
)

FORBIDDEN = ["iss_", "issue_id", "expected_amount_minor", "provenance", "causal_change",
             "invariants_", "truth.json", "seed", "statement"]

RESULTS: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((bool(ok), f"{name}{' — ' + detail if detail else ''}"))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def run_policy(env: AuditEnv, factory, seed: int, config: dict):
    env.reset(seed=seed, config=config)
    policy = factory(env)
    return run_episode(env, policy, seed=seed, config=config)


def sweep_config(name: str, config: dict, seeds: list[int]) -> None:
    """Свип: сверка обязана найти всё без FP, оракул обязан получить максимум."""
    env = AuditEnv()
    fp_total = fn_total = 0
    oracle_rewards, recon_rewards = [], []
    errors = []
    for seed in seeds:
        try:
            recon = run_policy(env, lambda e: ReconciliationPolicy(), seed, config)
            oracle = run_policy(env, lambda e: OraclePolicy(e.evaluator_truth()), seed, config)
            rm, om = recon["report"]["metrics"], oracle["report"]["metrics"]
            if rm is None or om is None:
                errors.append(f"seed={seed}: metrics is None")
                continue
            fp_total += rm["fp"]
            fn_total += rm["fn"]
            recon_rewards.append(recon["report"]["true_reward"])
            oracle_rewards.append(oracle["report"]["true_reward"])
            if om["f1"] != 1.0:
                errors.append(f"seed={seed}: oracle F1={om['f1']}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"seed={seed}: {type(exc).__name__}: {exc}")

    check(f"{name}: ошибок нет (сверка FP/FN)", fp_total == 0 and fn_total == 0 and not errors,
          f"FP={fp_total} FN={fn_total}" + (f" ошибки={errors[:2]}" if errors else ""))
    if oracle_rewards:
        check(f"{name}: оракул достигает максимума", all(r > 0.9 for r in oracle_rewards),
              f"min={min(oracle_rewards):+.3f} max={max(oracle_rewards):+.3f}")
    if recon_rewards:
        check(f"{name}: сверка собирает все находки", all(r > 0.9 for r in recon_rewards),
              f"min={min(recon_rewards):+.3f}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.monotonic()

    # 1. юнит-тесты -------------------------------------------------------- #
    print("\n1) Юнит-тесты")
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = [l for l in (proc.stdout + proc.stderr).splitlines() if l.strip().startswith(("Ran ", "OK", "FAILED"))]
    check("unittest", proc.returncode == 0, " ".join(tail[-2:]))

    # 2. свипы -------------------------------------------------------------- #
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1911]
    print("\n2) Свип по эпизодам (13 seed'ов x конфигурации)")
    sweep_config("чистые (clean)", {"n_invoices": 12, "clean": True, "n_issues": 0}, seeds)
    sweep_config("одиночные (1 тип)", {"n_invoices": 12, "n_issues": 1}, seeds)
    sweep_config("парные (2 типа)", {"n_invoices": 12,
                                     "issue_types": ["duplicate_entry", "period_cutoff"],
                                     "n_issues": 2}, seeds)
    sweep_config("три типа в одном эпизоде (S1/S2 — 20.11)",
                 {"n_invoices": 20,
                  "issue_types": ["duplicate_entry", "period_cutoff", "partial_allocation"],
                  "n_issues": 3}, seeds)

    # 3. атаки -------------------------------------------------------------- #
    print("\n3) Атакующие политики")
    env = AuditEnv()
    attack_ok = clean_ok = True
    for seed in seeds:
        bad = run_policy(env, lambda e: MarkAllPolicy(), seed,
                         {"n_invoices": 12, "n_issues": 2})
        if bad["report"]["metrics"]["f1"] != 0.0 or bad["report"]["true_reward"] != -1.0:
            attack_ok = False
        good = run_policy(env, lambda e: ReconciliationPolicy(), seed,
                          {"n_invoices": 12, "clean": True, "n_issues": 0})
        if good["report"]["metrics"]["fp"] != 0 or good["report"]["metrics"]["f1"] != 1.0:
            clean_ok = False
    check("mark-all на всех эпизодах получает F1=0 и -1", attack_ok)
    check("тишина/сверка на чистых: 0 ложных, F1=1", clean_ok)

    rnd = run_policy(env, lambda e: RandomExplorerPolicy(seed=7), 1911, {"n_invoices": 12})
    check("угадывание id не даёт TP", rnd["report"]["metrics"]["tp"] == 0)

    # 4. детерминизм -------------------------------------------------------- #
    print("\n4) Воспроизводимость")
    cfg = {"n_invoices": 12, "n_issues": 3,
           "issue_types": ["duplicate_entry", "period_cutoff", "partial_allocation"]}
    a, b = generate(42, cfg), generate(42, cfg)
    check("повтор seed+config -> байт-в-байт те же файлы", a["open"] == b["open"] and a["closed"] == b["closed"])
    r1 = run_policy(env, lambda e: ReconciliationPolicy(), 42, cfg)
    r2 = run_policy(env, lambda e: ReconciliationPolicy(), 42, cfg)
    check("повтор эпизода -> идентичная траектория и награда",
          json.dumps(r1["trajectory"], sort_keys=True) == json.dumps(r2["trajectory"], sort_keys=True))

    # 5. утечки -------------------------------------------------------------- #
    print("\n5) Утечки закрытого эталона")
    blob = json.dumps(r1["trajectory"], ensure_ascii=False)
    leaked = [t for t in FORBIDDEN if t in blob]
    check("нет запрещённых маркеров в траектории", not leaked, f"найдено: {leaked}" if leaked else "чисто")

    # 6. размеры и форматы --------------------------------------------------- #
    print("\n6) Рамки спеки")
    big = generate(7, {"n_invoices": 80})
    man = json.loads(big["open"]["manifest.json"])
    check("200 <= проводок <= 2000", 200 <= man["n_records"] <= 2000, f"{man['n_records']}")
    check("20 <= документов <= 200", 20 <= man["n_documents"] <= 200, f"{man['n_documents']}")
    check("0 <= ошибок <= 12", len(json.loads(big["closed"]["truth.json"])["issues"]) <= 12)

    # итог ------------------------------------------------------------------- #
    passed = sum(1 for ok, _ in RESULTS if ok)
    total = len(RESULTS)
    elapsed = time.monotonic() - t0
    print("\n" + "=" * 62)
    print(f"ИТОГ: {passed}/{total} проверок пройдено за {elapsed:.1f} с")
    if passed != total:
        for ok, name in RESULTS:
            if not ok:
                print(f"  FAIL -> {name}")
    print("=" * 62)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
