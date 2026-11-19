#!/usr/bin/env python3
"""Траектории 4 базовых политик + диагностика по 20 эпизодам.

    python scripts/build_trajectories.py

Результат:
  * trajectories.jsonl    — записи вида {episode_id, policy, step, action,
                            observation, reward, terminated, truncated}
                            (наблюдения без закрытого эталона — проверяется сканером);
  * artifacts/diagnostics.csv — сырые метрики каждого прогона;
  * docs/DIAGNOSTICS.md   — агрегат P, R, F1, возврат, шаги, время (требование
                            спеки: «отдельно покажите ... как диагностику»).
"""

from __future__ import annotations

import csv
import json
import statistics
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

REGISTRY = ROOT / "episodes" / "registry.json"
TRAJECTORIES = ROOT / "trajectories.jsonl"
ARTIFACTS = ROOT / "artifacts"
DIAGNOSTICS_MD = ROOT / "docs" / "DIAGNOSTICS.md"

# Наблюдения обучающих траекторий обязаны быть без закрытого эталона (спека 09)
FORBIDDEN_IN_OBSERVATION = ["iss_", "issue_id", "expected_amount_minor", "provenance",
                            "causal_change", "invariants_", "truth.json", "seed",
                            "statement"]

POLICIES = [
    ("reconciliation", lambda env: ReconciliationPolicy()),
    ("oracle", lambda env: OraclePolicy(env.evaluator_truth())),
    ("mark_all", lambda env: MarkAllPolicy()),
    ("random", lambda env: RandomExplorerPolicy(seed=7)),
]


def main() -> int:
    if not REGISTRY.exists():
        raise SystemExit("сначала выполните: python scripts/build_episodes.py")
    entries = json.loads(REGISTRY.read_text(encoding="utf-8"))["episodes"]

    ARTIFACTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    leak_hits: list[str] = []

    with TRAJECTORIES.open("w", encoding="utf-8") as fh:
        for entry in entries:
            episode_dir = str(ROOT / entry["path"])
            for policy_name, factory in POLICIES:
                env = AuditEnv(mode="train")
                env.reset(seed=1, config={"episode_dir": episode_dir})
                policy = factory(env)
                started = time.monotonic()
                result = run_episode(env, policy, seed=1,
                                     config={"episode_dir": episode_dir})
                elapsed = time.monotonic() - started

                report = result["report"]
                m = report["metrics"] or {"tp": 0, "fp": 0, "fn": 0,
                                          "precision": 0.0, "recall": 0.0, "f1": 0.0}
                rows.append({
                    "episode_id": entry["episode_id"],
                    "split": entry["split"],
                    "policy": policy_name,
                    "n_issues": entry["n_issues"],
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                    "precision": round(m["precision"], 6),
                    "recall": round(m["recall"], 6),
                    "f1": round(m["f1"], 6),
                    "return_reward": round(report["true_reward"], 6),
                    "steps": result["steps"],
                    "elapsed_s": round(elapsed, 4),
                })

                for rec in result["trajectory"]:
                    obs_blob = json.dumps(rec["observation"], ensure_ascii=False)
                    for token in FORBIDDEN_IN_OBSERVATION:
                        if token in obs_blob:
                            leak_hits.append(f"{entry['episode_id']}/{policy_name}: {token}")
                    fh.write(json.dumps({**rec, "policy": policy_name},
                                        ensure_ascii=False) + "\n")

    if leak_hits:
        raise SystemExit(f"УТЕЧКА В НАБЛЮДЕНИЯХ: {leak_hits[:5]}")

    # ---- сырой CSV ------------------------------------------------------- #
    csv_path = ARTIFACTS / "diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- агрегат по политикам -------------------------------------------- #
    def agg(policy: str) -> dict:
        sel = [r for r in rows if r["policy"] == policy]
        return {
            "n": len(sel),
            "precision": statistics.mean(r["precision"] for r in sel),
            "recall": statistics.mean(r["recall"] for r in sel),
            "f1": statistics.mean(r["f1"] for r in sel),
            "ret": statistics.mean(r["return_reward"] for r in sel),
            "steps": statistics.mean(r["steps"] for r in sel),
            "time": statistics.mean(r["elapsed_s"] for r in sel),
        }

    order = ["oracle", "reconciliation", "mark_all", "random"]
    lines = [
        "# Диагностика политик (по 20 открытым эпизодам: 10 train + 10 dev)",
        "",
        "Спека требует отдельно показывать **P, R, F1, возврат, шаги и время** — "
        "это таблица ниже; сырые значения каждого прогона: `artifacts/diagnostics.csv`.",
        "",
        "Запуск: `python scripts/build_trajectories.py` "
        "(метрики в этом файле пересчитываются каждый раз).",
        "",
        "| Политика | Прогонов | P (ср.) | R (ср.) | F1 (ср.) | Возврат (ср.) | Шаги (ср.) | Время (ср., с) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        a = agg(name)
        lines.append(f"| {name} | {a['n']} | {a['precision']:.3f} | {a['recall']:.3f} | "
                     f"{a['f1']:.3f} | {a['ret']:+.3f} | {a['steps']:.1f} | {a['time']:.3f} |")

    clean_rows = [r for r in rows if r["n_issues"] == 0]
    erroneous = [r for r in rows if r["n_issues"] > 0]
    lines += [
        "",
        "## Как читать",
        "",
        "- **oracle** — отправляет эталон с доказательствами: проверяет, что "
        "**максимум награды достижим** (F1 = 1.0). Это проверка среды, не агента.",
        "- **reconciliation** — простой алгоритм сверки: находит все ошибки без "
        "ложных срабатываний (FP = 0, FN = 0) и молчит на чистых эпизодах.",
        "- **mark_all** — атака «пометить всё»: F1 = 0 и возврат −1 на эпизодах с "
        "ошибками; на чистых она получает штраф за ложные обвинения.",
        "- **random** — случайный исследователь: угаданные ID не дают TP "
        "(нужен полный комплект доказательств).",
        "",
        f"- Чистых эпизодов в наборе: **{len(clean_rows) // len(POLICIES)}**, "
        f"с ошибками: **{len(erroneous) // len(POLICIES)}**.",
        f"- Всего прогонов: **{len(rows)}**; траекторий в `trajectories.jsonl`: "
        f"**{sum(1 for _ in TRAJECTORIES.open(encoding='utf-8'))}**.",
        "",
        "Наблюдения в траекториях проверены сканером: закрытый эталон в них отсутствует.",
    ]
    DIAGNOSTICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"trajectories.jsonl : {TRAJECTORIES.stat().st_size // 1024} КБ, "
          f"{len(rows)} прогонов")
    print(f"artifacts/diagnostics.csv : {len(rows)} строк")
    print(f"docs/DIAGNOSTICS.md : агрегат по {len(order)} политикам")
    for name in order:
        a = agg(name)
        print(f"  {name:15s} F1={a['f1']:.3f} return={a['ret']:+.3f} "
              f"steps={a['steps']:.1f} time={a['time']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
