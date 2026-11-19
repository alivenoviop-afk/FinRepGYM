#!/usr/bin/env python3
"""Сборка открытых эпизодов: 10 train + 10 dev + реестр.

    python scripts/build_episodes.py

Раскладка (docs/RULES.md, решение 6):
  train = 2 чистых + 4 одиночных (все 4 типа) + 2 с двумя типами + 2 сценария
  dev   = 1 чистый + 4 одиночных + 3 с двумя типами + 2 сценария

Каждый эпизод проверяется ПЕРЕД записью на диск:
  * размеры в рамках спеки (200-2000 проводок, 20-200 документов, 0-12 ошибок);
  * оракул получает максимум (F1 = 1.0);
  * сверка не даёт ни FP, ни FN;
  * в открытых файлах нет закрытого эталона.

Эти эпизоды — ОТКРЫТЫЕ (train/dev): их правда лежит рядом, она нужна для
разработки. Закрытые эпизоды жюри собирается из СВОИХ seed и наших осей
конфигурации — в этот репозиторий они не попадают.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_env import AuditEnv, generate, save_episode  # noqa: E402
from audit_env.policies import OraclePolicy, ReconciliationPolicy, run_episode  # noqa: E402

EPISODES_DIR = ROOT / "episodes"
REGISTRY = EPISODES_DIR / "registry.json"

DUP = "duplicate_entry"
CUT = "period_cutoff"
PART = "partial_allocation"
FX = "fx_rate"


def clean(n: int, **extra) -> dict:
    return {"n_invoices": n, "clean": True, "n_issues": 0, "scenarios": [], **extra}


def single(n: int, kind: str, **extra) -> dict:
    return {"n_invoices": n, "issue_types": [kind], "n_issues": 1, **extra}


def pair(n: int, a: str, b: str, **extra) -> dict:
    return {"n_invoices": n, "issue_types": [a, b], "n_issues": 2, **extra}


def scenario(n: int, name: str, **extra) -> dict:
    return {"n_invoices": n, "issue_types": [], "n_issues": 0, "scenarios": [name], **extra}


# (split, seed, config) — seed'ы открытые и зафиксированы в реестре
SPEC: list[tuple[str, int, dict]] = [
    # ---- train: 2 чистых + 4 одиночных + 2 парных + 2 сценария ----
    ("train", 1001, clean(60)),
    ("train", 1002, clean(70, utc_offset="+03:00")),
    ("train", 1003, single(60, DUP)),
    ("train", 1004, single(65, PART)),
    ("train", 1005, single(70, FX)),
    ("train", 1006, single(75, CUT)),
    ("train", 1007, pair(60, DUP, CUT)),
    ("train", 1008, pair(65, PART, FX)),
    ("train", 1009, scenario(70, "S1")),
    ("train", 1010, scenario(75, "S2")),
    # ---- dev: 1 чистый + 4 одиночных + 3 парных + 2 сценария ----
    ("dev", 2001, clean(80, utc_offset="-07:00")),
    ("dev", 2002, single(55, DUP)),
    ("dev", 2003, single(60, PART)),
    ("dev", 2004, single(65, FX)),
    ("dev", 2005, single(70, CUT)),
    ("dev", 2006, pair(60, DUP, PART)),
    ("dev", 2007, pair(65, FX, CUT)),
    ("dev", 2008, pair(70, DUP, FX)),
    ("dev", 2009, scenario(60, "S1")),
    ("dev", 2010, scenario(65, "S2")),
]

FORBIDDEN = ["expected_amount_minor", "provenance", "causal_change", "invariants_",
             "truth.json", "statement"]


def verify_and_save(split: str, seed: int, config: dict) -> dict:
    files = generate(seed, config)

    # --- рамки спеки --- #
    manifest = json.loads(files["open"]["manifest.json"])
    truth = json.loads(files["closed"]["truth.json"])["issues"]
    assert 200 <= manifest["n_records"] <= 2000, manifest["n_records"]
    assert 20 <= manifest["n_documents"] <= 200, manifest["n_documents"]
    assert 0 <= len(truth) <= 12, len(truth)

    # --- закрытая зона не утекает в открытую --- #
    open_blob = json.dumps(files["open"], ensure_ascii=False)
    for token in FORBIDDEN:
        assert token not in open_blob, f"утечка {token!r} в открытых файлах"

    # --- сохранить и проверить через среду --- #
    out = EPISODES_DIR / split / manifest["episode_id"]
    if out.exists():
        shutil.rmtree(out)
    save_episode(files, out)

    env = AuditEnv(mode="train")
    env.reset(seed=1, config={"episode_dir": str(out)})
    oracle = run_episode(env, OraclePolicy(env.evaluator_truth()), seed=1,
                         config={"episode_dir": str(out)})
    assert oracle["report"]["metrics"]["f1"] == 1.0, "оракул не получил максимум"

    env2 = AuditEnv(mode="train")
    env2.reset(seed=1, config={"episode_dir": str(out)})
    recon = run_episode(env2, ReconciliationPolicy(), seed=1,
                        config={"episode_dir": str(out)})
    m = recon["report"]["metrics"]
    assert (m["fp"], m["fn"]) == (0, 0), f"сверка FP={m['fp']} FN={m['fn']}"

    return {
        "episode_id": manifest["episode_id"],
        "split": split,
        "seed": seed,
        "config": config,
        "path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "n_records": manifest["n_records"],
        "n_documents": manifest["n_documents"],
        "n_issues": len(truth),
        "issue_types": sorted({t["type"] for t in truth}),
        "oracle_f1": oracle["report"]["metrics"]["f1"],
        "recon_fp": m["fp"],
        "recon_fn": m["fn"],
    }


def main() -> int:
    if EPISODES_DIR.exists():
        shutil.rmtree(EPISODES_DIR)
    registry = {"schema_version": "audit-rl/1.0",
                "note": "открытые train/dev эпизоды; закрытые эпизоды жюри "
                        "генерирует своими seed из наших осей конфигурации",
                "episodes": []}
    for split, seed, config in SPEC:
        entry = verify_and_save(split, seed, config)
        registry["episodes"].append(entry)
        print(f"  [{split:5s}] seed={seed} {entry['episode_id']} "
              f"rows={entry['n_records']:3d} docs={entry['n_documents']:3d} "
              f"issues={entry['n_issues']} {','.join(entry['issue_types']) or 'clean'}")

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    n_train = sum(1 for e in registry["episodes"] if e["split"] == "train")
    n_dev = sum(1 for e in registry["episodes"] if e["split"] == "dev")
    print(f"\nГотово: {n_train} train + {n_dev} dev эпизодов -> {EPISODES_DIR}")
    print(f"Реестр: {REGISTRY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
