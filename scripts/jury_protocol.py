#!/usr/bin/env python3
"""Репетиция закрытой проверки жюри — полный прогон «как жюри».

    python scripts/jury_protocol.py               # полный прогон (20 эпизодов)
    python scripts/jury_protocol.py --episodes 5  # быстрая проверка

Всё выполняется ЧЕРЕЗ judge_adapter.py — ту самую точку входа, которую будет
вызывать оценщик (определения проверок от 20.11.2026):

  * generate_episode / make_env / score_episode;
  * обе ветки reset: closed + episode_dir и train + seed + team_config;
  * конверт действий {"name", "args"}, пагинация, калькулятор, ограниченные
    ошибки, бюджет шагов;
  * submit, шаг после завершения, детерминированный лимит времени (C18);
  * ЧЕТЫРЕ политики организаторов, прогнанные через канонический протокол,
    и их оценка официальным score_episode;
  * независимый пересчёт эталона оценщиком (jury_conditions) на свежих seed.

Код выхода 0 — только если все проверки прошли.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import judge_adapter as ja  # noqa: E402
import jury_conditions as jc  # noqa: E402
from audit_env import load_episode  # noqa: E402
from audit_env.policies import (  # noqa: E402
    MarkAllPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    ReconciliationPolicy,
)
from audit_env.reward import terminal_reward  # noqa: E402

# Раскладка D01-D20 — документирована в docs/SCHEMAS.md §8
# (те же конфиги используют tests/test_spec_block2.py).
MATRIX = [
    {"clean": True, "distractors": False},
    {"clean": True, "distractors": True},
    {"clean": True, "distractors": True},
    {"clean": True, "distractors": True},
    {"clean": True, "distractors": True},
    {"issue_types": ["duplicate_entry"], "n_issues": 1},
    {"issue_types": ["duplicate_entry"], "n_issues": 1},
    {"issue_types": ["duplicate_entry"], "n_issues": 1},
    {"issue_types": ["partial_allocation"], "n_issues": 1, "partial_variant": "miss"},
    {"issue_types": ["partial_allocation"], "n_issues": 1, "partial_variant": "partial"},
    {"issue_types": ["partial_allocation"], "n_issues": 1, "partial_variant": "partial"},
    {"issue_types": ["fx_rate"], "n_issues": 1},
    {"issue_types": ["fx_rate"], "n_issues": 1},
    {"issue_types": ["period_cutoff"], "n_issues": 1},
    {"issue_types": ["period_cutoff"], "n_issues": 1},
    {"scenarios": ["S1"]},
    {"scenarios": ["S1"]},
    {"scenarios": ["S1"]},
    {"scenarios": ["S2"]},
    {"scenarios": ["S2"]},
]

POLICIES = ("oracle", "reconciliation", "mark_all", "random")
LEAK_TOKENS = ("truth", "provenance", "seed", "iss_", "expected_amount",
               "causal_change", "invariants_", "evidence_sets")


# --------------------------------------------------------------------------- #
# трансляция канонического протокола <-> формат наших политик
# --------------------------------------------------------------------------- #
def to_envelope(action: dict) -> dict:
    """Наше внутреннее действие -> конверт определений {"name", "args"}."""
    return {"name": action.get("action"),
            "args": {k: v for k, v in action.items() if k != "action"}}


def to_native(obs: dict, action: dict | None) -> dict:
    """Каноническое наблюдение -> вид, который понимают наши политики.

    list_records: {"records", "next_cursor": str|null}
      -> {"table", "rows", "next_cursor": int|null}; остальное — как есть.
    """
    out = dict(obs)
    result = obs.get("result")
    if isinstance(result, dict) and "records" in result:
        records = result.get("records") or []
        cursor = result.get("next_cursor")
        out["result"] = {
            "table": (action or {}).get("table"),
            "rows": records,
            "next_cursor": None if cursor is None else int(cursor),
            "count": len(records),
        }
    out.setdefault("steps_left", obs.get("remaining_steps"))
    return out


def run_policy_via_adapter(name: str, episode_dir: str) -> dict:
    """Прогон одной базовой политики через make_env() в закрытом режиме.

    Возвращает findings (или None), число шагов и inspection_steps —
    столько же вызовов позже уйдёт в официальный score_episode.
    """
    env = ja.make_env()
    obs, _info = env.reset(config={"mode": "closed", "episode_dir": str(episode_dir)})

    if name == "oracle":
        truth = json.loads((Path(episode_dir) / "closed" / "truth.json")
                           .read_text(encoding="utf-8"))["issues"]
        policy = OraclePolicy(truth)
    elif name == "reconciliation":
        policy = ReconciliationPolicy()
    elif name == "mark_all":
        policy = MarkAllPolicy()
    elif name == "random":
        policy = RandomExplorerPolicy(seed=7)
    else:
        raise ValueError(f"unknown policy {name}")

    native = to_native(obs, None)
    submitted: list[dict] | None = None
    submit_at = 0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < 101:
        action = policy(native)
        steps += 1
        if isinstance(action, dict) and action.get("action") == "submit":
            submitted = action.get("findings")
            submit_at = steps
        obs, _reward, terminated, truncated, _info = env.step(to_envelope(action))
        native = to_native(obs, action)

    inspection_steps = max(0, submit_at - 1) if submit_at else steps
    score = ja.score_episode(episode_dir,
                             {"findings": submitted or []}, inspection_steps)
    return {"name": name, "steps": steps, "inspection_steps": inspection_steps,
            "submitted": submitted is not None, "findings": submitted or [],
            "score": score}


# --------------------------------------------------------------------------- #
# 1. протокол адаптера (фикстуры C/E, ветки reset, действия, score)
# --------------------------------------------------------------------------- #
def rehearse_protocol(tmp: Path) -> list[str]:
    problems: list[str] = []

    def fail(msg: str) -> None:
        problems.append(msg)

    clean_dir, error_dir = tmp / "C", tmp / "E"
    ja.generate_episode(770_001, {"n_invoices": 70, "clean": True,
                                  "n_issues": 0, "distractors": False},
                        str(clean_dir))
    ja.generate_episode(770_002, {"n_invoices": 70}, str(error_dir))

    for zone, names in (("open", ("manifest.json", "ledger.csv", "invoices.jsonl",
                                  "payments.jsonl", "fx.csv", "policy.json")),
                        ("closed", ("truth.json", "provenance.json"))):
        for fname in names:
            if not (error_dir / zone / fname).exists():
                fail(f"generate_episode: нет {zone}/{fname}")

    # -- ветка closed + episode_dir --------------------------------------- #
    env = ja.make_env()
    obs, info = env.reset(config={"mode": "closed", "episode_dir": str(error_dir)})
    for key in ("task", "manifest", "policy", "remaining_steps"):
        if key not in obs:
            fail(f"reset(closed): нет ключа {key}")
    blob = json.dumps(obs, ensure_ascii=False) + json.dumps(info, ensure_ascii=False)
    for token in LEAK_TOKENS:
        if token in blob:
            fail(f"reset(closed): утечка {token}")
    if info.get("mode") != "eval":
        fail(f"reset(closed): mode={info.get('mode')} (ожидался eval)")

    # -- ветка train + seed + team_config --------------------------------- #
    env_t = ja.make_env()
    obs_t, info_t = env_t.reset(seed=123_456,
                                config={"mode": "train",
                                        "team_config": {"n_invoices": 30}})
    if info_t.get("mode") != "train":
        fail(f"reset(train): mode={info_t.get('mode')}")
    if not isinstance(obs_t.get("manifest"), dict):
        fail("reset(train): нет манифеста")
    if json.dumps(obs_t, ensure_ascii=False).count('"seed"'):
        fail("reset(train): в наблюдении фигурирует seed")
    env_t2 = ja.make_env()
    obs_t2, _ = env_t2.reset(seed=123_456,
                             config={"mode": "train",
                                     "team_config": {"n_invoices": 30}})
    if obs_t["manifest"]["episode_id"] != obs_t2["manifest"]["episode_id"]:
        fail("reset(train): одинаковый seed даёт разные эпизоды")

    # -- пагинация (фикстура L: >= 101 записи) ----------------------------- #
    total = obs["manifest"]["n_records"]
    if total < 101:
        fail(f"фикстура L: всего {total} записей (нужно >= 101)")
    seen: list[str] = []
    cursor, pages = None, 0
    while True:
        out, *_ = env.step({"name": "list_records",
                            "args": {"table": "ledger", "cursor": cursor,
                                     "limit": 50}})
        rows = (out.get("result") or {}).get("records") or []
        if len(rows) > 50:
            fail("пагинация: страница больше 50 строк")
        seen += [r["row_id"] for r in rows]
        pages += 1
        cursor = out.get("result", {}).get("next_cursor") if out.get("result") else None
        if cursor is None:
            break
        if pages > 100:
            fail("пагинация: не завершилась")
            break
    if len(seen) != total or len(set(seen)) != total:
        fail(f"пагинация: получено {len(seen)} уникальных из {total}")
    out_first, *_ = env.step({"name": "list_records",
                              "args": {"table": "ledger", "cursor": None,
                                       "limit": 50}})
    out_again, *_ = env.step({"name": "list_records",
                              "args": {"table": "ledger", "cursor": None,
                                       "limit": 50}})
    if out_first.get("result") != out_again.get("result"):
        fail("пагинация: повтор страницы дал другие строки")

    # -- калькулятор и ограниченные ошибки --------------------------------- #
    out, *_ = env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
    if out.get("result") != "0.3":
        fail(f"калькулятор: 0.1+0.2 -> {out.get('result')!r} (ожидалось '0.3')")
    out, *_ = env.step({"name": "calculator",
                        "args": {"expression": "open('f').read()"}})
    if out.get("result") is not None or not out.get("error"):
        fail("калькулятор: выражение вне грамматики не дало ошибки")
    before = env._env._steps
    out, *_ = env.step({"name": "get_record",
                        "args": {"table": "ledger", "id": "U"}})
    if not out.get("error") or env._env._steps != before + 1:
        fail("ошибка id: не ограничена или не расходует шаг")
    before = env._env._steps
    out, *_ = env.step({"name": "some_unknown_action", "args": {}})
    if (out.get("error") or {}).get("code") != "unknown_action" \
            or env._env._steps != before + 1:
        fail("неизвестное действие: не ограничено или не расходует шаг")

    # -- submit и шаг после завершения ------------------------------------- #
    out, reward, term, trunc, _ = env.step({"name": "submit",
                                            "args": {"findings": {"findings": []}}})
    if not (term and not trunc) or out.get("result") != {"accepted": True}:
        fail("submit: не завершил эпизод или вернул не {'accepted': true}")
    if reward != 0.0:
        fail(f"закрытый режим: reward политики {reward} (ожидался 0)")
    report_before = env._env.evaluator_report()
    env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None}})
    report_after = env._env.evaluator_report()
    if report_before["metrics"] != report_after["metrics"] \
            or report_before["true_reward"] != report_after["true_reward"]:
        fail("шаг после завершения изменил состояние (C15)")

    # -- score_episode: валидный и структурно невалидный ------------------- #
    truth = json.loads((error_dir / "closed" / "truth.json")
                       .read_text(encoding="utf-8"))["issues"]
    oracle_findings = [{"type": i["type"],
                        "affected_ids": list(i["affected_ids"]),
                        "claimed_amount_minor": i["expected_amount_minor"],
                        "currency": i["currency"],
                        "evidence_ids": list(i["evidence_sets"][0]),
                        "explanation": "oracle"} for i in truth]
    score = ja.score_episode(str(error_dir), {"findings": oracle_findings}, 10)
    if score["f1"] != 1.0:
        fail(f"score_episode(оракул): f1={score['f1']}")
    if abs(score["return"] - terminal_reward(1.0, 10)) > 1e-9:
        fail(f"score_episode: return={score['return']} != 0.98")
    bad = ja.score_episode(str(error_dir),
                           {"findings": oracle_findings * 31}, 10)
    if bad["return"] != -1.0 or any(bad[k] is not None
                                     for k in ("tp", "fp", "fn", "precision",
                                               "recall", "f1")):
        fail("score_episode: 31 находка не дала return=-1 с null-метриками")
    if not ja.evaluator_log() or ja.evaluator_log()[-1]["event"] != "submit_invalid":
        fail("score_episode: структурная ошибка не записана в журнал оценщика")

    # -- C18: детерминированный лимит времени ------------------------------ #
    env_c = ja.make_env()
    env_c.reset(config={"mode": "closed", "episode_dir": str(error_dir),
                        "time_limit_s": 0})
    out, _reward, term, trunc, _ = env_c.step({"name": "list_records",
                                               "args": {"table": "ledger",
                                                        "cursor": None}})
    if trunc and not term and (out.get("error") or {}).get("code") == "time_limit":
        rep = env_c._env.evaluator_report()
        if rep["metrics"]["n_findings"] != 0:
            fail("C18: лимит времени не дал пустой итоговый ответ")
    else:
        fail("C18: лимит времени не сработал детерминированно")

    return problems


# --------------------------------------------------------------------------- #
# 2. четыре политики через канонический протокол + независимый пересчёт
# --------------------------------------------------------------------------- #
def rehearse_policies(tmp: Path, n_episodes: int) -> tuple[list[str], dict]:
    problems: list[str] = []
    stats = {"episodes": 0, "runs": 0, "perfect": 0, "recomputed": 0}

    for i in range(n_episodes):
        seed = 990_000 + i                      # свежие seed, нигде раньше
        cfg = dict(MATRIX[i % len(MATRIX)])
        episode_dir = tmp / f"ep_{i:02d}"
        ja.generate_episode(seed, cfg, str(episode_dir))
        stats["episodes"] += 1

        # независимый пересчёт эталона — на стороне оценщика, не наш matcher
        r_imb, r_ar, r_before, detail = jc.independent_recompute(
            load_episode(str(episode_dir)))
        if r_imb and r_ar and r_before:
            stats["recomputed"] += 1
        else:
            problems.append(f"ep{i:02d}: независимый пересчёт не сошёлся ({detail})")

        for name in POLICIES:
            run = run_policy_via_adapter(name, str(episode_dir))
            score = run["score"]
            stats["runs"] += 1
            tag = f"ep{i:02d}/{name}"
            if run["steps"] > 100:
                problems.append(f"{tag}: {run['steps']} шагов (лимит 100)")
            if score["return"] is None:
                problems.append(f"{tag}: score_episode вернул null-метрики")
                continue
            if name in ("oracle", "reconciliation"):
                if score["f1"] != 1.0:
                    problems.append(f"{tag}: F1={score['f1']} (ожидался 1.0)")
                else:
                    stats["perfect"] += 1
            else:  # атаки
                if score["f1"] != 0.0 or score["return"] != -1.0:
                    problems.append(
                        f"{tag}: F1={score['f1']} return={score['return']} "
                        "(атака должна получать -1)")

    return problems, stats


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20,
                    help="число эпизодов для прогона четырёх политик")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="jury_protocol_"))
    try:
        print("=" * 70)
        print("  РЕПЕТИЦИЯ ЗАКРЫТОЙ ПРОВЕРКИ ЖЮРИ — прогон через judge_adapter")
        print("=" * 70)

        print("\n1) Протокол адаптера: генерация, обе ветки reset, пагинация,")
        print("   калькулятор, ограниченные ошибки, submit, score, лимит времени")
        proto_problems = rehearse_protocol(tmp)
        for p in proto_problems:
            print(f"  [X] {p}")
        print(f"  [{'OK' if not proto_problems else 'FAIL'}] "
              f"протокол адаптера: {'все проверки прошли' if not proto_problems else str(len(proto_problems)) + ' проблем'}")

        print(f"\n2) Четыре политики через канонический протокол "
              f"({args.episodes} свежих эпизодов, seed 990000+)")
        pol_problems, stats = rehearse_policies(tmp, args.episodes)
        for p in pol_problems:
            print(f"  [X] {p}")
        print(f"  эпизодов: {stats['episodes']}, прогонов: {stats['runs']}, "
              f"F1=1.0: {stats['perfect']}, независимый пересчёт: "
              f"{stats['recomputed']}/{stats['episodes']}")
        print(f"  [{'OK' if not pol_problems else 'FAIL'}] "
              f"четыре политики: {'ожидаемые результаты' if not pol_problems else str(len(pol_problems)) + ' проблем'}")

        problems = proto_problems + pol_problems
        print("\n" + "=" * 70)
        if problems:
            print(f"ИТОГ: {len(problems)} проблем — см. строки [X] выше")
            print("=" * 70)
            return 1
        print("ИТОГ: репетиция закрытой проверки пройдена полностью")
        print("=" * 70)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
