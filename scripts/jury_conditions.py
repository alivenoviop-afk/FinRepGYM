#!/usr/bin/env python3
"""Слепая проверка СРЕДЫ на условиях оценщиков (спека 06 и 10).

    python scripts/jury_conditions.py

Чем отличается от organizer_check: здесь берутся seed'ы и сочетания параметров,
которых НЕТ ни в train/dev, ни в check, ни в organizer — как это сделал бы жюри
(«скрытые seed и НОВЫЕ сочетания параметров»):

  * размеры 50-100 счетов, смещения +00:00/-03:00/+06:00, одна валюта,
    distractors=false, clean+scenarios (доминирование чистоты),
    сценарии с одиночными типами;
  * состав строго 5 чистых / 5 одиночных / 5 парных / 5 взаимодействий.

Проверки на каждой среде (все — независимо от доверия генератору):
  1) размеры в рамках спеки;               2) равенства ДО ошибки (provenance);
  3) НЕЗАВИСИМЫЙ пересчёт эталона из открытых записей (эффекты каждой ошибки);
  4) оракул = максимум;                    5) сверка без FP/FN;
  6) каждый прогон < 60 с;                 7) нет эталона в наблюдениях;
  8) 20 парных проверок (перестановка + смена ID).

Отчёт: artifacts/jury_report.md; код выхода 0 только при полном успехе.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_env import AuditEnv, permute_episode, remap_submission, save_episode  # noqa: E402
from audit_env.policies import (  # noqa: E402
    MarkAllPolicy, OraclePolicy, RandomExplorerPolicy, ReconciliationPolicy, run_episode,
)

REPORT = ROOT / "artifacts" / "jury_report.md"
FORBIDDEN = ["iss_", "issue_id", "expected_amount_minor", "provenance", "causal_change",
             "invariants_", "truth.json", "statement"]

DUP, CUT, PART, FX = ("duplicate_entry", "period_cutoff",
                      "partial_allocation", "fx_rate")


def clean(n, **e): return {"n_invoices": n, "clean": True, "n_issues": 0, "scenarios": [], **e}
def single(n, t, **e): return {"n_invoices": n, "issue_types": [t], "n_issues": 1, **e}
def pair(n, a, b, **e): return {"n_invoices": n, "issue_types": [a, b], "n_issues": 2, **e}
def scen(n, names, issues=None, **e): return {"n_invoices": n, "issue_types": issues or [],
                                             "n_issues": len(issues or []),
                                             "scenarios": names, **e}


# 20 никогда не использовавшихся seed + новые сочетания параметров
JURY: list[tuple[str, int, dict]] = [
    # -- 5 чистых (включая экзотику) --
    # n подобраны с запасом: при малых n разброс по seed уводит ниже 200 строк
    ("clean", 97001, clean(65, utc_offset="+00:00")),
    ("clean", 97002, clean(90, distractors=False)),
    ("clean", 97003, clean(70, currencies=["USD"])),          # одна валюта
    ("clean", 97004, clean(65, scenarios=["S1", "S2"])),      # clean главенствует
    ("clean", 97005, clean(65, utc_offset="-03:00")),
    # -- 5 одиночных --
    ("single", 97006, single(70, FX, distractors=False)),     # fx без отвлекающих
    ("single", 97007, single(65, DUP)),
    ("single", 97008, single(60, PART)),
    ("single", 97009, single(75, CUT, utc_offset="+06:00")),
    ("single", 97010, single(90, FX)),
    # -- 5 парных --
    ("pair", 97011, pair(65, DUP, FX)),
    ("pair", 97012, pair(70, PART, CUT)),
    ("pair", 97013, pair(60, FX, CUT)),
    ("pair", 97014, pair(90, DUP, PART)),
    ("pair", 97015, pair(65, PART, FX)),
    # -- 5 взаимодействий --
    ("interacting", 97016, scen(70, ["S1"], distractors=False)),
    ("interacting", 97017, scen(70, ["S2"])),
    ("interacting", 97018, scen(65, ["S1", "S2"])),
    ("interacting", 97019, scen(80, ["S1"], issues=[DUP])),
    ("interacting", 97020, scen(65, ["S2"], issues=[FX])),
]

POLICIES = [
    ("reconciliation", lambda env: ReconciliationPolicy()),
    ("oracle", lambda env: OraclePolicy(env.evaluator_truth())),
    ("mark_all", lambda env: MarkAllPolicy()),
    ("random", lambda env: RandomExplorerPolicy(seed=7)),   # фиксированный seed (спека 06)
]


# --------------------------------------------------------------------------- #
# Независимый пересчёт эффектов каждой ошибки из открытых записей
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, files):
        self.manifest = json.loads(files["open"]["manifest.json"])
        self.ledger = {r["row_id"]: r for r in
                       csv.DictReader(io.StringIO(files["open"]["ledger.csv"]))}
        for r in self.ledger.values():
            r["debit_minor"], r["credit_minor"] = int(r["debit_minor"]), int(r["credit_minor"])
        self.payments = {p["payment_id"]: p for p in
                         (json.loads(l) for l in files["open"]["payments.jsonl"].splitlines() if l)}
        for p in self.payments.values():
            p["amount_minor"] = int(p["amount_minor"])
        self.rates = {(r["date"], r["source_currency"], r["target_currency"]): r["rate"]
                      for r in csv.DictReader(io.StringIO(files["open"]["fx.csv"]))}
        self.policy_day = self.manifest["period_end"][:10]
        self.base = self.manifest["base_currency"]
        self.start = self.manifest["period_start"]
        self.end = self.manifest["period_end"]
        self.truth = json.loads(files["closed"]["truth.json"])["issues"]
        self.prov = json.loads(files["closed"]["provenance.json"])

    def conv(self, amount, src, tgt=None):
        from decimal import Decimal, ROUND_HALF_UP
        tgt = tgt or self.base
        if src == tgt:
            return int(amount)
        rate = Decimal(self.rates[(self.policy_day, src, tgt)])
        return int((Decimal(int(amount)) * rate).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def issue_effects(issue, ctx):
    """(effect на Σдебет−Σкредит, effect на AR-расхождение) в базовой валюте,
    вычисленные ТОЛЬКО по открытым записям и правилам."""
    t = issue["type"]
    ids = issue["affected_ids"]
    rows = [ctx.ledger[i] for i in ids if i in ctx.ledger]
    pay = next((ctx.payments[i] for i in ids if i in ctx.payments), None)

    if t == "duplicate_entry":
        row = rows[0]
        amount = row["credit_minor"] or row["debit_minor"]
        return -ctx.conv(amount, row["currency"]), 0

    if t == "period_cutoff":
        ar = next(r for r in rows if r["account"].startswith("1200"))
        assert not (ctx.start <= ar["timestamp"] <= ctx.end), "строка cutoff вне периода"
        return 0, ctx.conv(ar["credit_minor"], ar["currency"])

    if t == "partial_allocation":
        ar = next(r for r in rows if r["account"].startswith("1200"))
        assert pay is not None
        expected = ctx.conv(pay["amount_minor"], pay["currency"], ar["currency"])
        assert int(ar["credit_minor"]) < expected, "частичное распределение не видно"
        # Наблюдаемая разница = conv(документ) - conv(строка журнала), каждая
        # конвертируется отдельно: conv(a-b) может отличаться на единицу.
        effect = (ctx.conv(pay["amount_minor"], pay["currency"])
                  - ctx.conv(int(ar["credit_minor"]), ar["currency"]))
        return 0, effect

    if t == "fx_rate":
        ar = next(r for r in rows if r["account"].startswith("1200"))
        assert pay is not None
        correct = ctx.conv(pay["amount_minor"], pay["currency"], ar["currency"])
        assert int(ar["credit_minor"]) != correct, "неверный курс не обнаружен"
        effect = ctx.conv(correct, ar["currency"]) - ctx.conv(int(ar["credit_minor"]),
                                                              ar["currency"])
        return effect, effect

    raise AssertionError(f"неизвестный тип {t}")


def independent_recompute(files):
    """Сверяет наблюдаемые расхождения с суммой эффектов, посчитанных из записей."""
    ctx = Ctx(files)
    # Наблюдаемые суммы — ТОЖЕ в базовой валюте (иначе слагаемые в разных валютах)
    observed_imbalance = sum(ctx.conv(r["debit_minor"], r["currency"])
                             - ctx.conv(r["credit_minor"], r["currency"])
                             for r in ctx.ledger.values())
    ar_balance = sum(ctx.conv(r["debit_minor"], r["currency"])
                     - ctx.conv(r["credit_minor"], r["currency"])
                     for r in ctx.ledger.values()
                     if r["account"].startswith("1200") and ctx.start <= r["timestamp"] <= ctx.end)
    # выставлено и оплачено — с конвертацией каждой суммы в базовую валюту
    inv_base = 0
    for line in files["open"]["invoices.jsonl"].splitlines():
        if not line:
            continue
        i = json.loads(line)
        if ctx.start <= i["issued_at"] <= ctx.end:
            inv_base += ctx.conv(int(i["total_minor"]), i["currency"])
    pay_base = 0
    for line in files["open"]["payments.jsonl"].splitlines():
        if not line:
            continue
        p = json.loads(line)
        if ctx.start <= p["timestamp"] <= ctx.end:
            pay_base += ctx.conv(int(p["amount_minor"]), p["currency"])
    observed_ar = ar_balance - (inv_base - pay_base)

    sum_imbalance = 0
    sum_ar = 0
    for issue in ctx.truth:
        e_i, e_ar = issue_effects(issue, ctx)
        sum_imbalance += e_i
        sum_ar += e_ar

    before_ok = (ctx.prov["invariants_before"]["books_balanced"]
                 and ctx.prov["invariants_before"]["ar_matches_documents"])
    return (observed_imbalance == sum_imbalance,
            observed_ar == sum_ar,
            before_ok,
            f"наблюждено({observed_imbalance},{observed_ar}) "
            f"пересчитано({sum_imbalance},{sum_ar})")


def run_policies(seed, cfg, results, problems, leaks, times):
    env = AuditEnv(mode="train")
    env.reset(seed=seed, config=cfg)
    manifest = dict(env._manifest)
    n_issues = len(env.evaluator_truth())

    for name, factory in POLICIES:
        env2 = AuditEnv(mode="train")
        env2.reset(seed=seed, config=cfg)
        started = time.monotonic()
        res = run_episode(env2, factory(env2), seed=seed, config=cfg)
        elapsed = time.monotonic() - started
        times.append(elapsed)
        m = res["report"]["metrics"] or {"precision": 0, "recall": 0, "f1": 0,
                                         "tp": 0, "fp": 0, "fn": 0}
        results.append({"policy": name, "f1": m["f1"], "reward": res["report"]["true_reward"],
                        "steps": res["steps"], "elapsed": round(elapsed, 4),
                        "fp": m["fp"], "fn": m["fn"]})
        for rec in res["trajectory"]:
            blob = json.dumps(rec["observation"], ensure_ascii=False)
            leaks.extend(t for t in FORBIDDEN if t in blob)
        if name == "oracle" and m["f1"] != 1.0:
            problems.append(f"seed={seed}: оракул F1={m['f1']}")
        if name == "reconciliation" and (m["fp"] or m["fn"]):
            problems.append(f"seed={seed}: сверка FP={m['fp']} FN={m['fn']}")
        if elapsed >= 60:
            problems.append(f"seed={seed} {name}: {elapsed:.1f}s >= 60s")
    return manifest, n_issues


def paired_checks(episodes, problems):
    """20 парных проверок: перестановка + смена ID -> ответ и награда сохраняются."""
    ok = 0
    for idx, (group, seed, cfg) in enumerate(episodes):
        import shutil
        import tempfile
        tmp = None
        try:
            files = generate_for(cfg, seed)
            perm, id_map = permute_episode(files, seed=seed + 90909)
            tmp = Path(tempfile.mkdtemp(prefix="jury_pair_"))
            save_episode(files, tmp / "a")
            save_episode(perm, tmp / "b")
            env_o = AuditEnv()
            env_o.reset(seed=1, config={"episode_dir": str(tmp / "a")})
            res_o = run_episode(env_o, OraclePolicy(env_o.evaluator_truth()), seed=1,
                                config={"episode_dir": str(tmp / "a")})
            remapped = remap_submission(res_o["trajectory"][-1]["action"], id_map)
            env_p = AuditEnv()
            env_p.reset(seed=1, config={"episode_dir": str(tmp / "b")})
            _, reward, term, trunc, _ = env_p.step(remapped)
            rep = env_p.evaluator_report()
            same = (term and not trunc
                    and rep["metrics"]["f1"] == res_o["report"]["metrics"]["f1"] == 1.0
                    and abs(reward - res_o["report"]["true_reward"]) < 1e-12)
            ok += int(same)
            if not same:
                problems.append(f"парная #{idx + 1} seed={seed}: награда/ответ не сохранились")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"парная #{idx + 1} seed={seed}: {type(exc).__name__}: {exc}")
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
    return ok


def generate_for(cfg, seed):
    from audit_env import generate
    return generate(seed, cfg)


def main() -> int:
    started = time.monotonic()
    results, problems, leaks, times = [], [], [], []
    counts: dict[str, int] = {"clean": 0, "single": 0, "pair": 0, "interacting": 0}
    recompute_ok = before_ok = size_ok = 0
    lines: list[str] = ["# Слепая проверка на условиях оценщиков",
                        "", "Seed'ы 97001–97020 (не использовались нигде) и **новые "
                        "сочетания параметров**; отчёт сформирован автоматически.", ""]

    for group, seed, cfg in JURY:
        counts[group] += 1
        files = generate_for(cfg, seed)
        manifest, n_issues = run_policies(seed, cfg, results, problems, leaks, times)

        ok_size = 200 <= manifest["n_records"] <= 2000 and 20 <= manifest["n_documents"] <= 200 \
            and 0 <= n_issues <= 12
        size_ok += int(ok_size)
        if not ok_size:
            problems.append(f"seed={seed}: размер {manifest['n_records']}/"
                            f"{manifest['n_documents']}/{n_issues} вне рамок")

        r_imb, r_ar, r_before, detail = independent_recompute(files)
        recompute_ok += int(r_imb and r_ar)
        before_ok += int(r_before)
        if not (r_imb and r_ar):
            problems.append(f"seed={seed}: независимый пересчёт не сошёлся ({detail})")
        if not r_before:
            problems.append(f"seed={seed}: равенства до ошибки не подтверждаются")

        combo = ",".join(f"{k}={v}" for k, v in cfg.items()
                         if k in ("n_invoices", "clean", "issue_types", "scenarios",
                                  "distractors", "currencies", "utc_offset"))
        lines.append(f"- [{group} {counts[group]}/5] seed={seed} rows={manifest['n_records']} "
                     f"docs={manifest['n_documents']} issues={n_issues} :: {combo}")
        print(f"  [{group:11s} {counts[group]}/5] seed={seed} rows={manifest['n_records']:3d} "
              f"issues={n_issues} пересчёт={'OK' if (r_imb and r_ar) else 'FAIL'}")

    # ---- 20 парных проверок --------------------------------------------- #
    print("  парные проверки (перестановка + смена ID, 20 шт.)...")
    pairs_ok = paired_checks(JURY, problems)

    # ---- проверки --------------------------------------------------------- #
    checks = [
        ("состав 5/5/5/5 (чистые/одиночные/парные/взаимодействия)",
         all(v == 5 for v in counts.values()), str(counts)),
        ("размеры 200-2000 / 20-200 / 0-12 на всех 20",
         size_ok == 20, f"{size_ok}/20"),
        ("равенства бухгалтерии ДО внесения ошибки (provenance)",
         before_ok == 20, f"{before_ok}/20"),
        ("НЕЗАВИСИМЫЙ пересчёт эталона из открытых записей",
         recompute_ok == 20, f"{recompute_ok}/20"),
        ("оракул достигает максимума на всех эпизодах",
         not any("оракул" in p for p in problems), "F1 = 1.0"),
        ("сверка без FP/FN на всех эпизодах",
         not any("сверка" in p for p in problems), "FP = 0, FN = 0"),
        ("каждый прогон < 60 секунд", max(times, default=0) < 60,
         f"max={max(times, default=0):.3f} с, прогонов {len(times)}"),
        ("нет закрытого эталона в наблюдениях", not sorted(set(leaks)),
         "чисто" if not leaks else str(sorted(set(leaks))[:3])),
        ("20 парных проверок: ответ и награда сохраняются", pairs_ok == 20, f"{pairs_ok}/20"),
    ]

    passed = sum(1 for _, ok, _ in checks if ok)
    lines += ["", "## Проверки", ""]
    for name, ok, detail in checks:
        lines.append(f"- {'[PASS]' if ok else '[FAIL]'} {name} — {detail}")
    lines += ["", "## Сводка политик (среднее по 20 эпизодам)", "",
              "| Политика | F1 | Возврат | Шаги |", "|---|---:|---:|---:|"]
    for name in ("oracle", "reconciliation", "mark_all", "random"):
        sel = [r for r in results if r["policy"] == name]
        if sel:
            lines.append(f"| {name} | {sum(r['f1'] for r in sel) / len(sel):.3f} | "
                         f"{sum(r['reward'] for r in sel) / len(sel):+.3f} | "
                         f"{sum(r['steps'] for r in sel) / len(sel):.1f} |")
    total = time.monotonic() - started
    lines += ["", f"Время: {total:.1f} с. Прогонов: {len(results)}.", ""]
    if problems:
        lines += ["## Проблемы", ""] + [f"- {p}" for p in problems] + [""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("-" * 70)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    print(f"\n  ИТОГ: {passed}/{len(checks)} групп, прогонов {len(results)}, {total:.1f} с")
    print(f"  Отчёт: {REPORT.relative_to(ROOT)}")
    if problems:
        for p in problems[:8]:
            print(f"  ! {p}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
