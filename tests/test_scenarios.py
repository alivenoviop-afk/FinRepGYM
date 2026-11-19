"""Два обязательных сценария взаимодействия (спека 02: «Взаимодействующие ошибки»).

S1 «курс + граница»      : fx_rate + period_cutoff на одном кросс-валютном платеже.
S2 «частичный платёж ...» : partial_allocation + period_cutoff на одном кросс-валютном
                            платеже (буквальный пример спеки: частичный платёж в другой
                            валюте на границе периода).

Проверяем: структуру (две ошибки делят один платёж), обнаружение без ложных
срабатываний, независимый пересчёт расхождений, разрешимость доказательств и то,
что наивные детекторы обязаны терять ошибки (это и есть «взаимодействие»).
"""

from __future__ import annotations

import csv
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_env import AuditEnv, generate  # noqa: E402
from audit_env.policies import OraclePolicy, ReconciliationPolicy, run_episode  # noqa: E402
from support import SEED  # noqa: E402
from test_data import fx_conv  # noqa: E402

S1_CFG = {"n_invoices": 30, "issue_types": [], "n_issues": 0, "scenarios": ["S1"]}
S2_CFG = {"n_invoices": 30, "issue_types": [], "n_issues": 0, "scenarios": ["S2"]}
BOTH_CFG = {"n_invoices": 40,
            "issue_types": ["duplicate_entry", "period_cutoff"], "n_issues": 2,
            "scenarios": ["S1", "S2"]}


def run(cfg, factory, seed: int = SEED) -> dict:
    env = AuditEnv()
    env.reset(seed=seed, config=cfg)
    return run_episode(env, factory(env), seed=seed, config=cfg)


def truth_of(cfg, seed: int = SEED) -> list[dict]:
    return json.loads(generate(seed, cfg)["closed"]["truth.json"])["issues"]


def parse(files):
    manifest = json.loads(files["open"]["manifest.json"])
    ledger = list(csv.DictReader(io.StringIO(files["open"]["ledger.csv"])))
    payments = [json.loads(l) for l in files["open"]["payments.jsonl"].splitlines() if l]
    return manifest, ledger, payments


def naive_date_only_detector(files) -> int:
    """Наивный детектор «только даты»: считает только cutoff."""
    manifest, ledger, payments = parse(files)
    start, end = manifest["period_start"], manifest["period_end"]
    by_payment: dict[str, list[dict]] = {}
    for r in ledger:
        if r["payment_id"] and r["payment_id"] != "null":
            by_payment.setdefault(r["payment_id"], []).append(r)
    hits = 0
    for pay in payments:
        rows = by_payment.get(pay["payment_id"], [])
        if not rows:
            continue
        doc_in = start <= pay["timestamp"] <= end
        rows_out = any(not (start <= r["timestamp"] <= end) for r in rows)
        if doc_in and rows_out:
            hits += 1
    return hits


class TestScenarioStructure(unittest.TestCase):
    def test_s1_is_fx_plus_cutoff_on_one_payment(self):
        truth = truth_of(S1_CFG)
        self.assertEqual(len(truth), 2)
        self.assertEqual({t["type"] for t in truth}, {"fx_rate", "period_cutoff"})
        pays = [next(x for x in t["affected_ids"] if x.startswith("PAY-")) for t in truth]
        self.assertEqual(len(set(pays)), 1, "обе ошибки делят один платёж")
        for t in truth:
            self.assertTrue(t["evidence_sets"], t["issue_id"])

    def test_s2_is_partial_plus_cutoff_on_one_payment(self):
        truth = truth_of(S2_CFG)
        self.assertEqual(len(truth), 2)
        self.assertEqual({t["type"] for t in truth}, {"partial_allocation", "period_cutoff"})
        pays = [next(x for x in t["affected_ids"] if x.startswith("PAY-")) for t in truth]
        self.assertEqual(len(set(pays)), 1)

    def test_s2_payment_is_really_cross_currency(self):
        files = generate(SEED, S2_CFG)
        manifest, ledger, payments = parse(files)
        truth = json.loads(files["closed"]["truth.json"])["issues"]
        pid = next(x for x in truth[0]["affected_ids"] if x.startswith("PAY-"))
        pay = next(p for p in payments if p["payment_id"] == pid)
        rows = [r for r in ledger if r["payment_id"] == pid]
        self.assertNotEqual(pay["currency"], rows[0]["currency"],
                            "документ в одной валюте, постинг в другой")
        self.assertEqual({r["currency"] for r in rows}, {manifest["base_currency"]})

    def test_s2_rows_cross_the_boundary_while_document_does_not(self):
        files = generate(SEED, S2_CFG)
        manifest, ledger, payments = parse(files)
        start, end = manifest["period_start"], manifest["period_end"]
        truth = json.loads(files["closed"]["truth.json"])["issues"]
        cutoff = next(t for t in truth if t["type"] == "period_cutoff")
        pid = next(x for x in cutoff["affected_ids"] if x.startswith("PAY-"))
        pay = next(p for p in payments if p["payment_id"] == pid)
        rows = [r for r in ledger if r["payment_id"] == pid]
        self.assertTrue(start <= pay["timestamp"] <= end, "документ внутри периода")
        self.assertTrue(all(r["timestamp"] > end for r in rows), "строки вне периода")

    def test_singles_and_scenarios_do_not_collide(self):
        truth = truth_of(BOTH_CFG)
        self.assertEqual(len(truth), 2 + 4, "одиночные + две ошибки на каждый сценарий")
        self.assertEqual(len({t["issue_id"] for t in truth}), len(truth), "id уникальны")
        # платежи одиночных мутаций и сценариев не пересекаются
        scen_pays = {next(x for x in t["affected_ids"] if x.startswith("PAY-"))
                     for t in truth if t["issue_id"].startswith(("iss_s1", "iss_s2"))}
        single_pays = {next(x for x in t["affected_ids"] if x.startswith("PAY-"))
                       for t in truth if not t["issue_id"].startswith(("iss_s1", "iss_s2"))
                       and t["type"] != "duplicate_entry"}
        self.assertFalse(scen_pays & single_pays, "сценарии берут свои платежи")


class TestScenarioDetection(unittest.TestCase):
    def test_reconciliation_finds_every_scenario_issue_without_false_positives(self):
        for cfg, expected in ((S1_CFG, 2), (S2_CFG, 2), (BOTH_CFG, 6)):
            with self.subTest(cfg=str(cfg)):
                result = run(cfg, lambda e: ReconciliationPolicy())
                m = result["report"]["metrics"]
                self.assertEqual(m["n_truth"], expected, m)
                self.assertEqual((m["tp"], m["fp"], m["fn"]), (expected, 0, 0), m)
                self.assertEqual(m["f1"], 1.0)
                self.assertGreater(result["report"]["true_reward"], 0.9)

    def test_oracle_reaches_maximum_on_scenarios(self):
        for cfg in (S1_CFG, S2_CFG, BOTH_CFG):
            with self.subTest(cfg=str(cfg)):
                result = run(cfg, lambda e: OraclePolicy(e.evaluator_truth()))
                self.assertEqual(result["report"]["metrics"]["f1"], 1.0)
                self.assertGreater(result["report"]["true_reward"], 0.9)

    def test_naive_date_only_detector_misses_the_other_errors(self):
        """Взаимодействие: детектор «только даты» находит лишь часть эталона."""
        files = generate(SEED, BOTH_CFG)
        truth = json.loads(files["closed"]["truth.json"])["issues"]
        naive = naive_date_only_detector(files)
        self.assertLess(naive, len(truth),
                        f"наивный детектор нашёл {naive} из {len(truth)}")
        # а полная сверка находит всё
        result = run(BOTH_CFG, lambda e: ReconciliationPolicy())
        self.assertEqual(result["report"]["metrics"]["fn"], 0)

    def test_naive_amount_comparison_without_conversion_is_wrong(self):
        """Сравнение документа и журнала без конвертации даёт не ту дельту (S2)."""
        files = generate(SEED, S2_CFG)
        manifest, ledger, payments = parse(files)
        truth = json.loads(files["closed"]["truth.json"])["issues"]
        partial = next(t for t in truth if t["type"] == "partial_allocation")
        pid = next(x for x in partial["affected_ids"] if x.startswith("PAY-"))
        pay = next(p for p in payments if p["payment_id"] == pid)
        cash = next(r for r in ledger if r["payment_id"] == pid
                    and r["account"].startswith("1010"))
        naive_delta = abs(int(pay["amount_minor"]) - int(cash["debit_minor"]))
        self.assertNotEqual(naive_delta, partial["expected_amount_minor"],
                            "без fx.csv дельта считается неверно")


class TestScenarioDataIntegrity(unittest.TestCase):
    def test_declared_effects_match_open_records(self):
        files = generate(SEED, BOTH_CFG)
        manifest, ledger, payments = parse(files)
        invoices = [json.loads(l) for l in files["open"]["invoices.jsonl"].splitlines() if l]
        prov = json.loads(files["closed"]["provenance.json"])
        conv, _, _, _ = fx_conv(files, manifest)
        start, end = manifest["period_start"], manifest["period_end"]

        imbalance = sum(conv(r["debit_minor"], r["currency"])
                        - conv(r["credit_minor"], r["currency"]) for r in ledger)
        ar_balance = sum(conv(r["debit_minor"], r["currency"])
                         - conv(r["credit_minor"], r["currency"])
                         for r in ledger if r["account"].startswith("1200")
                         and start <= r["timestamp"] <= end)
        invoiced = sum(conv(i["total_minor"], i["currency"]) for i in invoices
                       if start <= i["issued_at"] <= end)
        paid = sum(conv(p["amount_minor"], p["currency"]) for p in payments
                   if start <= p["timestamp"] <= end)

        self.assertEqual(imbalance,
                         sum(int(m["effect_imbalance"]) for m in prov["mutations"]),
                         "баланс != заявленные эффекты")
        self.assertEqual(ar_balance - (invoiced - paid),
                         sum(int(m["effect_ar_diff"]) for m in prov["mutations"]),
                         "AR-расхождение != заявленные эффекты")
        self.assertTrue(prov["stages"][-1]["declared_discrepancies"]["verified"])
        # сценарии видны в провенансе
        scen = [m for m in prov["mutations"] if m["issue_id"].startswith(("iss_s1", "iss_s2"))]
        self.assertEqual(len(scen), 4)

    def test_expected_amounts_recompute_from_open_records(self):
        files = generate(SEED, BOTH_CFG)
        manifest, ledger, payments = parse(files)
        truth = json.loads(files["closed"]["truth.json"])["issues"]
        conv, policy_day, _, _ = fx_conv(files, manifest)
        base = manifest["base_currency"]

        for issue in truth:
            if issue["type"] == "period_cutoff":
                pid = next(x for x in issue["affected_ids"] if x.startswith("PAY-"))
                pay = next(p for p in payments if p["payment_id"] == pid)
                self.assertEqual(issue["expected_amount_minor"], int(pay["amount_minor"]))
            elif issue["type"] == "fx_rate":
                pid = next(x for x in issue["affected_ids"] if x.startswith("PAY-"))
                pay = next(p for p in payments if p["payment_id"] == pid)
                ar = next(r for r in ledger if r["payment_id"] == pid
                          and r["account"].startswith("1200"))
                expected = conv(pay["amount_minor"], pay["currency"], ar["currency"])
                self.assertEqual(issue["expected_amount_minor"],
                                 abs(int(ar["credit_minor"]) - expected))
                self.assertEqual(issue["currency"], base)
            elif issue["type"] == "partial_allocation":
                pid = next(x for x in issue["affected_ids"] if x.startswith("PAY-"))
                pay = next(p for p in payments if p["payment_id"] == pid)
                cash = next(r for r in ledger if r["payment_id"] == pid
                            and r["account"].startswith("1010"))
                ar = next(r for r in ledger if r["payment_id"] == pid
                          and r["account"].startswith("1200"))
                if cash["currency"] != pay["currency"]:
                    expected = conv(pay["amount_minor"], pay["currency"], cash["currency"])
                    self.assertEqual(issue["expected_amount_minor"],
                                     abs(expected - int(cash["debit_minor"])))
                else:
                    self.assertEqual(issue["expected_amount_minor"],
                                     int(pay["amount_minor"]) - int(ar["credit_minor"]))

    def test_scenario_evidence_resolves_in_open_files(self):
        files = generate(SEED, BOTH_CFG)
        manifest, ledger, payments = parse(files)
        invoices = {i["invoice_id"] for i in
                    (json.loads(l) for l in files["open"]["invoices.jsonl"].splitlines() if l)}
        fx_keys = {(r["date"], r["source_currency"], r["target_currency"]) for r in
                   csv.DictReader(io.StringIO(files["open"]["fx.csv"]))}
        policy = json.loads(files["open"]["policy.json"])
        row_ids = {r["row_id"] for r in ledger}
        pay_ids = {p["payment_id"] for p in payments}
        truth = json.loads(files["closed"]["truth.json"])["issues"]

        for issue in truth:
            if not issue["issue_id"].startswith(("iss_s1", "iss_s2")):
                continue
            for s in issue["evidence_sets"]:
                self.assertTrue(s, issue["issue_id"])
                for ev in s:
                    prefix, _, value = ev.partition(":")
                    if prefix == "ledger":
                        self.assertIn(value, row_ids, ev)
                    elif prefix == "invoice":
                        self.assertIn(value, invoices, ev)
                    elif prefix == "payment":
                        self.assertIn(value, pay_ids, ev)
                    elif prefix == "fx":
                        d, a, b = value.split(":")
                        self.assertIn((d, a, b), fx_keys, ev)
                    elif prefix == "policy":
                        self.assertIn("fx_rate_date" if "fx" in value else "period_boundary",
                                      policy, ev)
                    else:
                        self.fail(f"неизвестный префикс {ev}")


if __name__ == "__main__":
    unittest.main()
