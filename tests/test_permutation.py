"""Парные проверки (раздел 06 спеки): перестановка порядка и смена ID.

Требование: закрытые проверки включают 20 парных тестов, меняющих порядок и ID;
ответ оракула и награда обязаны сохраниться с учётом новых ID.

Здесь же — независимая проверка данных и эталона ПОСЛЕ преобразования
(равенства и заявленные расхождения обязаны остаться верными) и проверка
вариативности часовых поясов.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_env import (  # noqa: E402
    AuditEnv,
    generate,
    permute_episode,
    remap_submission,
    save_episode,
)
from audit_env.policies import OraclePolicy, ReconciliationPolicy, run_episode  # noqa: E402
from support import SEED, SMALL  # noqa: E402
from test_data import fx_conv  # noqa: E402

FORBIDDEN = ["iss_", "issue_id", "expected_amount_minor", "provenance", "causal_change",
             "invariants_", "truth.json", "seed", "statement"]
FOUR_TYPES = ["duplicate_entry", "period_cutoff", "partial_allocation", "fx_rate"]
CFG4 = {"n_invoices": 25, "issue_types": FOUR_TYPES, "n_issues": 4}


def _run_in_dir(dirpath: str, policy_factory, seed: int = 1) -> tuple[dict, AuditEnv]:
    env = AuditEnv()
    env.reset(seed=seed, config={"episode_dir": dirpath})
    policy = policy_factory(env)
    result = run_episode(env, policy, seed=seed, config={"episode_dir": dirpath})
    return result, env


class TestPermutationInvariant(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="perm_"))
        self.files = generate(SEED, CFG4)
        self.perm, self.id_map = permute_episode(self.files, seed=4242)
        save_episode(self.files, self.tmp / "orig")
        save_episode(self.perm, self.tmp / "perm")
        self.orig_dir = str(self.tmp / "orig")
        self.perm_dir = str(self.tmp / "perm")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- свойства преобразования ------------------------------------------ #
    def test_transform_changes_order_and_ids(self):
        orig_rows = list(csv.DictReader(io.StringIO(self.files["open"]["ledger.csv"])))
        perm_rows = list(csv.DictReader(io.StringIO(self.perm["open"]["ledger.csv"])))
        self.assertEqual(len(orig_rows), len(perm_rows), "состав строк не меняется")
        self.assertNotEqual([r["row_id"] for r in orig_rows],
                            [r["row_id"] for r in perm_rows], "id обязаны смениться")
        self.assertNotEqual(
            [(r["timestamp"], r["debit_minor"], r["credit_minor"]) for r in orig_rows],
            [(r["timestamp"], r["debit_minor"], r["credit_minor"]) for r in perm_rows],
            "порядок строк обязан измениться")
        old_manifest = json.loads(self.files["open"]["manifest.json"])
        new_manifest = json.loads(self.perm["open"]["manifest.json"])
        self.assertNotEqual(old_manifest["episode_id"], new_manifest["episode_id"])
        self.assertEqual(old_manifest["n_records"], new_manifest["n_records"])

    def test_transform_is_deterministic(self):
        again, id_map2 = permute_episode(self.files, seed=4242)
        self.assertEqual(again["open"], self.perm["open"])
        self.assertEqual(again["closed"], self.perm["closed"])
        self.assertEqual(id_map2, self.id_map)

    def test_amounts_dates_and_currencies_are_untouched(self):
        def fingerprint(files):
            rows = list(csv.DictReader(io.StringIO(files["open"]["ledger.csv"])))
            return sorted((r["timestamp"], r["account"], r["debit_minor"],
                           r["credit_minor"], r["currency"]) for r in rows)

        self.assertEqual(fingerprint(self.files), fingerprint(self.perm),
                         "безвредное преобразование не меняет суммы/даты/валюты")

    def test_no_old_ids_survive_in_open_files(self):
        blob = json.dumps(self.perm["open"], ensure_ascii=False)
        old_row_ids = [r["row_id"] for r in
                       csv.DictReader(io.StringIO(self.files["open"]["ledger.csv"]))]
        for old in old_row_ids[:10]:
            self.assertNotIn(f'"{old}"', blob, f"старый id {old} остался в открытых файлах")

    # -- данные и эталон после преобразования ------------------------------ #
    def test_declared_discrepancies_still_hold_after_permutation(self):
        """Независимый пересчёт в базовой валюте == эффекты мутаций (после смены ID)."""
        manifest = json.loads(self.perm["open"]["manifest.json"])
        ledger = list(csv.DictReader(io.StringIO(self.perm["open"]["ledger.csv"])))
        invoices = [json.loads(l) for l in self.perm["open"]["invoices.jsonl"].splitlines() if l]
        payments = [json.loads(l) for l in self.perm["open"]["payments.jsonl"].splitlines() if l]
        truth = json.loads(self.perm["closed"]["truth.json"])
        prov = json.loads(self.perm["closed"]["provenance.json"])
        conv, _, _, _ = fx_conv(self.perm, manifest)

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

        declared_i = sum(int(m["effect_imbalance"]) for m in prov["mutations"])
        declared_ar = sum(int(m["effect_ar_diff"]) for m in prov["mutations"])
        self.assertEqual(imbalance, declared_i, "баланс после перестановки разошёлся")
        self.assertEqual(ar_balance - (invoiced - paid), declared_ar,
                         "AR-расхождение после перестановки разошлось")
        self.assertEqual(len(truth["issues"]), len(prov["mutations"]))

    def test_truth_ids_exist_in_permuted_open_files(self):
        """Эталон обязан ссылаться только на существующие записи после смены ID."""
        manifest = json.loads(self.perm["open"]["manifest.json"])
        rows = {r["row_id"] for r in
                csv.DictReader(io.StringIO(self.perm["open"]["ledger.csv"]))}
        invoices = {i["invoice_id"] for i in
                    (json.loads(l) for l in self.perm["open"]["invoices.jsonl"].splitlines() if l)}
        payments = {p["payment_id"] for p in
                    (json.loads(l) for l in self.perm["open"]["payments.jsonl"].splitlines() if l)}
        fx_keys = {(r["date"], r["source_currency"], r["target_currency"]) for r in
                   csv.DictReader(io.StringIO(self.perm["open"]["fx.csv"]))}
        truth = json.loads(self.perm["closed"]["truth.json"])
        self.assertEqual(truth["episode_id"], manifest["episode_id"])
        for issue in truth["issues"]:
            for x in issue["affected_ids"]:
                self.assertTrue(x in rows or x in payments, f"висячий id {x}")
            for s in issue["evidence_sets"]:
                for ev in s:
                    prefix, _, value = ev.partition(":")
                    if prefix == "ledger":
                        self.assertIn(value, rows, ev)
                    elif prefix == "invoice":
                        self.assertIn(value, invoices, ev)
                    elif prefix == "payment":
                        self.assertIn(value, payments, ev)
                    elif prefix == "fx":
                        d, a, b = value.split(":")
                        self.assertIn((d, a, b), fx_keys, ev)

    # -- ответ оракула и политики ------------------------------------------ #
    def test_oracle_answer_survives_id_renaming(self):
        """Находки, построенные на оригинале, перенесённые на новые ID, дают ту же награду."""
        orig, _ = _run_in_dir(self.orig_dir, lambda e: OraclePolicy(e.evaluator_truth()))
        submission = orig["trajectory"][-1]["action"]
        remapped = remap_submission(submission, self.id_map)

        env = AuditEnv()
        env.reset(seed=1, config={"episode_dir": self.perm_dir})
        obs, reward, terminated, truncated, _ = env.step(remapped)
        report = env.evaluator_report()

        self.assertTrue(obs.get("result", {}).get("accepted"))
        self.assertTrue(terminated and not truncated)
        self.assertEqual(report["metrics"]["f1"], orig["report"]["metrics"]["f1"])
        self.assertEqual(report["metrics"]["f1"], 1.0)
        self.assertAlmostEqual(reward, orig["report"]["true_reward"], places=12,
                               msg="награда обязана сохраниться с учётом новых id")

    def test_reconciliation_generalizes_to_permuted_episode(self):
        result, _ = _run_in_dir(self.perm_dir, lambda e: ReconciliationPolicy())
        m = result["report"]["metrics"]
        self.assertEqual((m["fp"], m["fn"]), (0, 0),
                         f"сверка должна работать после перестановки и смены id: {m}")
        self.assertEqual(m["f1"], 1.0)
        self.assertGreater(result["report"]["true_reward"], 0.9)

    def test_no_leak_in_permuted_trajectory(self):
        result, _ = _run_in_dir(self.perm_dir, lambda e: ReconciliationPolicy())
        blob = json.dumps(result["trajectory"], ensure_ascii=False)
        for token in FORBIDDEN:
            self.assertNotIn(token, blob, f"утечка {token!r} в перестановленной траектории")


class TestTimeZoneAxis(unittest.TestCase):
    """Ось «часовой пояс»: смещение варьируется, равенства сохраняются."""

    def test_offsets_are_varying_and_consistent(self):
        for offset in ("+03:00", "-07:00", "+09:30"):
            cfg = {"n_invoices": 12, "utc_offset": offset, "n_issues": 2,
                   "issue_types": ["duplicate_entry", "partial_allocation"]}
            files = generate(7, cfg)
            manifest = json.loads(files["open"]["manifest.json"])
            prov = json.loads(files["closed"]["provenance.json"])
            self.assertEqual(manifest["utc_offset"], offset)
            self.assertTrue(manifest["period_start"].endswith(offset), manifest["period_start"])
            self.assertTrue(manifest["period_end"].endswith(offset), manifest["period_end"])
            self.assertTrue(prov["invariants_before"]["books_balanced"])
            self.assertTrue(prov["invariants_before"]["ar_matches_documents"])
            self.assertTrue(prov["stages"][-1]["declared_discrepancies"]["verified"])
            ledger = list(csv.DictReader(io.StringIO(files["open"]["ledger.csv"])))
            for row in ledger[:5]:
                self.assertTrue(row["timestamp"].endswith(offset),
                                f"{row['timestamp']} != {offset}")

    def test_reconciliation_works_on_foreign_offset(self):
        cfg = {"n_invoices": 25, "utc_offset": "-07:00", "n_issues": 4,
               "issue_types": FOUR_TYPES}
        files = generate(9, cfg)
        tmp = Path(tempfile.mkdtemp(prefix="tz_"))
        try:
            save_episode(files, tmp / "ep")
            env = AuditEnv()
            env.reset(seed=1, config={"episode_dir": str(tmp / "ep")})
            result = run_episode(env, ReconciliationPolicy(), seed=1,
                                 config={"episode_dir": str(tmp / "ep")})
            m = result["report"]["metrics"]
            self.assertEqual((m["fp"], m["fn"]), (0, 0), m)
            self.assertEqual(m["f1"], 1.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
