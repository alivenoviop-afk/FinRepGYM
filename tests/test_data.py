"""Data and ground truth: the self-check the jury re-computes independently.

Corresponds to the scoring block «Данные и эталон»: books balance *before* the
mutation, the declared discrepancy is exactly what the records show *after*, and
every evidence id is resolvable from the open files alone.
"""

from __future__ import annotations

import csv
import io
import json
import re
import unittest

from support import AuditEnv, SEED, SMALL, generate

ISO_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
DECIMAL = re.compile(r"^\d+\.\d+$")

OPEN_NAMES = ["manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
              "fx.csv", "policy.json"]


def open_data(seed=SEED, config=None):
    files = generate(seed, dict(config or SMALL))
    manifest = json.loads(files["open"]["manifest.json"])
    ledger = list(csv.DictReader(io.StringIO(files["open"]["ledger.csv"])))
    invoices = [json.loads(x) for x in files["open"]["invoices.jsonl"].splitlines() if x]
    payments = [json.loads(x) for x in files["open"]["payments.jsonl"].splitlines() if x]
    truth = json.loads(files["closed"]["truth.json"])["issues"]
    provenance = json.loads(files["closed"]["provenance.json"])
    return files, manifest, ledger, invoices, payments, truth, provenance


class TestStructure(unittest.TestCase):
    def test_all_open_files_present_and_counts_match(self):
        files, manifest, ledger, invoices, payments, *_ = open_data()
        for name in OPEN_NAMES:
            self.assertIn(name, files["open"], name)
        self.assertEqual(manifest["n_records"], len(ledger))
        self.assertEqual(manifest["n_documents"], len(invoices) + len(payments))
        self.assertEqual(manifest["schema_version"], "audit-rl/1.0")
        self.assertIn(manifest["base_currency"], {"USD", "EUR", "KZT"})

    def test_closed_files_are_not_in_the_open_set(self):
        files, *_ = open_data()
        self.assertNotIn("truth.json", files["open"])
        self.assertNotIn("provenance.json", files["open"])
        self.assertEqual(set(files["closed"]), {"truth.json", "provenance.json"})

    def test_formats(self):
        _, manifest, ledger, invoices, payments, truth, _ = open_data()
        for row in ledger[:20]:
            self.assertRegex(row["timestamp"], ISO_OFFSET)
            self.assertTrue(row["debit_minor"].lstrip("-").isdigit())
            self.assertTrue(row["credit_minor"].lstrip("-").isdigit())
        for inv in invoices:
            self.assertIsInstance(inv["total_minor"], int)
            self.assertRegex(inv["issued_at"], ISO_OFFSET)
            self.assertEqual(sum(li["unit_price_minor"] for li in inv["line_items"]),
                             inv["total_minor"])
        for pay in payments:
            self.assertIsInstance(pay["amount_minor"], int)
        for issue in truth:
            self.assertIsInstance(issue["expected_amount_minor"], int)
            self.assertIn(issue["type"],
                          ("duplicate_entry", "partial_allocation", "fx_rate", "period_cutoff"))

    def test_missing_references_are_explicit_null(self):
        files, *_ = open_data()
        nulls = 0
        for line in files["open"]["ledger.csv"].splitlines()[1:]:
            row = next(csv.reader([line]))
            invoice_id, payment_id = row[7], row[8]
            self.assertNotEqual(invoice_id, "", "invoice_id must never be an empty string")
            self.assertTrue(payment_id == "null" or payment_id.startswith("PAY-"),
                            f"payment_id must be explicit null or a real id: {payment_id!r}")
            nulls += payment_id == "null"
        self.assertGreater(nulls, 0, "sale rows must carry an explicit null payment_id")

    def test_returns_are_legal_and_never_flagged(self):
        """Возвраты — обязательный отвлекающий случай: пары строк, не в truth."""
        files, _, ledger, invoices, payments, truth, provenance = open_data()
        return_rows = [r for r in ledger if r["account"].startswith("4000")
                       and int(r["debit_minor"]) > 0]
        self.assertTrue(return_rows, "возвраты (отвлекающие случаи) не сгенерированы")
        invoice_ids = {i["invoice_id"] for i in invoices}
        truth_rows = {x for issue in truth for x in issue["affected_ids"]}
        for row in return_rows:
            self.assertEqual(row["payment_id"], "null", "возврат — не платёж")
            self.assertIn(row["invoice_id"], invoice_ids, "возврат привязан к счёту")
            self.assertNotIn(row["row_id"], truth_rows, "возврат не является ошибкой")
        # возврат не ломает баланс и дебиторку
        self.assertTrue(provenance["invariants_before"]["books_balanced"])
        self.assertTrue(provenance["invariants_before"]["ar_matches_documents"])
        # встречается и в эпизодах с ошибками, и в чистых
        _, _, _, _, _, clean_truth, clean_prov = open_data(
            config={"n_invoices": 12, "clean": True, "n_issues": 0})
        self.assertEqual(clean_truth, [])
        self.assertTrue(clean_prov["invariants_after"]["books_balanced"])

    def test_generation_is_deterministic(self):
        a = generate(SEED, SMALL)
        b = generate(SEED, SMALL)
        self.assertEqual(a["open"], b["open"])
        self.assertEqual(a["closed"], b["closed"])


class TestCleanHistoryThenDeclaredDiscrepancy(unittest.TestCase):
    def test_books_balance_before_the_mutation(self):
        *_, provenance = open_data()
        before = provenance["invariants_before"]
        self.assertTrue(before["books_balanced"])
        self.assertTrue(before["ar_matches_documents"])
        self.assertEqual(provenance["stages"][0]["stage"], "history")

    def test_declared_discrepancy_matches_open_records(self):
        """Independent recomputation from ledger.csv + documents + truth.json."""
        _, manifest, ledger, invoices, payments, truth, provenance = open_data()
        start, end = manifest["period_start"], manifest["period_end"]

        imbalance = (sum(int(r["debit_minor"]) for r in ledger)
                     - sum(int(r["credit_minor"]) for r in ledger))
        ar_balance = sum(int(r["debit_minor"]) - int(r["credit_minor"]) for r in ledger
                         if r["account"].startswith("1200") and start <= r["timestamp"] <= end)
        invoiced = sum(int(i["total_minor"]) for i in invoices if start <= i["issued_at"] <= end)
        paid = sum(int(p["amount_minor"]) for p in payments if start <= p["timestamp"] <= end)
        ar_diff = ar_balance - (invoiced - paid)

        dup_total = sum(i["expected_amount_minor"] for i in truth
                        if i["type"] == "duplicate_entry")
        cutoff_total = sum(i["expected_amount_minor"] for i in truth
                           if i["type"] == "period_cutoff")
        partial_total = sum(i["expected_amount_minor"] for i in truth
                            if i["type"] == "partial_allocation")

        self.assertEqual(imbalance, -dup_total, "books imbalance != declared duplicates")
        self.assertEqual(ar_diff, cutoff_total + partial_total,
                         "in-period AR discrepancy != cutoffs + partial allocations")
        self.assertTrue(provenance["stages"][-1]["declared_discrepancies"]["verified"])

    def test_mutation_is_recorded_with_cause_and_effect(self):
        *_, truth, provenance = open_data()
        mutations = provenance["mutations"]
        self.assertEqual(len(mutations), len(truth))
        for issue, mutation in zip(truth, mutations):
            self.assertEqual(issue["issue_id"], mutation["issue_id"])
            self.assertTrue(mutation["causal_change"])
            self.assertIn("before", mutation)
            self.assertIn("after", mutation)
            self.assertNotEqual(mutation["before"], mutation["after"])

    def test_period_cutoff_rows_really_left_the_period(self):
        _, manifest, ledger, _invoices, payments, truth, _ = open_data()
        start, end = manifest["period_start"], manifest["period_end"]
        cutoff = next(i for i in truth if i["type"] == "period_cutoff")
        payment_id = next(x for x in cutoff["affected_ids"] if x.startswith("PAY-"))
        pay = next(p for p in payments if p["payment_id"] == payment_id)
        self.assertTrue(start <= pay["timestamp"] <= end, "payment document must stay in period")
        rows = [r for r in ledger if r["payment_id"] == payment_id]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row["timestamp"] > end, "ledger row must move past period_end")


class TestEvidenceSufficiency(unittest.TestCase):
    def test_every_evidence_id_resolves_in_open_files(self):
        _, manifest, ledger, invoices, payments, truth, _ = open_data()
        row_ids = {r["row_id"] for r in ledger}
        invoice_ids = {i["invoice_id"] for i in invoices}
        payment_ids = {p["payment_id"] for p in payments}
        policy = json.loads(generate(SEED, SMALL)["open"]["policy.json"])

        self.assertIn("period_boundary", policy)
        for issue in truth:
            self.assertTrue(issue["evidence_sets"], f"{issue['issue_id']} has no evidence set")
            for evidence_set in issue["evidence_sets"]:
                self.assertTrue(evidence_set, "empty evidence set")
                for ev in evidence_set:
                    kind, _, value = ev.partition(":")
                    if kind == "ledger":
                        self.assertIn(value, row_ids, ev)
                    elif kind == "invoice":
                        self.assertIn(value, invoice_ids, ev)
                    elif kind == "payment":
                        self.assertIn(value, payment_ids, ev)
                    elif kind == "policy":
                        self.assertIn(value.replace("period", "period_boundary").replace(
                            "balance", "rounding"), policy, ev)
                    else:
                        self.fail(f"unknown evidence prefix: {ev}")

    def test_expected_amount_equals_the_cited_record(self):
        _, _, ledger, _, payments, truth, _ = open_data()
        by_row = {r["row_id"]: r for r in ledger}
        for issue in truth:
            if issue["type"] == "duplicate_entry":
                row = by_row[issue["affected_ids"][0]]
                cited = int(row["debit_minor"]) or int(row["credit_minor"])
                self.assertEqual(issue["expected_amount_minor"], cited)
            elif issue["type"] == "period_cutoff":
                pid = next(x for x in issue["affected_ids"] if x.startswith("PAY-"))
                pay = next(p for p in payments if p["payment_id"] == pid)
                self.assertEqual(issue["expected_amount_minor"], int(pay["amount_minor"]))
            elif issue["type"] == "partial_allocation":
                pid = next(x for x in issue["affected_ids"] if x.startswith("PAY-"))
                pay = next(p for p in payments if p["payment_id"] == pid)
                rows = [r for r in ledger if r["payment_id"] == pid]
                allocated = sum(int(r["credit_minor"]) for r in rows
                                if r["account"].startswith("1200"))
                self.assertEqual(issue["expected_amount_minor"],
                                 int(pay["amount_minor"]) - allocated,
                                 "документ платежа минус его распределение = заявленное расхождение")


class TestScope(unittest.TestCase):
    def test_episode_size_within_spec(self):
        # default config (n_invoices=80) is the shipping size: 200-2000 rows, 20-200 docs
        _, manifest, _, invoices, payments, truth, _ = open_data(config={"n_invoices": 80})
        self.assertGreaterEqual(manifest["n_records"], 200)
        self.assertLessEqual(manifest["n_records"], 2000)
        self.assertGreaterEqual(len(invoices) + len(payments), 20)
        self.assertLessEqual(len(invoices) + len(payments), 200)
        self.assertLessEqual(len(truth), 12)

    def test_issue_count_between_zero_and_twelve(self):
        for config in ({"n_invoices": 6, "clean": True, "n_issues": 0}, SMALL):
            _, _, _, _, _, truth, _ = open_data(config=config)
            self.assertTrue(0 <= len(truth) <= 12)

    def test_clean_episode_has_no_issues_and_balanced_books(self):
        _, _, _, _, _, truth, provenance = open_data(config={"n_invoices": 10, "clean": True,
                                                             "n_issues": 0})
        self.assertEqual(truth, [])
        self.assertTrue(provenance["invariants_after"]["books_balanced"])
        self.assertTrue(provenance["invariants_after"]["ar_matches_documents"])

    def test_closed_env_needs_truth(self):
        env = AuditEnv()
        with self.assertRaises(Exception):
            env.reset(seed=SEED, config={"episode_dir": "/nonexistent"})


if __name__ == "__main__":
    unittest.main()
