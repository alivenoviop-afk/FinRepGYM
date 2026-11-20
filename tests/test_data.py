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
from decimal import Decimal, ROUND_HALF_UP

from support import AuditEnv, ReconciliationPolicy, SCHEMA_VERSION, SEED, SMALL, generate, run_with

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


def fx_conv(files, manifest):
    """Независимая конвертация в тесте: только открытые fx.csv + policy-дата."""
    policy_day = manifest["period_end"][:10]
    base = manifest["base_currency"]
    rows = list(csv.DictReader(io.StringIO(files["open"]["fx.csv"])))
    rates = {(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"])
             for r in rows}

    def conv(amount, src, tgt=None):
        tgt = tgt or base
        if src == tgt:
            return int(amount)
        rate = rates[(policy_day, src, tgt)]
        return int((Decimal(int(amount)) * rate).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    return conv, policy_day, base, rates


class TestStructure(unittest.TestCase):
    def test_all_open_files_present_and_counts_match(self):
        files, manifest, ledger, invoices, payments, *_ = open_data()
        for name in OPEN_NAMES:
            self.assertIn(name, files["open"], name)
        self.assertEqual(manifest["n_records"], len(ledger))
        self.assertEqual(manifest["n_documents"], len(invoices) + len(payments))
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
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
        """Независимый пересчёт в базовой валюте (по fx.csv) == сумма эффектов мутаций."""
        files, manifest, ledger, invoices, payments, truth, provenance = open_data()
        start, end = manifest["period_start"], manifest["period_end"]
        conv, _, _, _ = fx_conv(files, manifest)

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
        ar_diff = ar_balance - (invoiced - paid)

        mutations = provenance["mutations"]
        declared_imbalance = sum(int(m["effect_imbalance"]) for m in mutations)
        declared_ar = sum(int(m["effect_ar_diff"]) for m in mutations)

        self.assertEqual(imbalance, declared_imbalance,
                         "наблюдаемый баланс (база) != сумме заявленных эффектов")
        self.assertEqual(ar_diff, declared_ar,
                         "наблюдаемое AR-расхождение (база) != сумме заявленных эффектов")
        self.assertTrue(provenance["stages"][-1]["declared_discrepancies"]["verified"])
        self.assertTrue(provenance["invariants_before"]["books_balanced"])
        self.assertTrue(provenance["invariants_before"]["ar_matches_documents"])
        # каждая эталонная ошибка имеет свою причинную запись и наоборот
        self.assertEqual({m["type"] for m in mutations}, {t["type"] for t in truth})
        for m in mutations:
            self.assertIn("effect_imbalance", m)
            self.assertIn("effect_ar_diff", m)

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
        files, manifest, ledger, invoices, payments, truth, _ = open_data()
        row_ids = {r["row_id"] for r in ledger}
        invoice_ids = {i["invoice_id"] for i in invoices}
        payment_ids = {p["payment_id"] for p in payments}
        policy = json.loads(files["open"]["policy.json"])
        fx_keys = {(r["date"], r["source_currency"], r["target_currency"])
                   for r in csv.DictReader(io.StringIO(files["open"]["fx.csv"]))}

        self.assertIn("period_boundary", policy)
        self.assertIn("fx_rate_date", policy)
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
                    elif kind == "fx":
                        day, src, tgt = value.split(":")
                        self.assertIn((day, src, tgt), fx_keys, ev)
                    elif kind == "policy":
                        self.assertIn(value.replace("period", "period_boundary").replace(
                            "balance", "rounding"), policy, ev)
                    else:
                        self.fail(f"unknown evidence prefix: {ev}")

    def test_expected_amount_equals_the_cited_record(self):
        files, manifest, ledger, _, payments, truth, _ = open_data()
        conv, _, _, _ = fx_conv(files, manifest)
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
            elif issue["type"] == "fx_rate":
                pid = next(x for x in issue["affected_ids"] if x.startswith("PAY-"))
                pay = next(p for p in payments if p["payment_id"] == pid)
                ar_row = next(r for r in ledger if r["payment_id"] == pid
                              and r["account"].startswith("1200"))
                expected = conv(pay["amount_minor"], pay["currency"], ar_row["currency"])
                self.assertEqual(issue["expected_amount_minor"],
                                 abs(int(ar_row["credit_minor"]) - expected),
                                 "расхождение = |журнал − документ × policy-курс|")


class TestMultiCurrency(unittest.TestCase):
    CONFIG = {"n_invoices": 40}

    def test_each_currency_balanced_separately_on_clean(self):
        """Мультивалюта сходится ПО КАЖДОЙ валюте, а не «в среднем»."""
        for seed in (1, 5, 9):
            files = generate(seed, {"n_invoices": 70, "clean": True, "n_issues": 0})
            per: dict[str, list[int]] = {}
            for row in csv.DictReader(io.StringIO(files["open"]["ledger.csv"])):
                acc = per.setdefault(row["currency"], [0, 0])
                acc[0] += int(row["debit_minor"])
                acc[1] += int(row["credit_minor"])
            self.assertGreaterEqual(len(per), 2, f"seed={seed}: валюты не разнообразны")
            for ccy, (debit, credit) in per.items():
                self.assertEqual(debit, credit,
                                 f"seed={seed}: валюта {ccy} не сбалансирована "
                                 f"({debit} != {credit})")

    def test_documents_use_several_currencies(self):
        """Спека требует разные валюты; конвертация обязательна для сравнения."""
        _, manifest, ledger, invoices, payments, *_ = open_data(config=self.CONFIG)
        used = {i["currency"] for i in invoices} | {p["currency"] for p in payments}
        self.assertGreaterEqual(len(used), 2, "в эпизоде должны использоваться разные валюты")
        self.assertEqual(set(manifest["currencies"]), {"USD", "EUR", "KZT"})
        self.assertEqual(manifest["base_currency"], "USD")
        led_ccy = {r["currency"] for r in ledger}
        self.assertTrue(led_ccy <= set(manifest["currencies"]),
                        "валюта записи должна быть заявлена в манифесте")

    def test_cross_currency_payment_posts_in_base(self):
        """Документ в одной валюте, строки постинга в базовой: без fx.csv не сравнить."""
        _, manifest, ledger, _, payments, *_ = open_data(config=self.CONFIG)
        base = manifest["base_currency"]
        docs = {p["payment_id"]: p for p in payments}
        rows_by_pay: dict[str, list[dict]] = {}
        for r in ledger:
            if r["payment_id"] and r["payment_id"] != "null":
                rows_by_pay.setdefault(r["payment_id"], []).append(r)
        cross = [pid for pid, rows in rows_by_pay.items()
                 if docs[pid]["currency"] != rows[0]["currency"]]
        self.assertTrue(cross, "кросс-валютные платежи обязательны (сценарий спеки)")
        for pid in cross:
            rows = rows_by_pay[pid]
            self.assertTrue(all(r["currency"] == base for r in rows),
                            "кросс-валютный платёж постится в базовой валюте")

    def test_fx_rates_cover_policy_date_for_every_pair(self):
        files, manifest, *_ = open_data(config=self.CONFIG)
        policy_day = manifest["period_end"][:10]
        rows = list(csv.DictReader(io.StringIO(files["open"]["fx.csv"])))
        for a in manifest["currencies"]:
            for b in manifest["currencies"]:
                if a == b:
                    continue
                hit = any(r["date"] == policy_day and r["source_currency"] == a
                          and r["target_currency"] == b for r in rows)
                self.assertTrue(hit, f"нет курса {a}->{b} на policy-дату {policy_day}")
        for r in rows:
            self.assertRegex(r["rate"], r"^\d+\.\d+$", "курс — десятичная строка")

    def test_fx_error_is_recomputed_from_open_records(self):
        config = {"n_invoices": 25, "issue_types": ["fx_rate"], "n_issues": 1}
        files, manifest, ledger, _, payments, truth, prov = open_data(config=config)
        self.assertEqual([t["type"] for t in truth], ["fx_rate"])
        conv, _, _, _ = fx_conv(files, manifest)
        issue = truth[0]
        pid = next(x for x in issue["affected_ids"] if x.startswith("PAY-"))
        pay = next(p for p in payments if p["payment_id"] == pid)
        ar_row = next(r for r in ledger if r["payment_id"] == pid
                      and r["account"].startswith("1200"))
        # журнал отличается от документа, приведённого по policy-курсу
        self.assertNotEqual(int(ar_row["credit_minor"]),
                            conv(pay["amount_minor"], pay["currency"], ar_row["currency"]))
        self.assertEqual(issue["currency"], manifest["base_currency"])
        self.assertTrue(provenance_ok(prov))


def provenance_ok(provenance: dict) -> bool:
    return bool(provenance["stages"][-1]["declared_discrepancies"]["verified"])


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


class TestBoundaryDates(unittest.TestCase):
    """Пограничные даты: счёт/платёж в первый день и в последний момент периода.

    Спека: policy.json объявляет границы включительно — данные обязаны эти
    границы реально затрагивать, иначе проверка включительности ничего не весит.
    """

    def test_boundary_timestamps_present_across_seeds(self):
        end_found = day0_found = False
        for seed in range(1, 13):
            files = generate(seed, {"n_invoices": 60})
            man = json.loads(files["open"]["manifest.json"])
            period_end = man["period_end"]
            day0 = man["period_start"][:10]
            stamps = {r["timestamp"] for r in
                      csv.DictReader(io.StringIO(files["open"]["ledger.csv"]))}
            end_found |= any(s == period_end for s in stamps)
            day0_found |= any(s.startswith(day0) for s in stamps)
            if end_found and day0_found:
                break
        self.assertTrue(end_found, "нет операции в последний момент периода (23:59:59)")
        self.assertTrue(day0_found, "нет операций в первый день периода")

    def test_boundary_does_not_create_false_cutoffs(self):
        """Операция РОВНО на границе должна считаться внутри периода (inclusive)."""
        for seed in (1, 2, 3, 4, 5, 6):
            result = run_with(lambda e: ReconciliationPolicy(), seed, {"n_invoices": 60})
            m = result["report"]["metrics"]
            self.assertEqual((m["fp"], m["fn"]), (0, 0),
                             f"seed={seed}: ложный сдвиг периода на границе {m}")


class TestConfigCombos(unittest.TestCase):
    """Новые сочетания параметров, которые может собрать жюри (спека 06)."""

    def test_clean_dominates_over_scenarios(self):
        files = generate(SEED, {"n_invoices": 30, "clean": True, "n_issues": 0,
                                "scenarios": ["S1", "S2"]})
        self.assertEqual(json.loads(files["closed"]["truth.json"])["issues"], [],
                         "clean=true должен главенствовать: чистый эпизод пуст")

    def test_fx_type_works_when_distractors_disabled(self):
        """Без этого ядро падало бы на валидном сочетании: fx требовал кросс-платежа,
        а кросс-платежи были заперты внутри distractors."""
        cfg = {"n_invoices": 25, "issue_types": ["fx_rate"], "n_issues": 1,
               "distractors": False}
        truth = json.loads(generate(SEED, cfg)["closed"]["truth.json"])["issues"]
        self.assertEqual([t["type"] for t in truth], ["fx_rate"])

    def test_scenarios_work_when_distractors_disabled(self):
        cfg = {"n_invoices": 30, "issue_types": [], "n_issues": 0,
               "scenarios": ["S1", "S2"], "distractors": False}
        truth = json.loads(generate(SEED, cfg)["closed"]["truth.json"])["issues"]
        self.assertEqual(len(truth), 4)

    def test_single_currency_contradiction_is_reported_clearly(self):
        from audit_env.generator import EpisodeError
        with self.assertRaises(EpisodeError) as ctx:
            generate(SEED, {"n_invoices": 25, "issue_types": ["fx_rate"], "n_issues": 1,
                            "currencies": ["USD"]})
        self.assertIn("2 валюты", str(ctx.exception))
        # а чистый эпизод с одной валютой обязан работать
        files = generate(SEED, {"n_invoices": 60, "clean": True, "n_issues": 0,
                                "currencies": ["USD"]})
        self.assertEqual(json.loads(files["closed"]["truth.json"])["issues"], [])


if __name__ == "__main__":
    unittest.main()
