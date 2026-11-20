"""Независимая ДОСТАТОЧНОСТЬ доказательств (спека, раздел 10).

Проверяем не просто «доказательство разрешимо в открытых файлах», а то, что
**по каждому полному набору доказательств можно вычислить заявленную сумму
и валюту** — то есть набор действительно достаточен для верификации находки.

Для КАЖДОГО эталонного набора берём ТОЛЬКО записи, на которые он ссылается
(плюс открытые правила: период из manifest и курсы из fx.csv — это часть
открытого контекста, а не скрытая информация), считаем расхождение своим
кодом и сверяем с truth.expected_amount_minor / truth.currency.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import unittest
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_env import generate  # noqa: E402
from support import SEED  # noqa: E402

FOUR_TYPES = {"n_invoices": 25,
              "issue_types": ["duplicate_entry", "period_cutoff",
                              "partial_allocation", "fx_rate"],
              "n_issues": 4}
SCENARIOS = {"n_invoices": 30, "issue_types": [], "n_issues": 0,
             "scenarios": ["S1", "S2"]}
DEFAULT = {"n_invoices": 60}

CONFIGS = [("default", DEFAULT, SEED), ("four_types", FOUR_TYPES, 4242),
           ("scenarios", SCENARIOS, 777), ("default_2", DEFAULT, 99)]


def load(cfg, seed):
    files = generate(seed, cfg)
    manifest = json.loads(files["open"]["manifest.json"])
    ledger = {r["row_id"]: r for r in
              csv.DictReader(io.StringIO(files["open"]["ledger.csv"]))}
    invoices = {i["invoice_id"]: i for i in
                (json.loads(l) for l in files["open"]["invoices.jsonl"].splitlines() if l)}
    payments = {p["payment_id"]: p for p in
                (json.loads(l) for l in files["open"]["payments.jsonl"].splitlines() if l)}
    fx = {(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"])
          for r in csv.DictReader(io.StringIO(files["open"]["fx.csv"]))}
    truth = json.loads(files["closed"]["truth.json"])["issues"]
    return {"manifest": manifest, "ledger": ledger, "invoices": invoices,
            "payments": payments, "fx": fx, "truth": truth}


def convert(ctx, amount, src, tgt):
    if src == tgt:
        return int(amount)
    day = ctx["manifest"]["period_end"][:10]
    rate = ctx["fx"][(day, src, tgt)]
    return int((Decimal(int(amount)) * rate).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def ids_by_prefix(evidence_set, prefix):
    return [e.split(":", 1)[1] for e in evidence_set if e.startswith(prefix + ":")]


def compute_from_set(ctx, issue, evidence_set):
    """Считает заявленную сумму/валюту ТОЛЬКО по записям из набора (+ открытые правила)."""
    start = ctx["manifest"]["period_start"]
    end = ctx["manifest"]["period_end"]
    rows = {rid: ctx["ledger"][rid] for rid in ids_by_prefix(evidence_set, "ledger")}
    pay_ids = ids_by_prefix(evidence_set, "payment")
    inv_ids = ids_by_prefix(evidence_set, "invoice")
    pay = ctx["payments"][pay_ids[0]] if pay_ids else None
    t = issue["type"]

    if t == "duplicate_entry":
        assert len(rows) >= 2, "набор должен содержать обе строки дубля"
        a, b = list(rows.values())[:2]
        for field in ("timestamp", "account", "debit_minor", "credit_minor",
                      "currency", "entity_id", "invoice_id", "payment_id"):
            assert a[field] == b[field], f"строки не идентичны по {field}"
        amount = int(a["credit_minor"]) or int(a["debit_minor"])
        assert amount > 0
        if inv_ids:  # если набор ссылается на счёт — его сумма должна подтверждать
            assert int(ctx["invoices"][inv_ids[0]]["total_minor"]) == amount
            assert ctx["invoices"][inv_ids[0]]["currency"] == a["currency"]
        return amount, a["currency"]

    if t == "period_cutoff":
        assert pay is not None and rows, "нужны платёж и строка журнала"
        assert start <= pay["timestamp"] <= end, "документ должен быть внутри периода"
        row = next(iter(rows.values()))
        assert not (start <= row["timestamp"] <= end), "строка должна быть вне периода"
        return int(pay["amount_minor"]), pay["currency"]

    if t == "partial_allocation":
        assert pay is not None and rows, "нужны платёж и строка журнала"
        ar = next((r for r in rows.values() if r["account"].startswith("1200")), None)
        cash = next((r for r in rows.values() if r["account"].startswith("1010")), None)
        assert ar is not None or cash is not None, "нужна строка дебеторки или кассы"
        value_row = ar if ar is not None else cash
        doc = int(pay["amount_minor"])
        if pay["currency"] != value_row["currency"]:
            doc = convert(ctx, doc, pay["currency"], value_row["currency"])
        recorded = (int(ar["credit_minor"]) if ar is not None
                    else int(cash["debit_minor"]))
        return abs(doc - recorded), value_row["currency"]

    if t == "fx_rate":
        assert pay is not None and rows, "нужны платёж и строка дебеторки"
        ar = next((r for r in rows.values() if r["account"].startswith("1200")), None)
        assert ar is not None, "нужна строка дебеторки"
        doc = convert(ctx, int(pay["amount_minor"]), pay["currency"], ar["currency"])
        return abs(int(ar["credit_minor"]) - doc), ar["currency"]

    raise AssertionError(f"неизвестный тип {t}")


class TestEvidenceSufficiency(unittest.TestCase):
    def test_every_full_set_suffices_to_recompute_the_amount(self):
        checked_sets = 0
        for name, cfg, seed in CONFIGS:
            with self.subTest(config=name):
                ctx = load(cfg, seed)
                for issue in ctx["truth"]:
                    self.assertLessEqual(len(issue["evidence_sets"]), 10)
                    self.assertTrue(issue["evidence_sets"], issue["issue_id"])
                    for evidence_set in issue["evidence_sets"]:
                        self.assertTrue(evidence_set, issue["issue_id"])
                        amount, currency = compute_from_set(ctx, issue, evidence_set)
                        self.assertEqual(
                            amount, issue["expected_amount_minor"],
                            f"{name}/{issue['issue_id']}/{evidence_set}: "
                            f"считано {amount}, в truth {issue['expected_amount_minor']}")
                        self.assertEqual(
                            currency, issue["currency"],
                            f"{name}/{issue['issue_id']}/{evidence_set}: валюта не совпала")
                        checked_sets += 1
        self.assertGreaterEqual(checked_sets, 20, "наборов проверено слишком мало")

    def test_multiple_seeds_cover_all_four_types(self):
        """Каждый из 4 типов встречается в наборах и проходит достаточность."""
        seen = set()
        for name, cfg, seed in CONFIGS:
            for issue in load(cfg, seed)["truth"]:
                seen.add(issue["type"])
        self.assertEqual(seen, {"duplicate_entry", "period_cutoff",
                                "partial_allocation", "fx_rate"})


if __name__ == "__main__":
    unittest.main()
