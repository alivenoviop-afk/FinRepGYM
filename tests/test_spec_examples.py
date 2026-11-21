"""Открытые примеры M01-M12 из определений проверок (точные ожидаемые результаты).

Общая фикстура определений: ошибка A, затронутые {R1, R2}, сумма 10000 KZT,
допустимый набор {D1, D2}, D3 — существующее дополнительное доказательство,
U отсутствует, tau = 0. Сопоставление идёт через compute_metrics с явным
множеством существующих evidence-id {"D1", "D2", "D3"}.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_env import compute_metrics, match, validate_submission  # noqa: E402

EXISTING = {"D1", "D2", "D3"}          # D3 существует, U — нет
A = {"issue_id": "iss_a", "type": "duplicate_entry",
     "affected_ids": ["R1", "R2"], "expected_amount_minor": 10000,
     "currency": "KZT", "amount_tolerance_minor": 0,
     "evidence_sets": [["D1", "D2"]]}


def finding(**kw):
    base = {"type": "duplicate_entry", "affected_ids": ["R1", "R2"],
            "claimed_amount_minor": 10000, "currency": "KZT",
            "evidence_ids": ["D1", "D2"], "explanation": "ok"}
    base.update(kw)
    return base


def metrics(truth, findings):
    return compute_metrics(truth, findings, EXISTING)


class TestOpenExamples(unittest.TestCase):
    # M01: повтор ID внутри списка не равен повторному finding
    def test_m01_repeated_id_in_list_is_a_set(self):
        m = metrics([A], [finding(affected_ids=["R2", "R1", "R1"])])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 0))

    # M02: существующий, но незатронутый R3 -> FP=1, FN=1
    def test_m02_extra_existing_record_is_fp(self):
        m = metrics([A], [finding(affected_ids=["R1", "R2", "R3"])])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))

    # M03: полный набор + существующее дополнительное доказательство -> TP
    def test_m03_extra_existing_evidence_still_matches(self):
        m = metrics([A], [finding(evidence_ids=["D2", "D1", "D3"])])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 0))

    # M04: несуществующий U -> не сопоставляется, хотя набор присутствует
    def test_m04_nonexistent_evidence_is_fp(self):
        m = metrics([A], [finding(evidence_ids=["D1", "D2", "U"])])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))

    # M05: 11 элементов (10 повторов D1) -> структурная ошибка ДО удаления повторов
    def test_m05_eleven_evidence_ids_is_structural(self):
        ok, code, _ = validate_submission(
            {"episode_id": "ep_x",
             "findings": [finding(evidence_ids=["D1"] * 10 + ["D2"])]}, "ep_x")
        self.assertFalse(ok)
        self.assertIn("too_many_evidence_ids", code)

    # M06: регистр/пробел валюты -> точное сравнение, FP=1, FN=1
    def test_m06_currency_case_is_exact(self):
        for bad in ("kzt", "KZT ", " kzt"):
            m = metrics([A], [finding(currency=bad)])
            self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1), bad)

    # M07: amount=10001 при tau=0 -> FP=1, FN=1
    def test_m07_amount_off_by_one_at_zero_tolerance(self):
        m = metrics([A], [finding(claimed_amount_minor=10001)])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))

    # M08: отдельная фикстура с tau=1: 10001 проходит, 10002 — нет
    def test_m08_boundary_tolerance_is_inclusive(self):
        tolerant = [dict(A, amount_tolerance_minor=1)]
        self.assertEqual(metrics(tolerant,
                                 [finding(claimed_amount_minor=10001)])["tp"], 1)
        self.assertEqual(metrics(tolerant,
                                 [finding(claimed_amount_minor=10002)])["tp"], 0)

    # M09: строка или bool вместо суммы -> структурная ошибка, reward=-1
    def test_m09_amount_string_or_bool_is_structural(self):
        for bad in ("10000", True, 10000.0):
            ok, code, _ = validate_submission(
                {"episode_id": "ep_x",
                 "findings": [finding(claimed_amount_minor=bad)]}, "ep_x")
            self.assertFalse(ok, repr(bad))
            self.assertIn("bad_structure", code)

    # M10: F0 подходит A и B, F1 только A -> F1->A, F0->B, TP=2
    def test_m10_greedy_first_fit_is_not_final(self):
        # A принимает полный набор {D1,D2} и дополнительный {D3}; B — только {D1,D2}
        a = dict(A, evidence_sets=[["D1", "D2"], ["D3"]])
        b = dict(A, issue_id="iss_b")
        truth = [a, b]
        f0 = finding(evidence_ids=["D1", "D2"])   # подходит обоим
        f1 = finding(evidence_ids=["D3"])         # только A
        m = metrics(truth, [f0, f1])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (2, 0, 0))
        # жадный F0->A оставил бы F1 без пары; правильное назначение:
        # F1 (индекс 1) -> A (iss_a), F0 (индекс 0) -> B (iss_b)
        pairs = match(truth, [f0, f1], EXISTING)["tp_pairs"]
        self.assertEqual(sorted(pairs), [(0, 1), (1, 0)])

    # M11: F0 и F1 подходят обеим -> при A<B детерминированно A->F0, B->F1
    def test_m11_both_fit_both_is_deterministic(self):
        b = dict(A, issue_id="iss_b")
        truth = [A, b]
        f0 = finding(evidence_ids=["D1", "D2"])
        f1 = finding(evidence_ids=["D1", "D2", "D3"])
        m = metrics(truth, [f0, f1])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (2, 0, 0))
        # A (iss_a < iss_b) берёт наименьший индекс вывода: A->F0, B->F1
        pairs = match(truth, [f0, f1], EXISTING)["tp_pairs"]
        self.assertEqual(sorted(pairs), [(0, 0), (1, 1)])
        # перестановка выводов не меняет TP и награду
        m2 = metrics(truth, [f1, f0])
        self.assertEqual((m2["tp"], m2["fp"], m2["fn"]), (2, 0, 0))
        self.assertEqual(m2["f1"], m["f1"])

    # M12: два полностью одинаковых правильных вывода -> TP=1, FP=1
    def test_m12_identical_duplicates_give_one_tp(self):
        m = metrics([A], [finding(), finding()])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
