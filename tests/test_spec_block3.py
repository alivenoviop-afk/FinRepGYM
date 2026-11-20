"""Зеркало блока 3 определений от 20.11: 20 тестов matcher и награды (25 баллов).

Фикстура общая для тестов 1-17: одна ошибка A, каноническая запись R1, сумма
10000 минимальных единиц KZT, допустимый набор {D1, D2}, tau = 0; в юнит-тестах
формулы inspection_steps задаётся напрямую (по умолчанию 10). Тесты 18-20 —
структурные лимиты на уровне среды: эпизод завершается с наградой -1.
Ожидаемые числа взяты ДОСЛОВНО из определений (включая 47/150 и -871/1050).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_env import AuditEnv, compute_metrics, generate, save_episode  # noqa: E402
from audit_env.reward import terminal_reward  # noqa: E402

A = {"issue_id": "iss_a", "type": "duplicate_entry", "affected_ids": ["R1"],
     "expected_amount_minor": 10000, "currency": "KZT",
     "amount_tolerance_minor": 0, "evidence_sets": [["D1", "D2"]]}
B = {"issue_id": "iss_b", "type": "duplicate_entry", "affected_ids": ["R7"],
     "expected_amount_minor": 10000, "currency": "KZT",
     "amount_tolerance_minor": 0, "evidence_sets": [["D1", "D3"]]}


def finding(**kw):
    base = {"type": "duplicate_entry", "affected_ids": ["R1"],
            "claimed_amount_minor": 10000, "currency": "KZT",
            "evidence_ids": ["D1", "D2"], "explanation": "ok"}
    base.update(kw)
    return base


class TestBlock3Unit(unittest.TestCase):
    """Тесты 1-17: формулы и сопоставление на синтетической фикстуре."""

    # -- 1 --------------------------------------------------------------- #
    def test_1_correct_finding_is_perfect(self):
        m = compute_metrics([A], [finding()])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 0))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(terminal_reward(m["f1"], 10), 0.98, places=9)

    # -- 2 --------------------------------------------------------------- #
    def test_2_empty_findings_against_truth_is_minus_one(self):
        m = compute_metrics([A], [])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 0, 1))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 3 --------------------------------------------------------------- #
    def test_3_empty_truth_and_empty_findings_is_perfect(self):
        m = compute_metrics([], [])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 0, 0))
        self.assertEqual(m["f1"], 1.0)
        self.assertAlmostEqual(terminal_reward(m["f1"], 10), 0.98, places=9)

    # -- 4 --------------------------------------------------------------- #
    def test_4_false_finding_on_empty_truth_is_minus_one(self):
        m = compute_metrics([], [finding()])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 0))
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 5 --------------------------------------------------------------- #
    def test_5_duplicate_finding_gains_nothing(self):
        m = compute_metrics([A], [finding(), finding()])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))
        self.assertAlmostEqual(m["precision"], 0.5, places=9)
        self.assertAlmostEqual(m["recall"], 1.0, places=9)
        self.assertAlmostEqual(m["f1"], 2 / 3, places=9)
        reward = terminal_reward(m["f1"], 10)
        self.assertAlmostEqual(reward, 47 / 150, places=9)      # ~0.3133333333
        self.assertLess(reward, terminal_reward(1.0, 10))

    # -- 6 --------------------------------------------------------------- #
    def test_6_wrong_type_is_not_a_tp(self):
        m = compute_metrics([A], [finding(type="period_cutoff")])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 7 --------------------------------------------------------------- #
    def test_7_other_existing_record_in_affected_is_not_a_tp(self):
        m = compute_metrics([A], [finding(affected_ids=["R2"])])   # R2 существует
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 8 --------------------------------------------------------------- #
    def test_8_amount_off_by_one_with_zero_tolerance_is_not_a_tp(self):
        m = compute_metrics([A], [finding(claimed_amount_minor=10001)])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 9 --------------------------------------------------------------- #
    def test_9_other_currency_is_not_a_tp(self):
        m = compute_metrics([A], [finding(currency="USD")])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 10 -------------------------------------------------------------- #
    def test_10_partial_evidence_is_not_a_tp(self):
        m = compute_metrics([A], [finding(evidence_ids=["D1"])])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 11 -------------------------------------------------------------- #
    def test_11_nonexistent_evidence_id_is_not_a_tp(self):
        m = compute_metrics([A], [finding(evidence_ids=["D1", "U"])])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))
        self.assertEqual(terminal_reward(m["f1"], 10), -1.0)

    # -- 12 -------------------------------------------------------------- #
    def test_12_mass_marking_attack_is_crushed(self):
        findings = [finding(type=t, affected_ids=[rec])
                    for rec in ("R1", "R2", "R3", "R4", "R5")
                    for t in ("duplicate_entry", "period_cutoff",
                              "partial_allocation", "fx_rate")]
        self.assertEqual(len(findings), 20)
        m = compute_metrics([A], findings)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 19, 0))
        self.assertAlmostEqual(m["precision"], 1 / 20, places=9)
        self.assertAlmostEqual(m["recall"], 1.0, places=9)
        self.assertAlmostEqual(m["f1"], 2 / 21, places=9)
        self.assertAlmostEqual(terminal_reward(m["f1"], 10), -871 / 1050, places=9)

    # -- 13 -------------------------------------------------------------- #
    def test_13_one_of_two_errors_found(self):
        m = compute_metrics([A, B], [finding()])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 1))
        self.assertAlmostEqual(m["precision"], 1.0, places=9)
        self.assertAlmostEqual(m["recall"], 0.5, places=9)
        self.assertAlmostEqual(m["f1"], 2 / 3, places=9)
        self.assertAlmostEqual(terminal_reward(m["f1"], 10), 47 / 150, places=9)

    # -- 14 -------------------------------------------------------------- #
    def test_14_one_true_plus_one_false_finding(self):
        m = compute_metrics([A, B],
                            [finding(), finding(affected_ids=["R9"])])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 1))
        self.assertAlmostEqual(m["precision"], 0.5, places=9)
        self.assertAlmostEqual(m["recall"], 0.5, places=9)
        self.assertAlmostEqual(m["f1"], 0.5, places=9)
        self.assertAlmostEqual(terminal_reward(m["f1"], 10), -0.02, places=9)

    # -- 15 -------------------------------------------------------------- #
    def test_15_zero_inspection_steps_give_full_reward(self):
        m = compute_metrics([A], [finding()])
        self.assertAlmostEqual(terminal_reward(m["f1"], 0), 1.0, places=9)

    # -- 16 -------------------------------------------------------------- #
    def test_16_fifty_steps_cost_exactly_0_08(self):
        r1 = terminal_reward(1.0, 10)
        r50 = terminal_reward(1.0, 50)
        self.assertAlmostEqual(r50, 0.90, places=9)
        self.assertAlmostEqual(r1 - r50, 0.08, places=9)

    # -- 17 -------------------------------------------------------------- #
    def test_17_hundred_steps_give_0_80(self):
        self.assertAlmostEqual(terminal_reward(1.0, 100), 0.80, places=9)


class TestBlock3StructuralLimits(unittest.TestCase):
    """Тесты 18-20: структурные лимиты -> эпизод завершается с reward = -1."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="spec_b3_"))
        cls.episode_dir = cls.tmp / "ep"
        save_episode(generate(880_001, {"n_invoices": 30}),
                     str(cls.episode_dir))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def submit(self, findings):
        env = AuditEnv(mode="train")
        env.reset(config={"episode_dir": str(self.episode_dir)})
        obs, reward, term, trunc, _ = env.step(
            {"action": "submit", "episode_id": env._manifest["episode_id"],
             "findings": findings})
        return obs, reward, term, trunc

    def test_18_thirty_one_findings_rejected(self):
        findings = [finding() for _ in range(31)]
        obs, reward, term, trunc = self.submit(findings)
        self.assertTrue(term and not trunc)
        self.assertEqual(reward, -1.0)
        self.assertEqual(obs["error"]["code"], "submit_invalid")

    def test_19_eleven_evidence_ids_rejected(self):
        finding_ = finding(evidence_ids=[f"D{i}" for i in range(11)])
        obs, reward, term, trunc = self.submit([finding_])
        self.assertTrue(term and not trunc)
        self.assertEqual(reward, -1.0)
        self.assertEqual(obs["error"]["code"], "submit_invalid")

    def test_20_explanation_of_1001_ascii_chars_rejected(self):
        finding_ = finding(explanation="a" * 1001)
        obs, reward, term, trunc = self.submit([finding_])
        self.assertTrue(term and not trunc)
        self.assertEqual(reward, -1.0)
        self.assertEqual(obs["error"]["code"], "submit_invalid")

    def test_boundary_1000_chars_is_accepted(self):
        """Граница лимита: ровно 1000 символов — структурно валидная отправка."""
        finding_ = finding(explanation="a" * 1000)
        obs, reward, term, _ = self.submit([finding_])
        self.assertTrue(term)
        # отправка ПРИНЯТА по структуре (в отличие от 1001 символа);
        # награда -1 законна: synthetic-находка не совпадает с эталоном эпизода
        self.assertIn("result", obs)
        self.assertEqual(obs["result"]["accepted"], True)


if __name__ == "__main__":
    unittest.main()
