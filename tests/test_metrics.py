"""Matcher, metrics edge cases and the reward formula (spec 05)."""

from __future__ import annotations

import unittest

from support import AuditEnv, compute_metrics, match, terminal_reward

TRUTH = [{
    "issue_id": "iss_1",
    "type": "duplicate_entry",
    "affected_ids": ["row_00002", "row_00009"],
    "expected_amount_minor": 1000,
    "currency": "usd",
    "amount_tolerance_minor": 5,
    "evidence_sets": [["ledger:row_00009", "ledger:row_00002", "invoice:INV-1"]],
}]


def finding(**kw):
    base = {
        "type": "duplicate_entry",
        "affected_ids": ["row_00009", "row_00002"],
        "claimed_amount_minor": 1000,
        "currency": "USD",
        "evidence_ids": ["ledger:row_00009", "ledger:row_00002", "invoice:INV-1"],
        "explanation": "d",
    }
    base.update(kw)
    return base


class TestMatching(unittest.TestCase):
    def test_perfect_match(self):
        m = compute_metrics(TRUTH, [finding()])
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 0))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (1.0, 1.0, 1.0))

    def test_id_order_and_currency_case_do_not_matter(self):
        m = compute_metrics(TRUTH, [finding(affected_ids=["row_00002", "row_00009"],
                                            currency="USD".lower())])
        self.assertEqual(m["tp"], 1)

    def test_wrong_amount_is_not_a_tp(self):
        self.assertEqual(compute_metrics(TRUTH, [finding(claimed_amount_minor=1006)])["tp"], 0)
        self.assertEqual(compute_metrics(TRUTH, [finding(claimed_amount_minor=1005)])["tp"], 1)

    def test_wrong_type_is_not_a_tp(self):
        self.assertEqual(compute_metrics(TRUTH, [finding(type="period_cutoff")])["tp"], 0)

    def test_wrong_currency_is_not_a_tp(self):
        self.assertEqual(compute_metrics(TRUTH, [finding(currency="EUR")])["tp"], 0)

    def test_partial_evidence_is_not_enough(self):
        self.assertEqual(
            compute_metrics(TRUTH, [finding(evidence_ids=["ledger:row_00009"])])["tp"], 0)

    def test_missing_evidence_is_not_enough(self):
        self.assertEqual(compute_metrics(TRUTH, [finding(evidence_ids=[])])["tp"], 0)

    def test_guessing_an_id_is_not_enough(self):
        self.assertEqual(compute_metrics(TRUTH, [finding(affected_ids=["row_00042"],
                                                         evidence_ids=["ledger:row_00042"])])["tp"], 0)

    def test_duplicate_submission_gains_nothing(self):
        m = compute_metrics(TRUTH, [finding(), finding()])
        self.assertEqual((m["tp"], m["fp"]), (1, 1))
        self.assertLess(terminal_reward(m["f1"], 4), terminal_reward(1.0, 3))

    def test_matching_is_one_to_one(self):
        two = [dict(TRUTH[0], issue_id="iss_2", affected_ids=["row_00077"])]
        m = match(TRUTH + two, [finding()])
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fn"], 1)


class TestMetricEdgeCases(unittest.TestCase):
    def test_both_empty_is_perfect(self):
        m = compute_metrics([], [])
        self.assertEqual(m["f1"], 1.0)

    def test_empty_truth_with_findings_is_zero(self):
        m = compute_metrics([], [finding()])
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(m["recall"], 0.0)  # нулевой знаменатель -> 0

    def test_empty_findings_with_truth_is_zero(self):
        m = compute_metrics(TRUTH, [])
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(m["precision"], 0.0)  # нулевой знаменатель -> 0
        self.assertEqual(m["recall"], 0.0)

    def test_precision_denominator_zero(self):
        m = compute_metrics([], [])
        self.assertEqual(m["precision"], 0.0)


class TestReward(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(terminal_reward(1.0, 1), 0.998)
        self.assertAlmostEqual(terminal_reward(1.0, 0), 1.0)
        self.assertAlmostEqual(terminal_reward(0.0, 0), -1.0)
        self.assertAlmostEqual(terminal_reward(0.0, 100), -1.0)  # clipped

    def test_clip_bounds(self):
        self.assertEqual(terminal_reward(0.0, 10_000), -1.0)
        self.assertEqual(terminal_reward(1.0, 0), 1.0)

    def test_step_cost_is_small(self):
        cheap = terminal_reward(0.9, 10)
        pricey = terminal_reward(0.9, 60)
        self.assertAlmostEqual(cheap - pricey, 0.002 * 50)
        self.assertLess(cheap - pricey, 0.5)

    def test_reward_ignores_explanation_text(self):
        env = AuditEnv()
        obs, _ = env.reset(seed=1911, config={"n_invoices": 12})
        issue = env.evaluator_truth()[0]
        base = {"type": issue["type"], "affected_ids": issue["affected_ids"],
                "claimed_amount_minor": issue["expected_amount_minor"],
                "currency": issue["currency"], "evidence_ids": issue["evidence_sets"][0]}

        def submit(explanation):
            e = AuditEnv()
            e.reset(seed=1911, config={"n_invoices": 12})
            _, r, term, trunc, _ = e.step({"action": "submit",
                                           "episode_id": e._manifest["episode_id"],
                                           "findings": [dict(base, explanation=explanation)]})
            self.assertTrue(term)
            return r

        self.assertEqual(submit("коротко"), submit("очень длинное и убедительное объяснение" * 10))


if __name__ == "__main__":
    unittest.main()
