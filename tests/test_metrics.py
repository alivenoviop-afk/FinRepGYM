"""Matcher, metrics edge cases and the reward formula (spec 05)."""

from __future__ import annotations

import unittest

from support import SEED, AuditEnv, compute_metrics, match, terminal_reward

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

    def test_finding_order_does_not_change_metrics_or_reward(self):
        """Порядок находок в submit не влияет на результат (спека: порядок)."""
        cfg = {"n_invoices": 25,
               "issue_types": ["duplicate_entry", "period_cutoff",
                               "partial_allocation", "fx_rate"],
               "n_issues": 4}

        def run(order):
            env = AuditEnv()
            env.reset(seed=SEED, config=cfg)
            truth = env.evaluator_truth()
            findings = [{"type": i["type"], "affected_ids": sorted(i["affected_ids"]),
                         "claimed_amount_minor": i["expected_amount_minor"],
                         "currency": i["currency"],
                         "evidence_ids": list(i["evidence_sets"][0]),
                         "explanation": "x"} for i in truth]
            selected = [findings[k] for k in order]
            _, reward, terminated, *_ = env.step(
                {"action": "submit", "episode_id": env._manifest["episode_id"],
                 "findings": selected})
            m = env.evaluator_report()["metrics"]
            return (m["tp"], m["fp"], m["fn"], round(m["f1"], 9),
                    round(reward, 9), terminated)

        probe = AuditEnv()
        probe.reset(seed=SEED, config=cfg)
        n = len(probe.evaluator_truth())
        self.assertEqual(n, 4, "нужны все четыре типа")

        straight = run(list(range(n)))
        self.assertEqual(straight, run(list(range(n - 1, -1, -1))), "обратный порядок")
        self.assertEqual(straight, run([2, 0, 3, 1]), "перемешанный порядок")


FOUR_TYPES_CFG = {"n_invoices": 25,
                  "issue_types": ["duplicate_entry", "period_cutoff",
                                  "partial_allocation", "fx_rate"],
                  "n_issues": 4}


def correct_findings(truth: list[dict]) -> list[dict]:
    """Ответ, собранный из эталона (только для тестов награды)."""
    return [{"type": i["type"], "affected_ids": sorted(i["affected_ids"]),
             "claimed_amount_minor": i["expected_amount_minor"],
             "currency": i["currency"],
             "evidence_ids": list(i["evidence_sets"][0]),
             "explanation": "x"} for i in truth]


def submit_once(findings: list[dict], cfg: dict = FOUR_TYPES_CFG, seed: int = SEED):
    env = AuditEnv()
    env.reset(seed=seed, config=cfg)
    _, reward, terminated, truncated, _ = env.step(
        {"action": "submit", "episode_id": env._manifest["episode_id"],
         "findings": findings})
    return reward, env.evaluator_report()["metrics"], terminated, truncated


class TestRewardMonotonicity(unittest.TestCase):
    """Награда монотонна по качеству ответа: по этой среде можно обучать агента.

    Проверяется не «оракул получает 1», а форма самого сигнала: каждая верная
    находка награду РАСТИТ, каждая ложная или недоказанная — снижает.
    """

    @classmethod
    def setUpClass(cls):
        env = AuditEnv()
        env.reset(seed=SEED, config=FOUR_TYPES_CFG)
        cls.truth = env.evaluator_truth()

    def setUp(self):
        self.assertEqual(len(self.truth), 4, "нужны все четыре типа ошибок")
        self.findings = correct_findings(self.truth)

    def test_reward_grows_with_each_correct_finding(self):
        rewards = []
        for k in range(len(self.findings) + 1):
            reward, m, terminated, truncated = submit_once(self.findings[:k])
            self.assertTrue(terminated and not truncated)
            self.assertEqual((m["tp"], m["fp"], m["fn"]),
                             (k, 0, len(self.truth) - k))
            rewards.append(reward)
        for k, (before, after) in enumerate(zip(rewards, rewards[1:]), start=1):
            self.assertLess(before, after,
                            f"награда не растёт при добавлении {k}-й верной находки")
        self.assertLess(rewards[0], 0.0, "пустой ответ на грязных книгах — штраф")
        self.assertGreater(rewards[-1], 0.9, "полный верный ответ почти +1")

    def test_false_finding_lowers_reward(self):
        full, full_m, *_ = submit_once(self.findings)
        other = ("period_cutoff" if self.findings[0]["type"] != "period_cutoff"
                 else "duplicate_entry")
        bogus = dict(self.findings[0], type=other)   # те же ID, неверный тип
        reward, m, *_ = submit_once(self.findings + [bogus])
        self.assertEqual((m["tp"], m["fp"]), (full_m["tp"], 1))
        self.assertLess(reward, full, "ложная находка обязана снижать награду")

    def test_weakened_evidence_lowers_reward(self):
        full, *_ = submit_once(self.findings)
        weak = [dict(self.findings[0],
                     evidence_ids=self.findings[0]["evidence_ids"][:1]),
                *self.findings[1:]]
        reward, m, *_ = submit_once(weak)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (3, 1, 1))
        self.assertLess(reward, full, "неполное доказательство обязано снижать награду")

    def test_spamming_duplicates_does_not_beat_the_honest_answer(self):
        full, *_ = submit_once(self.findings)
        spam, m, *_ = submit_once(self.findings * 3)
        self.assertEqual(m["tp"], len(self.truth), "сопоставление взаимно однозначное")
        self.assertLess(spam, full, "дубликаты не должны давать выгоды")

    def test_every_reward_stays_inside_bounds(self):
        for findings in ([], self.findings[:1], self.findings, self.findings * 3):
            reward, *_ = submit_once(findings)
            self.assertGreaterEqual(reward, -1.0)
            self.assertLessEqual(reward, 1.0)


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
