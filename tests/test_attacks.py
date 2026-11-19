"""Attacking answers: the five required comparisons plus guesses and bad evidence.

Required by spec section 05:

* a complete correct answer with evidence beats silence on erroneous books;
* silence beats false accusations on clean books;
* duplicates gain nothing;
* flagging every row/type scores badly;
* saving steps barely separates equal answers.
"""

from __future__ import annotations

import json
import unittest

from support import (
    CLEAN,
    SEED,
    SMALL,
    AuditEnv,
    MarkAllPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    ReconciliationPolicy,
    SilencePolicy,
    run_episode,
    run_with,
)


def reward_of(result):
    return result["report"]["true_reward"]


def metrics_of(result):
    return result["report"]["metrics"]


class TestRequiredComparisons(unittest.TestCase):
    def test_correct_answer_beats_silence_on_erroneous_books(self):
        correct = run_with(lambda e: OraclePolicy(e.evaluator_truth()))
        silence = run_with(lambda e: SilencePolicy())
        self.assertGreater(reward_of(correct), reward_of(silence))
        self.assertEqual(metrics_of(correct)["f1"], 1.0)
        self.assertEqual(metrics_of(silence)["f1"], 0.0)

    def test_silence_beats_false_accusations_on_clean_books(self):
        silence = run_with(lambda e: SilencePolicy(), config=CLEAN)
        accusations = run_with(lambda e: MarkAllPolicy(), config=CLEAN)
        self.assertEqual(metrics_of(silence)["f1"], 1.0)
        self.assertGreater(reward_of(silence), reward_of(accusations))
        self.assertEqual(metrics_of(accusations)["f1"], 0.0)
        self.assertEqual(reward_of(accusations), -1.0)

    def test_duplicates_gain_nothing(self):
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        truth = env.evaluator_truth()

        def oracle_with_copy(extra):
            findings = [{
                "type": i["type"], "affected_ids": sorted(i["affected_ids"]),
                "claimed_amount_minor": i["expected_amount_minor"], "currency": i["currency"],
                "evidence_ids": list(i["evidence_sets"][0]), "explanation": "x",
            } for i in truth]
            if extra:
                findings.append(dict(findings[0]))
            return findings

        env.reset(seed=SEED, config=SMALL)
        _, single_reward, *_ = env.step({"action": "submit",
                                         "episode_id": env._manifest["episode_id"],
                                         "findings": oracle_with_copy(False)})
        env.reset(seed=SEED, config=SMALL)
        _, dupe_reward, *_ = env.step({"action": "submit",
                                       "episode_id": env._manifest["episode_id"],
                                       "findings": oracle_with_copy(True)})
        self.assertLess(dupe_reward, single_reward)

    def test_mark_all_is_bad(self):
        correct = run_with(lambda e: OraclePolicy(e.evaluator_truth()))
        attack = run_with(lambda e: MarkAllPolicy())
        m = metrics_of(attack)
        self.assertGreaterEqual(m["fp"], 1)
        self.assertLess(reward_of(attack), reward_of(correct))

    def test_step_savings_only_slightly_separate_equal_answers(self):
        class DelayedOracle(OraclePolicy):
            def __init__(self, truth, delay):
                super().__init__(truth)
                self.delay = delay
                self.waited = 0

            def act(self, obs):
                if self.waited < self.delay:
                    self.waited += 1
                    return {"action": "calculator", "expression": "1 + 1"}
                return super().act(obs)

        fast = run_with(lambda e: OraclePolicy(e.evaluator_truth()))
        slow = run_with(lambda e: DelayedOracle(e.evaluator_truth(), delay=40))
        delta = reward_of(fast) - reward_of(slow)
        self.assertAlmostEqual(delta, 0.002 * (slow["steps"] - fast["steps"]), places=9)
        self.assertLess(delta, 0.2)  # равные ответы почти не различаются


class TestBadEvidenceAndGuesses(unittest.TestCase):
    def test_guessed_ids_score_nothing(self):
        result = run_with(lambda e: RandomExplorerPolicy(seed=7))
        self.assertEqual(metrics_of(result)["tp"], 0)
        self.assertEqual(metrics_of(result)["f1"], 0.0)

    def test_unverified_evidence_scores_nothing(self):
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        issue = env.evaluator_truth()[0]
        finding = {"type": issue["type"], "affected_ids": issue["affected_ids"],
                   "claimed_amount_minor": issue["expected_amount_minor"],
                   "currency": issue["currency"], "evidence_ids": [],
                   "explanation": "sounds convincing"}
        _, reward, *_ = env.step({"action": "submit", "episode_id": env._manifest["episode_id"],
                                  "findings": [finding]})
        self.assertLess(reward, 0.0)

    def test_wrong_currency_scores_nothing(self):
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        issue = env.evaluator_truth()[0]
        finding = {"type": issue["type"], "affected_ids": issue["affected_ids"],
                   "claimed_amount_minor": issue["expected_amount_minor"],
                   "currency": "EUR", "evidence_ids": issue["evidence_sets"][0],
                   "explanation": "wrong currency"}
        _, reward, *_ = env.step({"action": "submit", "episode_id": env._manifest["episode_id"],
                                  "findings": [finding]})
        self.assertLess(reward, 0.0)

    def test_resubmission_is_rejected(self):
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        from support import ContractViolation
        env.step({"action": "submit", "episode_id": env._manifest["episode_id"], "findings": []})
        with self.assertRaises(ContractViolation):
            env.step({"action": "submit", "episode_id": env._manifest["episode_id"],
                      "findings": []})


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_actions_same_rewards(self):
        one = run_with(lambda e: ReconciliationPolicy())
        two = run_with(lambda e: ReconciliationPolicy())
        self.assertEqual(
            json.dumps(one["trajectory"], sort_keys=True, ensure_ascii=False),
            json.dumps(two["trajectory"], sort_keys=True, ensure_ascii=False))
        self.assertEqual(reward_of(one), reward_of(two))

    def test_reconciliation_finds_every_issue_without_false_positives(self):
        """Data sufficiency: the baseline proves each issue from open records only."""
        result = run_with(lambda e: ReconciliationPolicy())
        m = metrics_of(result)
        self.assertEqual(m["fp"], 0, f"false positives on distractors: {m}")
        self.assertEqual(m["fn"], 0, f"missed issues: {m}")
        self.assertEqual(m["f1"], 1.0)

    def test_reconciliation_stays_silent_on_clean_books(self):
        result = run_with(lambda e: ReconciliationPolicy(), config=CLEAN)
        m = metrics_of(result)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 0, 0))
        self.assertEqual(m["f1"], 1.0)

    def test_reconciliation_finds_all_four_implemented_types(self):
        config = {"n_invoices": 25,
                  "issue_types": ["duplicate_entry", "period_cutoff",
                                  "partial_allocation", "fx_rate"],
                  "n_issues": 4}
        result = run_with(lambda e: ReconciliationPolicy(), config=config)
        m = metrics_of(result)
        self.assertEqual(m["n_truth"], 4, "должны быть все четыре типа")
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (4, 0, 0),
                         f"сверка должна найти все ошибки без ложных: {m}")

    def test_partial_and_cutoff_are_not_confused(self):
        """Частичное распределение и сдвиг периода живут на разных платежах."""
        config = dict(SMALL)
        config.update({"issue_types": ["period_cutoff", "partial_allocation"],
                       "n_issues": 2})
        result = run_with(lambda e: ReconciliationPolicy(), config=config)
        m = metrics_of(result)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (2, 0, 0), m)
        self.assertEqual(m["n_truth"], 2)


if __name__ == "__main__":
    unittest.main()
