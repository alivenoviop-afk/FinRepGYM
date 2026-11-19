"""Contract, isolation, limits, invalid input, mode separation, leakage (spec 04)."""

from __future__ import annotations

import json
import unittest

from support import CLEAN, SEED, SMALL, AuditEnv, ContractViolation, ReconciliationPolicy, run_episode


class TestReset(unittest.TestCase):
    def test_reset_returns_observation_and_info(self):
        env = AuditEnv()
        obs, info = env.reset(seed=SEED, config=SMALL)
        self.assertEqual(set(obs), {"task", "manifest", "policy", "steps_left"})
        self.assertEqual(obs["steps_left"], env.max_steps)
        self.assertEqual(env.max_steps, 100)
        self.assertEqual(env.page_limit, 50)
        self.assertIn("episode_id", obs["manifest"])
        self.assertEqual(obs["manifest"]["schema_version"], "audit-rl/1.0")
        self.assertIn("brief", obs["policy"])
        self.assertIn("period_boundary", obs["policy"])  # policy.json is open
        self.assertEqual(info["steps_used"], 0)

    def test_reset_isolates_episode_state(self):
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        for _ in range(4):
            env.step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 5})
        obs, info = env.reset(seed=SEED + 1, config=SMALL)
        self.assertEqual(obs["steps_left"], 100)
        self.assertEqual(info["steps_used"], 0)
        self.assertNotEqual(obs["manifest"]["episode_id"], "")

    def test_same_seed_same_records(self):
        a = AuditEnv().reset(seed=SEED, config=SMALL)[0]
        b = AuditEnv().reset(seed=SEED, config=SMALL)[0]
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        page1 = env.step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 50})
        env2 = AuditEnv()
        env2.reset(seed=SEED, config=SMALL)
        page2 = env2.step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 50})
        self.assertEqual(page1[0], page2[0])
        self.assertEqual(page1[1], page2[1])

    def test_different_seed_different_episode(self):
        a = AuditEnv().reset(seed=SEED, config=SMALL)[0]["manifest"]["episode_id"]
        b = AuditEnv().reset(seed=SEED + 1, config=SMALL)[0]["manifest"]["episode_id"]
        self.assertNotEqual(a, b)


class TestActions(unittest.TestCase):
    def setUp(self):
        self.env = AuditEnv()
        self.obs, _ = self.env.reset(seed=SEED, config=SMALL)

    def _step(self, action):
        return self.env.step(action)

    def test_pagination_limit_and_stable_cursor(self):
        obs, *_ = self._step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 50})
        self.assertLessEqual(obs["result"]["count"], 50)
        first = obs["result"]["rows"]
        obs2, *_ = self._step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 50})
        self.assertEqual(first, obs2["result"]["rows"])  # stable cursor

    def test_page_over_limit_is_an_error_and_costs_a_step(self):
        before = self.env._steps
        obs, *_ = self._step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 51})
        self.assertEqual(obs["error"]["code"], "limit_too_large")
        self.assertEqual(self.env._steps, before + 1)

    def test_unknown_action_costs_a_step(self):
        obs, reward, term, trunc, info = self._step({"action": "shell", "cmd": "ls"})
        self.assertEqual(obs["error"]["code"], "unknown_action")
        self.assertEqual(info["steps_used"], 1)
        self.assertEqual((reward, term, trunc), (0.0, False, False))

    def test_unknown_id_costs_a_step(self):
        obs, _, _, _, info = self._step({"action": "get_record", "table": "ledger",
                                         "id": "row_does_not_exist"})
        self.assertEqual(obs["error"]["code"], "unknown_id")
        self.assertEqual(info["steps_used"], 1)

    def test_get_document_roundtrip(self):
        page, *_ = self._step({"action": "list_records", "table": "invoices", "cursor": 0, "limit": 1})
        doc_id = page["result"]["rows"][0]["doc_id"]
        obs, *_ = self._step({"action": "get_document", "doc_id": doc_id})
        self.assertEqual(obs["result"]["document"]["doc_id"], doc_id)
        obs2, *_ = self._step({"action": "get_document", "doc_id": "nope"})
        self.assertEqual(obs2["error"]["code"], "unknown_document")

    def test_calculator_accepts_arithmetic_only(self):
        obs, *_ = self._step({"action": "calculator", "expression": "(1.5 + 2.5) * 3"})
        self.assertEqual(obs["result"]["result"], "12.0")
        for bad in ["__import__('os')", "open('x')", "1 if 1 else 2", "2 ** 10", "a + 1", "1/0"]:
            obs, *_ = self._step({"action": "calculator", "expression": bad})
            self.assertIn("error", obs, bad)
            self.assertEqual(obs["error"]["code"], "invalid_expression"
                             if bad != "1/0" else "division_by_zero", bad)

    def test_truncation_at_100_steps_with_empty_final_answer(self):
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        truncated = False
        for i in range(100):
            obs, reward, terminated, truncated, info = env.step(
                {"action": "list_records", "table": "ledger", "cursor": 0, "limit": 10})
            self.assertFalse(terminated)
            if truncated:
                self.assertEqual(i, 99)
                break
        self.assertTrue(truncated)
        report = env.evaluator_report()
        self.assertEqual(report["metrics"]["n_findings"], 0)  # пустой итоговый ответ
        self.assertTrue(report["log"] and report["log"][-1]["event"] == "truncated")

    def test_submit_terminates_and_step_afterwards_is_rejected(self):
        obs, *_ = self._step({"action": "submit", "episode_id": obs_episode(self.env),
                              "findings": []})
        self.assertTrue(obs.get("result", {}).get("accepted"))
        with self.assertRaises(ContractViolation):
            self._step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 1})

    def test_malformed_submit_terminates_with_minus_one(self):
        obs, reward, terminated, _, info = self._step({"action": "submit",
                                                       "episode_id": obs_episode(self.env),
                                                       "findings": [{"type": "duplicate_entry"}]})
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(obs["error"]["code"], "submit_invalid")
        report = self.env.evaluator_report()
        self.assertEqual(report["log"][0]["event"], "submit_invalid")
        self.assertIsNone(report["metrics"])

    def test_wrong_episode_id_is_rejected(self):
        _, reward, terminated, _, _ = self._step({"action": "submit", "episode_id": "ep_lies",
                                                  "findings": []})
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)

    def test_too_many_findings_rejected(self):
        _, reward, terminated, _, _ = self._step({
            "action": "submit", "episode_id": obs_episode(self.env),
            "findings": [self._finding()] * 31})
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)

    def _finding(self):
        return {"type": "duplicate_entry", "affected_ids": ["row_00001"],
                "claimed_amount_minor": 1, "currency": "USD",
                "evidence_ids": ["ledger:row_00001", "ledger:row_00002"], "explanation": "x"}


def obs_episode(env: AuditEnv) -> str:
    if env._manifest:
        return env._manifest["episode_id"]
    raise AssertionError("no manifest")


class TestModes(unittest.TestCase):
    def test_eval_mode_withholds_reward_and_diagnostics(self):
        from support import oracle_run
        result = oracle_run(mode="eval")
        rewards = [row["reward"] for row in result["trajectory"]]
        self.assertTrue(all(r == 0.0 for r in rewards), rewards)
        self.assertGreater(result["report"]["true_reward"], 0.5)

    def test_train_mode_returns_terminal_reward_only(self):
        from support import oracle_run
        result = oracle_run(mode="train")
        rewards = [row["reward"] for row in result["trajectory"]]
        self.assertEqual(rewards[:-1], [0.0] * (len(rewards) - 1))
        self.assertGreater(rewards[-1], 0.5)


class TestNoLeak(unittest.TestCase):
    FORBIDDEN = ["iss_", "issue_id", "expected_amount_minor", "provenance", "causal_change",
                 "invariants_", "truth.json", "seed", "statement"]

    def test_observations_carry_no_closed_truth(self):
        env = AuditEnv()
        env.reset(seed=SEED, config=SMALL)
        policy = ReconciliationPolicy()
        result = run_episode(env, policy, seed=SEED, config=SMALL)
        blob = json.dumps(result["trajectory"], ensure_ascii=False)
        for token in self.FORBIDDEN:
            self.assertNotIn(token, blob, f"leak: {token!r} in trajectory")
        # closed files are not part of the open file set either
        self.assertNotIn("truth", json.dumps(env._manifest, ensure_ascii=False))

    def test_eval_report_is_not_in_info(self):
        env = AuditEnv(mode="eval")
        obs, info = env.reset(seed=SEED, config=SMALL)
        _, _, terminated, _, info2 = env.step(
            {"action": "submit", "episode_id": obs["manifest"]["episode_id"], "findings": []})
        self.assertTrue(terminated)
        self.assertEqual(set(info2), {"mode", "steps_used"})
        self.assertNotIn("true_reward", json.dumps(info2))


if __name__ == "__main__":
    unittest.main()
