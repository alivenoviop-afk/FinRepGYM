"""Тесты judge_adapter.py — точки входа оценщика (определения 20.11.2026).

Проверяется канонический протокол: конверт действий, форма наблюдений,
пагинация, закрытый режим (reward=0), score_episode с точными числами
формулы награды и структурными ошибками (31 находка -> return=-1).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # judge_adapter.py лежит в корне
sys.path.insert(0, str(ROOT / "src"))

import judge_adapter as ja  # noqa: E402
from audit_env import terminal_reward  # noqa: E402

SEED = 4242
CONFIG = {"n_invoices": 25}


class AdapterTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="judge_"))
        self.episode_dir = self.tmp / "ep"
        ja.generate_episode(SEED, dict(CONFIG), str(self.episode_dir))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestGenerateEpisode(AdapterTestCase):
    def test_writes_open_and_closed_files(self):
        for name in ("manifest.json", "ledger.csv", "invoices.jsonl",
                     "payments.jsonl", "fx.csv", "policy.json"):
            self.assertTrue((self.episode_dir / "open" / name).exists(), name)
        for name in ("truth.json", "provenance.json"):
            self.assertTrue((self.episode_dir / "closed" / name).exists(), name)

    def test_manifest_has_required_fields(self):
        man = json.loads((self.episode_dir / "open" / "manifest.json").read_text("utf-8"))
        for key in ("schema_version", "episode_id", "base_currency",
                    "period_start", "period_end", "timezone",
                    "n_records", "n_documents", "rounding"):
            self.assertIn(key, man)
        self.assertEqual(man["schema_version"], ja.SCHEMA_VERSION)


class TestMakeEnv(AdapterTestCase):
    def _closed_env(self):
        env = ja.make_env()
        obs, info = env.reset(config={"mode": "closed",
                                      "episode_dir": str(self.episode_dir)})
        return env, obs, info

    def test_first_observation_is_canonical_and_leak_free(self):
        _, obs, info = self._closed_env()
        for key in ("task", "manifest", "policy", "remaining_steps"):
            self.assertIn(key, obs)
        self.assertEqual(obs["remaining_steps"], 100)
        blob = json.dumps(obs, ensure_ascii=False)
        for token in ("truth", "provenance", "seed", "issue_id", "iss_"):
            self.assertNotIn(token, blob)
        self.assertEqual(info["mode"], "eval")

    def test_list_records_envelope_and_pagination(self):
        env, obs, _ = self._closed_env()
        total = obs["manifest"]["n_records"]
        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            obs, reward, term, trunc, info = env.step(
                {"name": "list_records", "args": {"table": "ledger",
                                                  "cursor": cursor, "limit": 50}})
            self.assertIsNone(obs["error"])
            self.assertEqual(reward, 0.0, "в закрытом режиме reward всегда 0")
            self.assertFalse(term or trunc)
            rows = obs["result"]["records"]
            self.assertLessEqual(len(rows), 50)
            seen += [r["row_id"] for r in rows]
            pages += 1
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
            self.assertIsInstance(cursor, str, "next_cursor — строка или null")
            self.assertLess(pages, 100, "пагинация не должна зависнуть")
        self.assertEqual(len(seen), total)
        self.assertEqual(len(set(seen)), total, "без пропусков и повторов")
        self.assertEqual(obs["remaining_steps"], 100 - pages)

    def test_same_page_twice_is_identical(self):
        env, _, _ = self._closed_env()
        first, *_ = env.step({"name": "list_records",
                              "args": {"table": "ledger", "cursor": None, "limit": 50}})
        second, *_ = env.step({"name": "list_records",
                               "args": {"table": "ledger", "cursor": None, "limit": 50}})
        self.assertEqual(first["result"]["records"], second["result"]["records"])
        self.assertEqual(first["result"]["next_cursor"], second["result"]["next_cursor"])

    def test_get_record_and_document_return_objects(self):
        env, obs, _ = self._closed_env()
        page, *_ = env.step({"name": "list_records",
                             "args": {"table": "ledger", "cursor": None, "limit": 1}})
        row_id = page["result"]["records"][0]["row_id"]
        rec, *_ = env.step({"name": "get_record",
                            "args": {"table": "ledger", "id": row_id}})
        self.assertEqual(rec["result"]["row_id"], row_id)

        ipage, *_ = env.step({"name": "list_records",
                              "args": {"table": "invoices", "cursor": None, "limit": 1}})
        doc_id = ipage["result"]["records"][0]["doc_id"]
        doc, *_ = env.step({"name": "get_document", "args": {"doc_id": doc_id}})
        self.assertEqual(doc["result"]["doc_id"], doc_id)

    def test_calculator_returns_decimal_string(self):
        env, _, _ = self._closed_env()
        obs, *_ = env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
        self.assertEqual(obs["result"], "0.3")
        obs2, *_ = env.step({"name": "calculator",
                             "args": {"expression": "open('f')"}})
        self.assertIsNone(obs2["result"])
        self.assertIsNotNone(obs2["error"])

    def test_unknown_action_costs_a_step_and_gives_limited_error(self):
        env, _, _ = self._closed_env()
        before = env._env._steps
        obs, *_ = env.step({"name": "definitely_not_an_action", "args": {}})
        self.assertEqual(obs["error"]["code"], "unknown_action")
        self.assertEqual(env._env._steps, before + 1)

    def test_missing_record_gives_limited_error_and_costs_a_step(self):
        env, _, _ = self._closed_env()
        before = env._env._steps
        obs, *_ = env.step({"name": "get_record",
                            "args": {"table": "ledger", "id": "нет_такого"}})
        self.assertEqual(obs["error"]["code"], "unknown_id")
        self.assertEqual(env._env._steps, before + 1)

    def test_submit_terminates_with_zero_reward_in_closed_mode(self):
        env, _, _ = self._closed_env()
        obs, reward, term, trunc, info = env.step(
            {"name": "submit", "args": {"findings": {"findings": []}}})
        self.assertEqual(obs["result"], {"accepted": True})
        self.assertTrue(term and not trunc)
        self.assertEqual(reward, 0.0, "закрытый режим: reward политики = 0")
        # после завершения среда не падает и не меняет состояние
        obs2, reward2, term2, trunc2, _ = env.step(
            {"name": "list_records", "args": {"table": "ledger", "cursor": None}})
        self.assertEqual(obs2["error"]["code"], "episode_finished")
        self.assertTrue(term2 and not trunc2)
        self.assertEqual(reward2, 0.0)

    def test_train_mode_returns_terminal_reward(self):
        env = ja.make_env()
        env.reset(seed=SEED, config={"mode": "train", "team_config": dict(CONFIG)})
        obs, reward, term, *_ = env.step({"name": "submit",
                                          "args": {"findings": {"findings": []}}})
        self.assertTrue(term)
        # пустой ответ на эпизоде с ошибками -> штраф
        self.assertEqual(reward, -1.0)


class TestScoreEpisode(AdapterTestCase):
    def _oracle_findings(self) -> list[dict]:
        truth = json.loads((self.episode_dir / "closed" / "truth.json").read_text("utf-8"))
        out = []
        for issue in truth["issues"]:
            out.append({"type": issue["type"],
                        "affected_ids": list(issue["affected_ids"]),
                        "claimed_amount_minor": issue["expected_amount_minor"],
                        "currency": issue["currency"],
                        "evidence_ids": list(issue["evidence_sets"][0]),
                        "explanation": "oracle"})
        return out

    def test_valid_submit_returns_exact_metrics_and_reward(self):
        findings = self._oracle_findings()
        report = ja.score_episode(str(self.episode_dir),
                                  {"findings": findings}, 10)
        self.assertEqual((report["tp"], report["fp"], report["fn"]),
                         (len(findings), 0, 0))
        self.assertEqual(report["f1"], 1.0)
        self.assertAlmostEqual(report["return"], terminal_reward(1.0, 10), places=9)
        # формула определений: inspection_steps по умолчанию 10 -> 0.98
        self.assertAlmostEqual(report["return"], 0.98, places=9)

    def test_inspection_steps_shift_the_reward_exactly(self):
        findings = self._oracle_findings()
        r0 = ja.score_episode(str(self.episode_dir), {"findings": findings}, 0)
        r50 = ja.score_episode(str(self.episode_dir), {"findings": findings}, 50)
        r100 = ja.score_episode(str(self.episode_dir), {"findings": findings}, 100)
        self.assertAlmostEqual(r0["return"], 1.0, places=9)
        self.assertAlmostEqual(r50["return"], 0.90, places=9)
        self.assertAlmostEqual(r100["return"], 0.80, places=9)
        self.assertAlmostEqual(r0["return"] - r50["return"], 0.10, places=9)

    def test_empty_truth_and_empty_findings_is_perfect(self):
        clean_dir = self.tmp / "clean"
        ja.generate_episode(SEED + 7, {"n_invoices": 20, "clean": True,
                                       "n_issues": 0}, str(clean_dir))
        report = ja.score_episode(str(clean_dir), {"findings": []}, 10)
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (0, 0, 0))
        self.assertEqual(report["f1"], 1.0)
        self.assertAlmostEqual(report["return"], 0.98, places=9)

    def test_too_many_findings_is_structural_error(self):
        big = [self._oracle_findings()[0]] * 31
        report = ja.score_episode(str(self.episode_dir), {"findings": big}, 10)
        self.assertEqual(report["return"], -1.0)
        for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
            self.assertIsNone(report[key])
        self.assertEqual(ja.evaluator_log()[-1]["event"], "submit_invalid")

    def test_nonexistent_evidence_id_is_fp_not_tp(self):
        """M04: полный набор + несуществующий evidence_id -> не сопоставляется."""
        findings = self._oracle_findings()
        for f in findings:
            f["evidence_ids"] = list(f["evidence_ids"]) + ["ledger:нет_такой_строки"]
        report = ja.score_episode(str(self.episode_dir), {"findings": findings}, 10)
        self.assertEqual(report["tp"], 0)
        self.assertEqual(report["fp"], len(findings))
        self.assertEqual(report["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
