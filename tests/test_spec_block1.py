"""Зеркало блока 1 определений проверок от 20.11: C01-C20 (25 баллов).

Каждый тест отвечает строке официального чек-листа жюри и гоняется через
`judge_adapter` — ту самую точку входа, которую будет вызывать оценщик.
Фикстуры: C — чистый эпизод, E — эпизод с заявленной ошибкой, L — таблица
не менее 101 записи (наш конфиг даёт ~350), U — несуществующий идентификатор.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import judge_adapter as ja  # noqa: E402

SEED = 770_001
CLEAN_CFG = {"n_invoices": 70, "clean": True, "n_issues": 0, "distractors": False}
ERROR_CFG = {"n_invoices": 70}          # default issue_types -> есть ошибки
LEAK_TOKENS = ("truth", "provenance", "seed", "iss_", "expected_amount",
               "causal_change", "invariants_", "statement", "evidence_sets")


class Block1Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="spec_b1_"))
        cls.clean_dir = cls.tmp / "C"      # фикстура C
        cls.error_dir = cls.tmp / "E"      # фикстура E
        ja.generate_episode(SEED, dict(CLEAN_CFG), str(cls.clean_dir))
        ja.generate_episode(SEED + 1, dict(ERROR_CFG), str(cls.error_dir))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def closed_env(self, episode_dir, **extra):
        env = ja.make_env()
        config = {"mode": "closed", "episode_dir": str(episode_dir)}
        config.update(extra)
        obs, info = env.reset(config=config)
        return env, obs, info

    # -- C01 ------------------------------------------------------------- #
    def test_c01_reset_returns_json_compatible_pair(self):
        env, obs, info = self.closed_env(self.clean_dir)
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)
        json.dumps(obs)
        json.dumps(info)
        out, reward, term, trunc, step_info = env.step(
            {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 5}})
        json.dumps(out)
        json.dumps(step_info)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)

    # -- C02 ------------------------------------------------------------- #
    def test_c02_first_observation_has_task_manifest_policy_budget_without_secrets(self):
        _, obs, _ = self.closed_env(self.clean_dir)
        for key in ("task", "manifest", "policy", "remaining_steps"):
            self.assertIn(key, obs)
        self.assertEqual(obs["remaining_steps"], 100)
        blob = json.dumps(obs, ensure_ascii=False)
        for token in LEAK_TOKENS:
            self.assertNotIn(token, blob, f"в первом наблюдении найден {token}")

    # -- C03 ------------------------------------------------------------- #
    def test_c03_manifest_matches_actual_data(self):
        _, obs, _ = self.closed_env(self.clean_dir)
        man = obs["manifest"]
        for key in ("schema_version", "episode_id", "base_currency",
                    "period_start", "period_end", "timezone",
                    "n_records", "n_documents", "rounding"):
            self.assertIn(key, man)
        ledger_rows = (self.clean_dir / "open" / "ledger.csv").read_text("utf-8")
        n_ledger = len(ledger_rows.strip().splitlines()) - 1
        n_invoices = len((self.clean_dir / "open" / "invoices.jsonl")
                         .read_text("utf-8").strip().splitlines())
        n_payments = len((self.clean_dir / "open" / "payments.jsonl")
                         .read_text("utf-8").strip().splitlines())
        self.assertEqual(man["n_records"], n_ledger)
        self.assertEqual(man["n_documents"], n_invoices + n_payments)
        self.assertEqual(man["tables"]["ledger"], n_ledger)
        self.assertTrue(man["rounding"])

    # -- C04 ------------------------------------------------------------- #
    def test_c04_step_returns_five_typed_values(self):
        env, _, _ = self.closed_env(self.clean_dir)
        out = env.step({"name": "list_records",
                        "args": {"table": "ledger", "cursor": None, "limit": 5}})
        self.assertEqual(len(out), 5)
        obs, reward, term, trunc, info = out
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        self.assertIn("remaining_steps", obs)
        json.dumps(obs)
        json.dumps({"reward": reward, "info": info})

    # -- C05 ------------------------------------------------------------- #
    def test_c05_pagination_reaches_every_row_without_gaps_or_dupes(self):
        env, obs, _ = self.closed_env(self.clean_dir)   # L: >101 записи
        total = obs["manifest"]["n_records"]
        self.assertGreaterEqual(total, 101, "фикстура L требует >= 101 записи")
        seen: list[str] = []
        cursor, pages = None, 0
        while True:
            out, *_ = env.step({"name": "list_records",
                                "args": {"table": "ledger", "cursor": cursor,
                                         "limit": 50}})
            rows = out["result"]["records"]
            self.assertLessEqual(len(rows), 50)
            seen += [r["row_id"] for r in rows]
            pages += 1
            cursor = out["result"]["next_cursor"]
            if cursor is None:
                break
            self.assertLess(pages, 100, "пагинация должна завершиться")
        self.assertEqual(len(seen), total)
        self.assertEqual(len(set(seen)), total)

    # -- C06 ------------------------------------------------------------- #
    def test_c06_same_page_twice_is_identical(self):
        env, _, _ = self.closed_env(self.clean_dir)
        first, *_ = env.step({"name": "list_records",
                              "args": {"table": "ledger", "cursor": None, "limit": 50}})
        second, *_ = env.step({"name": "list_records",
                               "args": {"table": "ledger", "cursor": None, "limit": 50}})
        self.assertEqual(first["result"], second["result"])
        self.assertNotEqual(first["remaining_steps"], second["remaining_steps"],
                            "повторный запрос расходует шаг")

    # -- C07 ------------------------------------------------------------- #
    def test_c07_get_record_returns_existing_record_only(self):
        env, _, _ = self.closed_env(self.clean_dir)
        page, *_ = env.step({"name": "list_records",
                             "args": {"table": "ledger", "cursor": None, "limit": 1}})
        row = page["result"]["records"][0]
        out, *_ = env.step({"name": "get_record",
                            "args": {"table": "ledger", "id": row["row_id"]}})
        self.assertEqual(out["result"]["row_id"], row["row_id"])
        blob = json.dumps(out, ensure_ascii=False)
        for token in LEAK_TOKENS:
            self.assertNotIn(token, blob)

    # -- C08 ------------------------------------------------------------- #
    def test_c08_get_document_without_arbitrary_file_access(self):
        env, _, _ = self.closed_env(self.clean_dir)
        page, *_ = env.step({"name": "list_records",
                             "args": {"table": "invoices", "cursor": None, "limit": 1}})
        doc_id = page["result"]["records"][0]["doc_id"]
        out, *_ = env.step({"name": "get_document", "args": {"doc_id": doc_id}})
        self.assertEqual(out["result"]["doc_id"], doc_id)
        # произвольный путь -> ограниченная ошибка, а не содержимое файла
        evil, *_ = env.step({"name": "get_document",
                             "args": {"doc_id": "../../closed/truth.json"}})
        self.assertIsNone(evil["result"])
        self.assertIsNotNone(evil["error"])
        self.assertNotIn("issues", json.dumps(evil, ensure_ascii=False))

    # -- C09 ------------------------------------------------------------- #
    def test_c09_calculator_decimal_result(self):
        env, _, _ = self.closed_env(self.clean_dir)
        out, *_ = env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
        self.assertEqual(out["result"], "0.3")

    # -- C10 ------------------------------------------------------------ #
    def test_c10_calculator_outside_grammar_is_controlled_error(self):
        env, _, _ = self.closed_env(self.clean_dir)
        out, *_ = env.step({"name": "calculator",
                            "args": {"expression": "open('f').read()"}})
        self.assertIsNone(out["result"])
        self.assertEqual(out["error"]["code"], "invalid_expression")
        self.assertNotIn("Traceback", json.dumps(out, ensure_ascii=False))
        self.assertNotIn("ModuleNotFound", json.dumps(out, ensure_ascii=False))

    # -- C11 ------------------------------------------------------------ #
    def test_c11_unknown_record_limited_error_costs_step(self):
        env, _, info = self.closed_env(self.clean_dir)
        before = info["steps_used"]
        out, *_ = env.step({"name": "get_record",
                            "args": {"table": "ledger", "id": "U-нет-такой-записи"}})
        self.assertEqual(out["error"]["code"], "unknown_id")
        self.assertEqual(out["remaining_steps"], 99 - before)
        # среда жива: следующий запрос проходит
        again, *_ = env.step({"name": "list_records",
                              "args": {"table": "ledger", "cursor": None, "limit": 1}})
        self.assertIsNotNone(again["result"])

    # -- C12 ------------------------------------------------------------ #
    def test_c12_unknown_document_limited_error_no_disclosure(self):
        env, _, _ = self.closed_env(self.clean_dir)
        out, *_ = env.step({"name": "get_document", "args": {"doc_id": "U"}})
        self.assertEqual(out["error"]["code"], "unknown_document")
        self.assertLessEqual(len(out["error"]["message"]), 200)
        self.assertNotIn("invoice", json.dumps(out["result"]).lower()
                         if out["result"] else "")

    # -- C13 ------------------------------------------------------------ #
    def test_c13_unknown_action_limited_error_costs_step(self):
        env, _, _ = self.closed_env(self.clean_dir)
        out, reward, term, trunc, info = env.step(
            {"name": "definitely_unknown_action", "args": {}})
        self.assertEqual(out["error"]["code"], "unknown_action")
        self.assertEqual(info["steps_used"], 1)
        self.assertFalse(term or trunc)

    # -- C14 ------------------------------------------------------------ #
    def test_c14_budget_decreases_one_per_request_including_errors(self):
        env, _, _ = self.closed_env(self.clean_dir)
        used = []
        for action in ({"name": "list_records",
                        "args": {"table": "ledger", "cursor": None, "limit": 1}},
                       {"name": "get_record",
                        "args": {"table": "ledger", "id": "U"}},
                       {"name": "calculator", "args": {"expression": "2+2"}}):
            _, _, _, _, info = env.step(action)
            used.append(info["steps_used"])
        self.assertEqual(used, [1, 2, 3], "ошибка не даёт бесплатное действие")

    # -- C15 ------------------------------------------------------------ #
    def test_c15_submit_final_and_later_actions_change_nothing(self):
        env, _, _ = self.closed_env(self.error_dir)
        out, reward, term, trunc, _ = env.step(
            {"name": "submit", "args": {"findings": {"findings": []}}})
        self.assertEqual(out["result"], {"accepted": True})
        self.assertTrue(term and not trunc)
        report_before = env._env.evaluator_report()
        for _ in range(3):
            env.step({"name": "list_records",
                      "args": {"table": "ledger", "cursor": None, "limit": 1}})
        report_after = env._env.evaluator_report()
        self.assertEqual(report_before["metrics"], report_after["metrics"])
        self.assertEqual(report_before["true_reward"], report_after["true_reward"])
        self.assertEqual(report_before["steps_used"], report_after["steps_used"])

    # -- C16 ------------------------------------------------------------ #
    def test_c16_submit_as_action_100_is_accepted(self):
        env, _, _ = self.closed_env(self.clean_dir)
        for _ in range(99):
            _, _, term, trunc, _ = env.step(
                {"name": "calculator", "args": {"expression": "1+1"}})
            self.assertFalse(term or trunc)
        out, _, term, trunc, _ = env.step(
            {"name": "submit", "args": {"findings": {"findings": []}}})
        self.assertEqual(out["result"], {"accepted": True})
        self.assertTrue(term and not trunc)

    # -- C17 ------------------------------------------------------------ #
    def test_c17_hundred_steps_without_submit_truncates_empty(self):
        env, _, _ = self.closed_env(self.error_dir)
        for _ in range(99):
            _, _, term, trunc, _ = env.step(
                {"name": "calculator", "args": {"expression": "1+1"}})
            self.assertFalse(term or trunc)
        out, _, term, trunc, _ = env.step(
            {"name": "calculator", "args": {"expression": "1+1"}})
        self.assertFalse(term)
        self.assertTrue(trunc)
        report = env._env.evaluator_report()
        self.assertEqual(report["log"][-1]["event"], "truncated")
        self.assertEqual(report["metrics"]["n_findings"], 0, "итог пустой")
        # запрос после завершения не добавляет результаты задним числом
        env.step({"name": "submit", "args": {"findings": {"findings": []}}})
        self.assertEqual(env._env.evaluator_report()["metrics"],
                         report["metrics"])

    # -- C18 ------------------------------------------------------------ #
    def test_c18_time_limit_is_reproducible_without_slow_hardware(self):
        env = ja.make_env()
        env.reset(config={"mode": "closed", "episode_dir": str(self.error_dir),
                          "time_limit_s": 0})
        out, _, term, trunc, _ = env.step(
            {"name": "list_records", "args": {"table": "ledger", "cursor": None}})
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(out["error"]["code"], "time_limit")
        report = env._env.evaluator_report()
        self.assertEqual(report["log"][-1]["event"], "truncated")
        self.assertEqual(report["metrics"]["n_findings"], 0, "пустой итоговый ответ")

    # -- C19 ------------------------------------------------------------ #
    def test_c19_closed_mode_zero_reward_true_reward_evaluator_only(self):
        env, _, info = self.closed_env(self.error_dir)
        self.assertEqual(info["mode"], "eval")
        _, reward1, *_ = env.step({"name": "list_records",
                                   "args": {"table": "ledger", "cursor": None,
                                            "limit": 1}})
        self.assertEqual(reward1, 0.0, "исследовательские награды нулевые")
        _, reward2, term, *_ = env.step(
            {"name": "submit", "args": {"findings": {"findings": []}}})
        self.assertTrue(term)
        self.assertEqual(reward2, 0.0, "в закрытом режиме reward = 0, включая завершение")
        # истинная награда у оценщика (пустой ответ на грязном эпизоде -> -1)
        self.assertEqual(env._env.evaluator_report()["true_reward"], -1.0)
        # в info нет метрик/эталона
        self.assertNotIn("metrics", json.dumps(info, ensure_ascii=False))

    # -- C20 ------------------------------------------------------------ #
    def test_c20_no_leak_and_no_shell_sql_fs_network_anywhere(self):
        for episode_dir in (self.clean_dir, self.error_dir):
            env, obs, info = self.closed_env(episode_dir)
            blobs = [json.dumps(obs, ensure_ascii=False),
                     json.dumps(info, ensure_ascii=False)]
            for table in ("ledger", "invoices", "payments", "fx"):
                cursor = None
                for _ in range(8):
                    out, *_ = env.step({"name": "list_records",
                                        "args": {"table": table, "cursor": cursor,
                                                 "limit": 50}})
                    blobs.append(json.dumps(out, ensure_ascii=False))
                    if out["result"] is None:
                        break
                    cursor = out["result"]["next_cursor"]
                    if cursor is None:
                        break
            # обычные ошибки
            for action in ({"name": "get_record",
                            "args": {"table": "ledger", "id": "U"}},
                           {"name": "get_document", "args": {"doc_id": "U"}},
                           {"name": "list_records",
                            "args": {"table": "users", "cursor": None}},
                           {"name": "shell", "args": {"cmd": "rm -rf /"}},
                           {"name": "eval", "args": {"expression": "1"}},
                           {"name": "sql", "args": {"query": "SELECT 1"}}):
                out, *_ = env.step(action)
                blobs.append(json.dumps(out, ensure_ascii=False))
            joined = "\n".join(blobs)
            for token in LEAK_TOKENS:
                self.assertNotIn(token, joined,
                                 f"{episode_dir.name}: утечка {token}")
            # никакого shell/sql/файловой системы/сети в ответах
            for banned in ("Traceback", "/etc/passwd", "SELECT", "socket"):
                self.assertNotIn(banned, joined)


if __name__ == "__main__":
    unittest.main()
