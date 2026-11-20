"""Фаззинг действий: сотни вредоносных/случайных вызовов — среда не падает.

Спека, раздел 05: «Проверьте ... неверный ввод»; раздел 04: наблюдение всегда
JSON-совместимо, шаги не превышают бюджет, ошибки ограничены и не содержат
подсказок.

Зерно фиксировано: прогон детерминирован и воспроизводим.
"""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_env import AuditEnv, ContractViolation  # noqa: E402

GARBAGE = [
    "' OR '1'='1", "__import__('os').system('dir')", "../../etc/passwd",
    "\U0001F600\U0001F600", "", "x" * 5000, None, -1, 0, True, 1e308,
    {"nested": {"deep": [1] * 50}}, [1, 2, 3], 3.14, "ledger; DROP TABLE--",
]
TABLES = ["ledger", "invoices", "payments", "fx", "users", "ledger; DROP TABLE--"]


def random_action(rng: random.Random, env: AuditEnv):
    kind = rng.randrange(7)
    if kind == 0:
        return {"action": rng.choice(["list_records", "list_records", "shell", "eval",
                                      "", "LIST_RECORDS"]),
                "table": rng.choice(TABLES),
                "cursor": rng.choice([0, -1, 10 ** 9, "abc", None, 1.5, True]),
                "limit": rng.choice([0, 1, -5, 50, 51, 10 ** 9, "много", None])}
    if kind == 1:
        return {"action": "get_record", "table": rng.choice(TABLES),
                "id": rng.choice(["row_00001", None, 123, "", "' OR 1=1--",
                                  "\U0001F600"] + GARBAGE[:5])}
    if kind == 2:
        return {"action": "get_document",
                "doc_id": rng.choice(["inv_doc_0001", None, "", "nope"] + GARBAGE[:4])}
    if kind == 3:
        return {"action": "calculator", "expression": rng.choice(
            ["1+1", "1/0", "__import__('os')", "9" * 5000, None, "0x10", "1e999",
             "()", "-(-(-1))", GARBAGE[6]])}
    if kind == 4:
        # настоящая отправка (валидная или нет) — обязана завершать эпизод честно
        return {"action": "submit",
                "episode_id": rng.choice([env._manifest["episode_id"], "ep_lie", None]),
                "findings": rng.choice([
                    [],
                    [{}],
                    [{"type": "x"}] * 40,
                    [{"type": "duplicate_entry", "affected_ids": ["a"],
                      "claimed_amount_minor": "много", "currency": "usd",
                      "evidence_ids": [], "explanation": ""}],
                ])}
    if kind == 5:
        return {"action": rng.choice(["get_record", "calculator", "submit"]),
                "payload": rng.choice(GARBAGE)}
    return rng.choice(GARBAGE)  # вообще не объект


class TestFuzz(unittest.TestCase):
    def test_five_hundred_hostile_actions_never_crash(self):
        rng = random.Random(20261120)
        env = AuditEnv()
        env.reset(seed=7, config={"n_invoices": 20})
        episode_id = env._manifest["episode_id"]
        steps = 0
        finished = False
        for _ in range(500):
            action = random_action(rng, env)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except ContractViolation:
                finished = True
                break
            steps += 1
            # инварианты контракта на КАЖДОМ шаге
            json.dumps(obs)                      # только JSON-совместимое
            self.assertIsInstance(obs, dict)
            self.assertIn("steps_left", obs)
            self.assertIsInstance(reward, float)
            self.assertLessEqual(info["steps_used"], 100)
            self.assertGreaterEqual(obs["steps_left"], 0)
            if terminated or truncated:
                finished = True
                break
        self.assertTrue(finished, "эпизод обязан завершиться (submit или лимит шагов)")
        self.assertLessEqual(steps, 101, "бюджет шагов не может быть превышен")
        # после завершения любой шаг — ограниченная ошибка, среда не падает (C15/C17)
        obs, reward, terminated, truncated, _ = env.step(
            {"action": "list_records", "table": "ledger", "cursor": 0, "limit": 1})
        self.assertEqual(obs["error"]["code"], "episode_finished")
        self.assertTrue(terminated and not truncated)
        self.assertEqual(reward, 0.0)

    def test_hostile_queries_return_limited_json_errors(self):
        env = AuditEnv()
        env.reset(seed=8, config={"n_invoices": 15})
        for action in ({"action": "list_records", "table": "users", "cursor": -5,
                        "limit": 999},
                       {"action": "get_record", "table": "ledger", "id": "' OR '1'='1"},
                       {"action": "calculator", "expression": "2 +"},
                       {"action": None}):
            obs, *_ = env.step(action)
            json.dumps(obs)
            self.assertTrue({"result", "error"} & set(obs), obs)
            if "error" in obs:
                # ограниченная ошибка: только код и короткое сообщение
                self.assertLessEqual(len(obs["error"].get("message", "")), 200)


if __name__ == "__main__":
    unittest.main()
