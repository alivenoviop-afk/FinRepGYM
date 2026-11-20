"""Параллельный детерминизм: одновременные прогоны не влияют друг на друга.

Спека 04: «одинаковые seed, config, версия и действия воспроизводят записи,
переходы и награды». Здесь это проверяется в условиях, близких к реальному
прогону оценщика: несколько экземпляров среды работают ОДНОВРЕМЕННО в разных
потоках, и каждый обязан дать байт-в-байт тот же результат, что и одиночный
прогон (общего состояния нет, гонок нет).
"""

from __future__ import annotations

import hashlib
import json
import threading
import unittest

from support import SEED, AuditEnv, ReconciliationPolicy, run_with

CFG = {"n_invoices": 30,
       "issue_types": ["duplicate_entry", "period_cutoff",
                       "partial_allocation", "fx_rate"],
       "n_issues": 3}
THREADS = 3


def digest(result: dict) -> str:
    """Отпечаток всего прогона: наблюдения, действия, метрики, награда, шаги."""
    blob = json.dumps({
        "metrics": result["report"]["metrics"],
        "reward": result["report"]["true_reward"],
        "steps": result["steps"],
        "trajectory": [
            {"action": rec["action"], "observation": rec["observation"],
             "reward": rec["reward"], "terminated": rec["terminated"],
             "truncated": rec["truncated"]}
            for rec in result["trajectory"]
        ],
    }, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class TestParallelDeterminism(unittest.TestCase):
    def test_concurrent_runs_match_the_sequential_baseline(self):
        baseline = digest(run_with(lambda env: ReconciliationPolicy(), SEED, CFG))

        results: list[str | None] = [None] * THREADS
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                results[index] = digest(
                    run_with(lambda env: ReconciliationPolicy(), SEED, CFG))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertFalse(errors, f"параллельный прогон упал: {errors[:1]}")
        self.assertEqual(results, [baseline] * THREADS,
                         "одновременные прогоны дали разные наблюдения или награды")

    def test_second_env_is_unaffected_by_the_first(self):
        """Действия в одном экземпляре не меняют состояние другого."""
        busy = AuditEnv()
        busy.reset(seed=SEED, config=CFG)
        for cursor in (0, 7, 14, 21):
            busy.step({"action": "list_records", "table": "ledger",
                       "cursor": cursor, "limit": 7})

        left = AuditEnv()
        left.reset(seed=SEED, config=CFG)
        right = AuditEnv()
        right.reset(seed=SEED, config=CFG)

        action = {"action": "list_records", "table": "ledger", "cursor": 0, "limit": 7}
        obs_left, reward_left, *_ = left.step(action)
        obs_right, reward_right, *_ = right.step(action)

        self.assertEqual(obs_left, obs_right)
        self.assertEqual(reward_left, reward_right)
        self.assertIsNot(left._manifest, right._manifest)


if __name__ == "__main__":
    unittest.main()
