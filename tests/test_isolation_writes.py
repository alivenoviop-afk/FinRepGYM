"""Среда ничего не пишет: файлы эпизода неизменны после полного прогона.

Спека, раздел 10: «запись в оценщик» — блокирующее условие оценивания.
Здесь доказывается обратное утверждение: даже проход ВСЕМИ действиями
(чтение всех таблиц, неверные id, калькулятор, вредоносные команды, отправка)
не меняет ни один байт в папке эпизода и не создаёт новых файлов.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_env import AuditEnv, generate, save_episode  # noqa: E402


def snapshot(root: Path) -> dict[str, str]:
    """sha256 каждого файла в папке (включая новые появившиеся)."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class TestEnvironmentWritesNothing(unittest.TestCase):
    def test_full_episode_does_not_modify_any_file(self):
        files = generate(1911, {"n_invoices": 20})
        tmp = Path(tempfile.mkdtemp(prefix="nowrite_"))
        try:
            save_episode(files, tmp / "ep")
            before = snapshot(tmp)

            env = AuditEnv()
            env.reset(seed=1, config={"episode_dir": str(tmp / "ep")})
            episode_id = env._manifest["episode_id"]

            # чтение всех таблиц + стабильный повтор страницы
            for table in ("ledger", "invoices", "payments", "fx"):
                obs, *_ = env.step({"action": "list_records", "table": table,
                                    "cursor": 0, "limit": 10})
                self.assertIn("result", obs, table)
            env.step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 5})

            # записи по id: валидный, невалидный, документы
            page, *_ = env.step({"action": "list_records", "table": "ledger",
                                 "cursor": 0, "limit": 1})
            env.step({"action": "get_record", "table": "ledger",
                      "id": page["result"]["rows"][0]["row_id"]})
            env.step({"action": "get_record", "table": "ledger", "id": "нет_такого"})
            ipage, *_ = env.step({"action": "list_records", "table": "invoices",
                                  "cursor": 0, "limit": 1})
            env.step({"action": "get_document", "doc_id": ipage["result"]["rows"][0]["doc_id"]})
            env.step({"action": "get_document", "doc_id": "нет_такого"})

            # калькулятор: валидный и вредоносный
            env.step({"action": "calculator", "expression": "123.45 * 2"})
            env.step({"action": "calculator", "expression": "__import__('os')"})

            # попытки записи/выполнения команд
            env.step({"action": "shell", "cmd": "rm -rf /"})
            env.step("просто строка вместо объекта")

            # отправка -> завершение
            obs, _, terminated, trunc, _ = env.step(
                {"action": "submit", "episode_id": episode_id, "findings": []})
            self.assertTrue(obs.get("result", {}).get("accepted"))
            self.assertTrue(terminated and not trunc)

            after = snapshot(tmp)
            self.assertEqual(before, after,
                             "файлы эпизода изменились во время прогона — "
                             "значит среда пишет там, где не должна")
            # отдельно: закрытая зона тоже нетронута
            self.assertTrue((tmp / "ep" / "closed" / "truth.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
