"""Классификация результата привязки к ядрам в марафоне (журнал сбоев №16).

Привязка к 2 ядрам — эмуляция «железа жюри» (спека 07):
  * на Linux и Windows API есть и привязка выполняется -> PASS;
  * на macOS такого API нет вообще -> ограничение ОС, статус SKIP;
  * API есть, но привязка не получилась -> прогон не был ограничен ядрами -> FAIL.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jury_marathon as jm  # noqa: E402


class TestAffinityCheck(unittest.TestCase):
    def test_successful_pin_is_pass(self):
        for note in ("SetProcessAffinityMask: ядра 0-1", "sched_setaffinity: ядра 0-1"):
            status, name, detail = jm.affinity_check(note)
            self.assertEqual(status, "pass", note)
            self.assertEqual(name, jm.AFFINITY_CHECK)
            self.assertEqual(detail, note)

    def test_macos_without_affinity_api_is_skip(self):
        status, name, detail = jm.affinity_check("привязка недоступна на этой ОС")
        self.assertEqual(status, "skip")
        self.assertEqual(name, jm.AFFINITY_CHECK)
        self.assertIn("на Linux и Windows привязка выполняется", detail)

    def test_affinity_rejected_by_os_is_fail(self):
        """Если API есть, но привязка не получилась — прогон не был ограничен ядрами."""
        for note in ("привязка отклонена ОС", "привязка недоступна: ошибка"):
            status, _, _ = jm.affinity_check(note)
            self.assertEqual(status, "fail", note)

    def test_skip_is_not_a_failure_and_not_a_pass(self):
        """SKIP не должен засчитываться как пройденная проверка ядер."""
        status, _, _ = jm.affinity_check("привязка недоступна на этой ОС")
        self.assertNotIn(status, ("pass", "fail"))


if __name__ == "__main__":
    unittest.main()
