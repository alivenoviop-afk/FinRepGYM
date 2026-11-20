"""Эталонный отпечаток эпизода: кросс-платформенный контроль воспроизводимости.

Спека, раздел 04: «одинаковые seed, config, ВЕРСИЯ и действия воспроизводят
записи, переходы и награды».

Константа GOLDEN_HASH вычислена на Windows / Python 3.12. CI гоняет этот же
тест на Linux / Python 3.11 / 2 ядрах. Если обе стороны совпадают с одной
константой — доказано формально: разные ОС, разное железо и разная версия
Python дают абсолютно одинаковый эпизод (данные не зависят от машины).

⚠️ Если генератор изменялся ОСОЗНАННО — обновите константу командой
`python scripts/golden_hash.py` (он напечатает новый хеш) и объясните
изменение в коммите. Слепое обновление хеши = потеря контроля.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_env import episode_fingerprint  # noqa: E402

GOLDEN_SEED = 42
GOLDEN_HASH = "05eb2eb1446aba20cd205dcbee41f174e7021bee99389d25636809d975d1ebf8"


class TestGoldenFingerprint(unittest.TestCase):
    def test_fingerprint_matches_reference(self):
        """Эпизод с seed=42 обязан совпадать с зафиксированным отпечатком.

        Локально (Windows/3.12) и в CI (Linux/3.11) — одна и та же константа
        доказывает независимость данных от ОС, железа и версии Python.
        """
        self.assertEqual(episode_fingerprint(GOLDEN_SEED), GOLDEN_HASH)

    def test_fingerprint_is_stable_within_process(self):
        first = episode_fingerprint(GOLDEN_SEED)
        second = episode_fingerprint(GOLDEN_SEED)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_fingerprint_changes_with_seed(self):
        self.assertNotEqual(episode_fingerprint(GOLDEN_SEED), episode_fingerprint(GOLDEN_SEED + 1))


if __name__ == "__main__":
    unittest.main()
