"""Консоль Windows: печать кириллицы не должна падать (журнал сбоев №15).

Проблема: в CI на Windows-раннерах локаль en-US (cp1252), а все наши сообщения
на русском — первый же print ронял прогон с UnicodeEncodeError. Здесь проверяется
защита: поток, не умеющий кириллицу, переключается на UTF-8; потоки, которые её
умеют, не трогаются.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_env import protect_stream_encoding  # noqa: E402


class FakeStream:
    """Поток с заданной кодировкой и ловушкой на reconfigure."""

    def __init__(self, encoding) -> None:
        self.encoding = encoding
        self.reconfigured = None

    def reconfigure(self, **kwargs) -> None:
        self.reconfigured = kwargs


class TestConsoleEncoding(unittest.TestCase):
    def test_foreign_locale_stream_is_switched_to_utf8(self):
        stream = FakeStream("cp1252")           # en-US: кириллица не помещается
        self.assertTrue(protect_stream_encoding(stream))
        self.assertEqual(stream.reconfigured,
                         {"encoding": "utf-8", "errors": "replace"})

    def test_cyrillic_capable_stream_is_untouched(self):
        for encoding in ("cp1251", "utf-8"):
            stream = FakeStream(encoding)
            self.assertFalse(protect_stream_encoding(stream), encoding)
            self.assertIsNone(stream.reconfigured, encoding)

    def test_stream_without_encoding_is_untouched(self):
        stream = FakeStream(None)
        self.assertFalse(protect_stream_encoding(stream))
        self.assertIsNone(stream.reconfigured)

    def test_stdout_can_print_cyrillic_after_import(self):
        """Смоук-тест: импорт пакета уже защитил вывод."""
        self.assertFalse(protect_stream_encoding(sys.stdout))
        print("проверка вывода: кириллица")
        sys.stdout.flush()


if __name__ == "__main__":
    unittest.main()
