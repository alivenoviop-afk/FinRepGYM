#!/usr/bin/env python3
"""Эталонный отпечаток эпизода — доказательство независимости от железа.

    python scripts/golden_hash.py            # сверить с константой в тесте
    python scripts/golden_hash.py --seed 7   # посчитать для другого seed

Спека (раздел 04): «одинаковые seed, config, ВЕРСИЯ и действия воспроизводят
записи». Константа хеширована на Windows/Python 3.12; CI гоняет тот же тест
на Linux/Python 3.11 на 2 ядрах. Совпадение с обеих сторон = байт-в-байт
одинаковый эпизод на разном железе, ОС и версии Python.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_env import episode_fingerprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    actual = episode_fingerprint(args.seed)

    if args.seed != 42:
        print(f"seed={args.seed}  sha256={actual}")
        print("(эталонная константа есть только для seed=42 в tests/test_golden.py)")
        return 0

    # константа дублируется в tests/test_golden.py — держать синхронно
    expected = None
    golden = ROOT / "tests" / "test_golden.py"
    for line in golden.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("GOLDEN_HASH"):
            expected = line.split('"')[1]
            break

    print(f"seed          : 42")
    print(f"config        : по умолчанию (DEFAULT_CONFIG)")
    print(f"ожидается     : {expected}")
    print(f"получено      : {actual}")
    if expected == actual:
        print("OK — отпечаток совпадает: эпизод байт-в-байт воспроизводим "
              "на этой машине (проверяется и в CI на Linux)")
        return 0
    print("MISMATCH — генератор изменялся: обновите GOLDEN_HASH в tests/test_golden.py "
          "осознанно (история git покажет, что изменилось)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
