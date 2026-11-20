"""Public package surface for FinRepGYM."""

import sys

from .env import AuditEnv, ContractViolation, safe_calculate
from .generator import (  # noqa: F401
    DEFAULT_CONFIG, ISSUE_TYPES, SCHEMA_VERSION, episode_fingerprint, generate,
    load_episode, save_episode,
)
from .matcher import canonical_ids, compute_metrics, match, normalize_finding, validate_submission
from .reward import terminal_reward
from .transforms import permute_episode, permute_order, remap_submission, rename_ids


def protect_stream_encoding(stream) -> bool:
    """Не даёт кириллице ронять печать в не-кириллической консоли (журнал №15).

    Windows-консоль в локали en-US (cp1252) не умеет кодировать русские буквы:
    любой print падает с UnicodeEncodeError. Если кодировка потока не представляет
    кириллицу — переключаем его на UTF-8 (непредставимые символы заменяются '?').
    Кодировки, которые кириллицу умеют (utf-8, cp1251, ...), не трогаем.
    Возвращает True, если поток был переключен.
    """
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        "проверка".encode(encoding)
        return False
    except (LookupError, UnicodeError):
        pass
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
        return True
    except Exception:  # noqa: BLE001  (например, перенаправленный/недоступный поток)
        return False


def _protect_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        protect_stream_encoding(stream)


_protect_stdio()

__all__ = [
    "AuditEnv", "ContractViolation", "safe_calculate",
    "DEFAULT_CONFIG", "ISSUE_TYPES", "SCHEMA_VERSION", "generate", "load_episode", "save_episode",
    "canonical_ids", "compute_metrics", "match", "normalize_finding", "validate_submission",
    "terminal_reward", "permute_episode", "permute_order", "rename_ids", "remap_submission",
]
__version__ = "0.1.0-skeleton"
