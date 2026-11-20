#!/usr/bin/env python3
"""judge_adapter.py — единая точка входа оценщика (определения проверок 20.11.2026).

Жюри вызывает ровно три точки входа из корня репозитория (рабочая директория
контейнера — корень, см. README «Инструкция контейнера»)::

    generate_episode(seed, config, output_dir)      # эпизод в каталоге жюри
    env = make_env()                                # объект с reset/step
    score_episode(episode_dir, findings, inspection_steps)  # закрытая оценка

Снаружи — канонический протокол определений:

* действия приходят конвертом ``{"name": ..., "args": {...}}``;
* первое observation: ``{"task", "manifest", "policy", "remaining_steps"}``;
* далее: ``{"result", "error", "remaining_steps"}`` (незаполненное поле — null);
* ``list_records`` -> ``{"records": [...], "next_cursor": str|null}`` (первый
  cursor — null);
* ``get_record``/``get_document`` -> объект записи/документа;
* ``calculator`` -> десятичная строка;
* ``submit`` -> ``{"accepted": true}`` либо ограниченная ошибка;
* в закрытом режиме ``reward`` для политики всегда 0, включая завершение.

Адаптер только переводит форму (envelope, имена полей, курсоры) во внутренний
формат среды ``audit_env``: смысл задачи, действия и оценку не меняет, действий
от имени политики не принимает. Закрытая зона (seed, truth, provenance, пути)
доступна только процессу жюри.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audit_env import AuditEnv, SCHEMA_VERSION, generate, save_episode  # noqa: E402
from audit_env.matcher import compute_metrics, validate_submission  # noqa: E402
from audit_env.reward import terminal_reward  # noqa: E402

__all__ = ["generate_episode", "make_env", "score_episode",
           "evaluator_log", "SCHEMA_VERSION"]

# Журнал оценщика: структурные ошибки submit (определения: «ошибка записывается
# в журнале оценщика»). Доступен только вызывающей стороне-оценщику.
_EVALUATOR_LOG: list[dict] = []


def evaluator_log() -> list[dict]:
    """Копия журнала оценщика (структурные ошибки submit и т.п.)."""
    return list(_EVALUATOR_LOG)


# --------------------------------------------------------------------------- #
# 1. генерация эпизода в каталоге жюри
# --------------------------------------------------------------------------- #
def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Создаёт эпизод в ``output_dir``: manifest, ledger, invoices, payments,
    fx, policy + закрытые truth и provenance.

    Это обёртка над генератором команды, а не отдельный генератор: файлы кладутся
    в ``output_dir/open`` и ``output_dir/closed`` (плоская раскладка тоже
    читается средой — см. ``load_episode``).
    """
    files = generate(int(seed), dict(config or {}))
    save_episode(files, Path(output_dir))


# --------------------------------------------------------------------------- #
# 2. среда с каноническим протоколом
# --------------------------------------------------------------------------- #
def _unwrap(action: Any) -> tuple[Any, dict]:
    """Конверт ``{"name", "args"}`` -> (имя, аргументы); иначе — как есть."""
    if isinstance(action, dict) and "name" in action and not isinstance(action.get("name"), dict):
        args = action.get("args")
        return action.get("name"), (args if isinstance(args, dict) else {})
    return (action.get("action") if isinstance(action, dict) else None), \
           (action if isinstance(action, dict) else {})


def _to_int_cursor(value: Any) -> Any:
    """Первый курсор — null; дальше канонический строковый курсор -> int."""
    if value is None:
        return 0
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value          # среда вернёт ограниченную ошибку
    return value


def _as_findings_list(raw: Any) -> Any:
    """Принимает и список, и объект findings.json ({"episode_id", "findings"})."""
    if isinstance(raw, dict):
        return raw.get("findings", raw)
    return raw


class _AdapterEnv:
    """Объект среды с reset/step канонического протокола."""

    def __init__(self) -> None:
        self._env: AuditEnv | None = None

    # -- reset ----------------------------------------------------------- #
    def reset(self, seed: int | None = None,
              config: dict | None = None) -> tuple[dict, dict]:
        cfg = dict(config or {})
        mode_raw = str(cfg.get("mode", "train")).lower()
        mode = "eval" if mode_raw in ("closed", "eval") else "train"
        env = AuditEnv(mode=mode)

        episode_dir = cfg.get("episode_dir")
        if episode_dir:
            # Закрытая ветка: грузим ИМЕННО переданный эпизод, не генерируем
            # другой и не меняем ID. Seed на этой ветке не используется.
            obs, info = env.reset(seed=None, config={"episode_dir": str(episode_dir)})
        else:
            team_config = dict(cfg.get("team_config") or {})
            obs, info = env.reset(seed=int(seed or 0), config=team_config)
        self._env = env
        return {
            "task": obs["task"],
            "manifest": obs["manifest"],
            "policy": obs["policy"],
            "remaining_steps": obs["steps_left"],
            # сохраняем и документированное имя поля контракта команды
            "steps_left": obs["steps_left"],
        }, info

    # -- step ------------------------------------------------------------ #
    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        env = self._require()
        name, args = _unwrap(action)

        if name == "list_records":
            inner: dict[str, Any] = {"action": "list_records",
                                     "table": args.get("table"),
                                     "cursor": _to_int_cursor(args.get("cursor"))}
            if args.get("limit") is not None:      # отсутствующий limit -> умолчание среды
                inner["limit"] = args["limit"]
        elif name == "get_record":
            inner = {"action": "get_record", "table": args.get("table"),
                     "id": args.get("id")}
        elif name == "get_document":
            inner = {"action": "get_document", "doc_id": args.get("doc_id")}
        elif name == "calculator":
            inner = {"action": "calculator", "expression": args.get("expression")}
        elif name == "submit":
            payload = args.get("findings")
            findings_obj = payload if isinstance(payload, dict) else {}
            inner = {"action": "submit",
                     "episode_id": findings_obj.get("episode_id") or env._manifest["episode_id"],
                     "findings": _as_findings_list(payload)}
        elif name is None:
            inner = action                     # не объект -> invalid_action в среде
        else:
            inner = {"action": name}           # неизвестное имя -> unknown_action

        obs, reward, terminated, truncated, info = env.step(inner)
        return _to_canonical(obs, name), reward, terminated, truncated, info

    # -- helpers ---------------------------------------------------------- #
    def _require(self) -> AuditEnv:
        if self._env is None:
            raise RuntimeError("call reset() before step()")
        return self._env


def _to_canonical(obs: dict, name: Any) -> dict:
    """Наблюдение среды -> канонический вид определений."""
    budget = obs.get("steps_left")
    result = obs.get("result")
    error = obs.get("error")

    if isinstance(result, dict) and name == "list_records":
        nxt = result.get("next_cursor")
        result = {"records": result.get("rows", []),
                  "next_cursor": None if nxt is None else str(nxt)}
    elif isinstance(result, dict) and name == "get_record":
        result = result.get("record", result)
    elif isinstance(result, dict) and name == "get_document":
        result = result.get("document", result)
    elif isinstance(result, dict) and name == "calculator":
        result = result.get("result", result)      # десятичная строка
    elif isinstance(result, dict) and name == "submit" and "accepted" in result:
        result = {"accepted": bool(result["accepted"])}  # «только accepted»
    elif isinstance(result, dict) and "accepted" in result and name != "submit":
        result = {"accepted": bool(result["accepted"])}

    return {"result": result, "error": error, "remaining_steps": budget,
            "steps_left": budget}


def make_env() -> _AdapterEnv:
    """Объект среды: ``reset(seed, config)`` и ``step(action)`` канонического
    протокола. ``config``: ``{"mode": "closed"|"train", "episode_dir": ...,
    "team_config": {...}}``."""
    return _AdapterEnv()


# --------------------------------------------------------------------------- #
# 3. закрытая оценка
# --------------------------------------------------------------------------- #
def score_episode(episode_dir: str, findings: dict,
                  inspection_steps: int) -> dict:
    """Оценка ответа политики на стороне жюри.

    Валидный submit -> ``{tp, fp, fn, precision, recall, f1, return}``.
    Невалидная структура -> ``return = -1``, остальные поля null (и запись в
    журнале оценщика). ``inspection_steps`` — число исследовательских действий
    ДО submit (submit в него не входит — определения 20.11).
    """
    env = AuditEnv(mode="eval")
    env.reset(seed=None, config={"episode_dir": str(episode_dir)})

    if isinstance(findings, dict) and "episode_id" in findings:
        payload = findings
    else:
        payload = {"episode_id": env._manifest["episode_id"],
                   "findings": _as_findings_list(findings)}

    ok, code, normalised = validate_submission(payload, env._manifest["episode_id"])
    if not ok:
        _EVALUATOR_LOG.append({"episode_dir": str(episode_dir),
                               "event": "submit_invalid", "code": code,
                               "inspection_steps": int(inspection_steps)})
        return {"tp": None, "fp": None, "fn": None, "precision": None,
                "recall": None, "f1": None, "return": -1.0}

    metrics = compute_metrics(env.evaluator_truth(), normalised["findings"],
                              env._valid_evidence_ids)
    return {
        "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
        "precision": metrics["precision"], "recall": metrics["recall"],
        "f1": metrics["f1"],
        "return": float(terminal_reward(metrics["f1"], int(inspection_steps))),
    }


if __name__ == "__main":  # pragma: no cover - самопроверка точки входа
    print(f"judge_adapter: schema {SCHEMA_VERSION}, "
          f"точки входа: generate_episode, make_env, score_episode")
