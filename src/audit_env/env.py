"""Gymnasium-style audit environment (spec section 04).

Contract
--------
``reset(seed, config) -> (observation, info)``
``step(action)     -> (observation, reward, terminated, truncated, info)``

Inheriting ``gym.Env`` is optional and not required by the spec, so this class is
plain Python. Objects are JSON-compatible.

Isolation and leakage
---------------------
* ``reset`` rebuilds *all* episode state from ``seed`` + ``config``; nothing from
  a previous episode survives.
* Observations contain only: the requested records, the result or a limited
  error, and the remaining step budget. No shell, SQL, filesystem, network,
  seed, ``truth.json`` or ``provenance.json``.
* ``reward`` is ``0.0`` for every investigation step. The terminal reward is
  returned to the policy only in ``mode="train"``. In ``mode="eval"`` the policy
  always receives ``0.0`` and the true reward/diagnostics stay on the evaluator
  side until the run ends (accessible through :meth:`evaluator_report`).
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .generator import DEFAULT_CONFIG, SCHEMA_VERSION, generate, load_episode
from .matcher import compute_metrics, validate_submission
from .reward import terminal_reward

TASK_TEXT = (
    "Вы — аудитор. Изучите записи и документы компании за отчётный период с помощью "
    "инструментов расследования и отправьте action=submit со списком финансовых ошибок. "
    "Каждая находка должна опираться на существующие записи (evidence_ids). "
    "Скрытый эталон, зерно генерации и правильность промежуточных шагов недоступны."
)

POLICY_BRIEF = [
    "Действия: list_records(table, cursor, limit), get_record(table, id), "
    "get_document(doc_id), calculator(expression), submit(findings)",
    "Страница не больше 50 строк; курсоры стабильны; неизвестный id или действие "
    "дают ограниченную ошибку и расходуют шаг",
    "Бюджет: 100 шагов, 60 секунд; submit завершает эпизод; лимит — truncated с пустым ответом",
    "Награда за исследование равна нулю; терминальная награда зависит только от "
    "полей находок и доказательств, не от текста объяснения",
    "Shell, SQL, файловая система и сеть недоступны",
]

OPEN_FILENAMES = ["manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                  "fx.csv", "policy.json"]
TABLES = ("ledger", "invoices", "payments", "fx")

ACTION_NAMES = ("list_records", "get_record", "get_document", "calculator", "submit")

_ID_FIELD = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id"}

_CALC_ALLOWED_CHARS = re.compile(r"^[0-9+\-*/(). \t]+$")
_CALC_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


class ContractViolation(RuntimeError):
    """Raised only for programmer errors (e.g. step() before reset())."""


# --------------------------------------------------------------------------- #
# safe calculator
# --------------------------------------------------------------------------- #
def _eval_decimal(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        val = _eval_decimal(node.operand)
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp):
        left, right = _eval_decimal(node.left), _eval_decimal(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError()
            return left / right
    raise ValueError("unsupported node")


def safe_calculate(expression: Any, max_len: int = 200) -> str:
    """Arithmetic and decimal numbers only: no names, no calls, no files, no code."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > max_len:
        raise ValueError("invalid_expression")
    if not _CALC_ALLOWED_CHARS.match(expression):
        raise ValueError("invalid_expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid_expression") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _CALC_ALLOWED_NODES):
            raise ValueError("invalid_expression")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("invalid_expression")
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise ValueError("invalid_expression")
    try:
        value = _eval_decimal(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError("division_by_zero") from exc
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid_expression") from exc
    return str(value)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class AuditEnv:
    metadata = {"name": "FinRepGYM", "schema_version": SCHEMA_VERSION,
                "reward_aggregation": "terminal only"}

    def __init__(
        self,
        mode: str = "train",
        max_steps: int = 100,
        page_limit: int = 50,
        time_limit_s: float = 60.0,
        truncation_reward: str = "score_empty",
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if truncation_reward not in ("score_empty", "zero"):
            raise ValueError("truncation_reward must be 'score_empty' or 'zero'")
        self.mode = mode
        self.max_steps = int(max_steps)
        self.page_limit = int(page_limit)
        self.time_limit_s = float(time_limit_s)
        self.truncation_reward = truncation_reward

        self._ready = False
        self._finished = False
        self._steps = 0
        self._t0 = 0.0
        self._eval_log: list[dict] = []
        self._last_metrics: dict | None = None
        self._last_reward = 0.0
        self._tables: dict[str, list[dict]] = {}
        self._index: dict[str, dict[str, dict]] = {}
        self._documents: dict[str, dict] = {}
        self._manifest: dict = {}
        self._policy: dict = {}
        self._episode_id = ""
        self._truth: list[dict] = []

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, config: dict | None = None) -> tuple[dict, dict]:
        cfg = dict(config or {})
        if "episode_dir" in cfg:
            files = load_episode(cfg["episode_dir"])
        else:
            files = generate(int(seed or 0), cfg)

        open_files = files["open"]
        missing = [n for n in OPEN_FILENAMES if n not in open_files]
        if missing:
            raise ContractViolation(f"episode is missing open files: {missing}")

        self._manifest = json.loads(open_files["manifest.json"])
        self._policy = json.loads(open_files["policy.json"])
        self._episode_id = self._manifest["episode_id"]
        self._tables = {
            "ledger": list(csv.DictReader(io.StringIO(open_files["ledger.csv"]))),
            "invoices": [json.loads(line) for line in open_files["invoices.jsonl"].splitlines() if line],
            "payments": [json.loads(line) for line in open_files["payments.jsonl"].splitlines() if line],
            "fx": list(csv.DictReader(io.StringIO(open_files["fx.csv"]))),
        }
        for row in self._tables["ledger"]:
            for key in ("debit_minor", "credit_minor"):
                row[key] = int(row[key])
        for row in self._tables["invoices"]:
            row["total_minor"] = int(row["total_minor"])
        for row in self._tables["payments"]:
            row["amount_minor"] = int(row["amount_minor"])

        self._index = {"ledger": {r["row_id"]: r for r in self._tables["ledger"]},
                       "invoices": {r["doc_id"]: r for r in self._tables["invoices"]},
                       "payments": {r["doc_id"]: r for r in self._tables["payments"]},
                       "fx": {f'{r["date"]}:{r["source_currency"]}:{r["target_currency"]}': r
                              for r in self._tables["fx"]}}
        self._documents = {r["doc_id"]: {"kind": "invoice", **r} for r in self._tables["invoices"]}
        self._documents.update({r["doc_id"]: {"kind": "payment", **r}
                                for r in self._tables["payments"]})

        closed = files.get("closed") or {}
        if "truth.json" not in closed:
            raise ContractViolation("episode has no closed truth.json; evaluator cannot score it")
        self._truth = json.loads(closed["truth.json"])["issues"]

        # ---- episode isolation ---- #
        self._steps = 0
        self._finished = False
        self._eval_log = []
        self._last_metrics = None
        self._last_reward = 0.0
        self._t0 = time.monotonic()
        self._ready = True

        observation = {
            "task": TASK_TEXT,
            "manifest": self._manifest,
            # policy.json is an OPEN file: exposing it here is not a leak
            "policy": {**self._policy, "brief": list(POLICY_BRIEF)},
            "steps_left": self.max_steps,
        }
        info = {"mode": self.mode, "steps_used": 0, "schema_version": SCHEMA_VERSION}
        return observation, info

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        if not self._ready:
            raise ContractViolation("call reset() before step()")
        if self._finished:
            raise ContractViolation("episode already finished; call reset()")
        self._steps += 1
        steps_left = max(0, self.max_steps - self._steps)

        # Лимит ВРЕМЕНИ проверяется до выполнения действия: по спеке истёкшее время
        # даёт truncated=true с пустым итоговым ответом, а действие не исполняется.
        # >= вместо > : на Windows time.monotonic() может не продвинуться между
        # соседними вызовами (granularity), а лимит 0 должен срабатывать всегда.
        if (time.monotonic() - self._t0) >= self.time_limit_s:
            reward = self._finalise_empty()
            observation = {"error": {"code": "time_limit",
                                     "message": "episode time limit exceeded"},
                           "steps_left": steps_left}
            info = {"mode": self.mode, "steps_used": self._steps}
            return observation, float(reward), False, True, info

        is_submit = isinstance(action, dict) and action.get("action") == "submit"
        terminated = truncated = False
        reward = 0.0
        observation: dict

        if is_submit:
            observation, terminated, reward = self._handle_submit(action)
        else:
            payload, error = self._dispatch(action)
            observation = {"result": payload} if error is None else {"error": error}
            observation["steps_left"] = steps_left
            if steps_left <= 0 or (time.monotonic() - self._t0) >= self.time_limit_s:
                truncated = True
                reward = self._finalise_empty()

        observation["steps_left"] = steps_left
        info = {"mode": self.mode, "steps_used": self._steps}
        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # submit + terminal reward
    # ------------------------------------------------------------------ #
    def _handle_submit(self, action: dict) -> tuple[dict, bool, float]:
        # The submit action IS the findings.json object: {episode_id, findings}.
        ok, code, payload = validate_submission(action, self._episode_id)
        steps_left = max(0, self.max_steps - self._steps)
        self._finished = True

        if not ok:
            self._eval_log.append({"event": "submit_invalid", "code": code,
                                   "steps_used": self._steps})
            self._last_metrics = None
            self._last_reward = -1.0 if self.mode == "train" else 0.0
            observation = {"error": {"code": "submit_invalid", "message": "submit rejected: invalid structure"},
                           "steps_left": steps_left}
            # Spec: malformed submission terminates with reward -1 (logged on the evaluator side).
            return observation, True, (-1.0 if self.mode == "train" else 0.0)

        metrics = compute_metrics(self._truth, payload["findings"])
        reward = terminal_reward(metrics["f1"], self._steps)
        self._last_metrics = metrics
        self._last_reward = reward
        self._eval_log.append({"event": "submit_accepted", "steps_used": self._steps,
                               "n_findings": len(payload["findings"])})
        observation = {"result": {"accepted": True, "findings_count": len(payload["findings"])},
                       "steps_left": steps_left}
        return observation, True, (reward if self.mode == "train" else 0.0)

    def _finalise_empty(self) -> float:
        """Terminal value for a truncated episode (empty final answer)."""
        metrics = compute_metrics(self._truth, [])
        self._last_metrics = metrics
        if self.truncation_reward == "zero":
            reward = 0.0
        else:
            reward = terminal_reward(metrics["f1"], self._steps)
        self._last_reward = reward
        self._eval_log.append({"event": "truncated", "steps_used": self._steps})
        self._finished = True
        return reward if self.mode == "train" else 0.0

    # ------------------------------------------------------------------ #
    # action dispatch
    # ------------------------------------------------------------------ #
    def _dispatch(self, action: Any) -> tuple[Any, dict | None]:
        if not isinstance(action, dict):
            return None, {"code": "invalid_action", "message": "action must be an object"}
        name = action.get("action")
        if name not in ACTION_NAMES:
            return None, {"code": "unknown_action",
                          "message": f"unknown action; allowed: {', '.join(ACTION_NAMES)}"}
        try:
            if name == "list_records":
                return self._list_records(action), None
            if name == "get_record":
                return self._get_record(action), None
            if name == "get_document":
                return self._get_document(action), None
            if name == "calculator":
                return self._calculator(action), None
        except _ActionError as exc:
            return None, {"code": exc.code, "message": exc.message}
        return None, {"code": "unknown_action", "message": "unhandled action"}

    def _list_records(self, action: dict) -> dict:
        table = action.get("table")
        if table not in TABLES:
            raise _ActionError("unknown_table", "unknown table")
        cursor = action.get("cursor", 0)
        limit = action.get("limit", self.page_limit)
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise _ActionError("invalid_cursor", "cursor must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise _ActionError("invalid_limit", "limit must be a positive integer")
        if limit > self.page_limit:
            raise _ActionError("limit_too_large", f"limit must be <= {self.page_limit}")
        rows = self._tables[table]
        page = rows[cursor:cursor + limit]
        next_cursor = cursor + len(page) if cursor + len(page) < len(rows) else None
        return {"table": table, "cursor": cursor, "count": len(page),
                "total": len(rows), "next_cursor": next_cursor, "rows": page}

    def _get_record(self, action: dict) -> dict:
        table = action.get("table")
        if table not in TABLES:
            raise _ActionError("unknown_table", "unknown table")
        record_id = action.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise _ActionError("invalid_id", "id must be a non-empty string")
        record = self._index[table].get(record_id)
        if record is None:
            raise _ActionError("unknown_id", "record not found")
        return {"table": table, "record": record}

    def _get_document(self, action: dict) -> dict:
        doc_id = action.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise _ActionError("invalid_id", "doc_id must be a non-empty string")
        doc = self._documents.get(doc_id)
        if doc is None:
            raise _ActionError("unknown_document", "document not found")
        return {"document": doc}

    def _calculator(self, action: dict) -> dict:
        try:
            result = safe_calculate(action.get("expression"))
        except ValueError as exc:
            raise _ActionError(str(exc), "expression must be arithmetic over decimal numbers") from exc
        return {"expression": action.get("expression"), "result": result}

    # ------------------------------------------------------------------ #
    # evaluator side (never reachable through observations)
    # ------------------------------------------------------------------ #
    def evaluator_report(self) -> dict:
        """Harness/evaluator only: metrics, true terminal reward, log."""
        if not self._ready:
            raise ContractViolation("call reset() first")
        return {
            "episode_id": self._episode_id,
            "mode": self.mode,
            "steps_used": self._steps,
            "finished": self._finished,
            "metrics": self._last_metrics,
            "true_reward": self._last_reward,
            "log": list(self._eval_log),
        }

    def evaluator_truth(self) -> list[dict]:
        """Evaluator/oracle only. Never put this into an observation."""
        if not self._ready:
            raise ContractViolation("call reset() first")
        return json.loads(json.dumps(self._truth))

    def close(self) -> None:
        self._ready = False


class _ActionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
