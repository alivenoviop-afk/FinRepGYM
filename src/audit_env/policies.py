"""Baseline policy adapters and the episode runner.

Four adapters required by the spec (section 06):

* :class:`RandomExplorerPolicy` — random investigator with a fixed seed;
* :class:`ReconciliationPolicy` — simple reconciliation algorithm;
* :class:`MarkAllPolicy` — the "flag everything" attack;
* :class:`OraclePolicy` — evaluator-side oracle that submits the correct findings
  with evidence (checks the maximum reward, not agent ability).

Rules respected here:

* a policy sees **only** observations — no truth, no seed, no files;
* ``manifest`` exists in the reset observation only, so policies capture it once;
* submit happens at most once, and an already-finished episode is never stepped.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Iterable

from .env import AuditEnv
from .generator import ISSUE_TYPES

PAGE = 50

Policy = Callable[[dict], dict]


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_episode(env: AuditEnv, policy: Policy, seed: int,
                config: dict | None = None) -> dict[str, Any]:
    """Run one episode; returns a JSON-compatible trajectory with no closed truth."""
    observation, info = env.reset(seed=seed, config=config)
    episode_id = observation["manifest"]["episode_id"]
    records: list[dict] = [{
        "episode_id": episode_id,
        "step": 0,
        "action": None,
        "observation": observation,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }]
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        step += 1
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        records.append({
            "episode_id": episode_id,
            "step": step,
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })
        if step >= env.max_steps:
            break
    return {"episode_id": episode_id, "steps": step,
            "trajectory": records, "report": env.evaluator_report()}


def write_trajectory(result: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in result["trajectory"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
class BasePolicy:
    """Captures the reset observation (the only place manifest lives)."""

    def __init__(self) -> None:
        self.manifest: dict | None = None
        self.step = 0

    def __call__(self, obs: dict) -> dict:
        self.step += 1
        if isinstance(obs.get("manifest"), dict):
            self.manifest = obs["manifest"]
        return self.act(obs)

    def act(self, obs: dict) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def submit(self, findings: list[dict]) -> dict:
        assert self.manifest is not None, "policy never saw the reset observation"
        return {"action": "submit", "episode_id": self.manifest["episode_id"],
                "findings": findings}


class Pager:
    """Stable cursor pagination over read-only tables."""

    def __init__(self) -> None:
        self.cursors: dict[str, int] = {}
        self.rows: dict[str, list[dict]] = {"ledger": [], "payments": [], "invoices": [], "fx": []}
        self.pending: str | None = None
        self.finished: set[str] = set()

    def start(self, table: str) -> None:
        self.pending = table

    def action(self) -> dict | None:
        if not self.pending:
            return None
        return {"action": "list_records", "table": self.pending,
                "cursor": self.cursors.get(self.pending, 0), "limit": PAGE}

    def feed(self, obs: dict) -> bool:
        """Ingest the last result. Returns True when the pending table is complete."""
        result = obs.get("result")
        if not (isinstance(result, dict) and result.get("table")):
            return False
        table = result["table"]
        self.rows.setdefault(table, []).extend(result.get("rows") or [])
        if result.get("next_cursor") is None:
            self.finished.add(table)
            if self.pending == table:
                self.pending = None
            return True
        self.cursors[table] = int(result["next_cursor"])
        return False


# --------------------------------------------------------------------------- #
# 1. random explorer (fixed seed)
# --------------------------------------------------------------------------- #
class RandomExplorerPolicy(BasePolicy):
    """Explores pseudo-randomly for N steps, then submits ID guesses."""

    def __init__(self, seed: int = 7, explore_steps: int = 20, guess_findings: int = 5) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.explore_steps = explore_steps
        self.guess_findings = guess_findings
        self.seen_rows: list[dict] = []
        self.submitted = False

    def act(self, obs: dict) -> dict:
        if self.submitted:
            return {"action": "calculator", "expression": "1 + 1"}
        result = obs.get("result")
        if isinstance(result, dict) and result.get("rows") and result["table"] == "ledger":
            self.seen_rows.extend(result["rows"])

        if self.step <= self.explore_steps:
            roll = self.rng.random()
            if roll < 0.45 and isinstance(result, dict) and result.get("next_cursor") is not None:
                return {"action": "list_records", "table": result["table"],
                        "cursor": result["next_cursor"], "limit": PAGE}
            if roll < 0.65:
                return {"action": "list_records", "table": "ledger",
                        "cursor": self.rng.randrange(0, 200), "limit": PAGE}
            if roll < 0.80:
                return {"action": "get_record", "table": "ledger",
                        "id": f"row_{self.rng.randrange(1, 600):05d}"}
            if roll < 0.90:
                return {"action": "calculator", "expression": f"{self.rng.randrange(1, 99)} * 1.5"}
            return {"action": "get_document", "doc_id": f"inv_doc_{self.rng.randrange(1, 60):04d}"}

        self.submitted = True
        findings = []
        for _ in range(self.guess_findings):
            row = self.rng.choice(self.seen_rows) if self.seen_rows else {
                "row_id": "row_00001", "debit_minor": 100, "credit_minor": 0, "currency": "USD"}
            findings.append({
                "type": self.rng.choice(ISSUE_TYPES),
                "affected_ids": [row["row_id"]],
                "claimed_amount_minor": int(row["debit_minor"] or row["credit_minor"] or 0),
                "currency": row["currency"],
                "evidence_ids": [f"ledger:{row['row_id']}"],
                "explanation": "случайная догадка",
            })
        return self.submit(findings)


# --------------------------------------------------------------------------- #
# 2. simple reconciliation
# --------------------------------------------------------------------------- #
class ReconciliationPolicy(BasePolicy):
    """Finds all four implemented issue types from records alone."""

    def __init__(self) -> None:
        super().__init__()
        self.pager = Pager()
        self.pager.start("ledger")
        self.stage = "collect"
        self.submitted = False

    SEQUENCE = ("payments", "fx")

    def act(self, obs: dict) -> dict:
        if self.stage == "collect":
            self.pager.feed(obs)
            if self.pager.pending is None:
                for table in self.SEQUENCE:
                    if table not in self.pager.finished:
                        self.pager.start(table)
                        break
            if self.pager.pending:
                return self.pager.action()
            self.stage = "submit"
        if self.submitted:
            return {"action": "calculator", "expression": "2 + 2"}
        self.submitted = True
        return self.submit(self._detect())

    def _detect(self) -> list[dict]:
        findings: list[dict] = []
        assert self.manifest is not None
        period = (self.manifest["period_start"], self.manifest["period_end"])
        policy_day = self.manifest["period_end"][:10]
        rows = self.pager.rows["ledger"]
        payments = self.pager.rows["payments"]
        rates = {(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"])
                 for r in self.pager.rows["fx"]}

        def convert(amount, src: str, tgt: str):
            """Курс policy-даты из fx.csv; None, если пара не опубликована."""
            if src == tgt:
                return int(amount)
            rate = rates.get((policy_day, src, tgt))
            if rate is None:
                return None
            return int((Decimal(int(amount)) * rate).quantize(Decimal(1), rounding=ROUND_HALF_UP))

        # (a) duplicates: identical in every field except row_id
        seen: dict[tuple, str] = {}
        for row in rows:
            key = (row["timestamp"], row["account"], row["debit_minor"], row["credit_minor"],
                   row["currency"], row["entity_id"], row["invoice_id"], row["payment_id"])
            if key in seen:
                findings.append({
                    "type": "duplicate_entry",
                    # обе строки пары: после перестановки нельзя сказать, какая была копией
                    "affected_ids": sorted([seen[key], row["row_id"]]),
                    "claimed_amount_minor": int(row["debit_minor"] or row["credit_minor"]),
                    "currency": row["currency"],
                    "evidence_ids": [f"ledger:{row['row_id']}", f"ledger:{seen[key]}",
                                     f"invoice:{row['invoice_id']}"],
                    "explanation": f"строка {row['row_id']} идентична {seen[key]} "
                                   f"по всем полям кроме row_id",
                })
            else:
                seen[key] = row["row_id"]

        # (b..d) платежи: сдвиг периода / кросс-валютный курс / частичное распределение
        by_payment: dict[str, list[dict]] = {}
        for row in rows:
            if row["payment_id"]:
                by_payment.setdefault(row["payment_id"], []).append(row)
        for pay in payments:
            pay_rows = by_payment.get(pay["payment_id"], [])
            if not pay_rows:
                continue
            doc_inside = period[0] <= pay["timestamp"] <= period[1]
            rows_outside = [r for r in pay_rows if not (period[0] <= r["timestamp"] <= period[1])]
            # cutoff: строки вне периода при документе внутри. НЕ прерываем цикл —
            # суммы проверяются ТОЖЕ: вторая ошибка сценария живёт в сумме, а не во времени.
            if doc_inside and rows_outside:
                ev_row = next((r for r in pay_rows if r["account"].startswith("1200")),
                              pay_rows[0])
                findings.append({
                    "type": "period_cutoff",
                    "affected_ids": sorted([pay["payment_id"]] + [r["row_id"] for r in pay_rows]),
                    "claimed_amount_minor": int(pay["amount_minor"]),
                    "currency": pay["currency"],
                    "evidence_ids": [f"payment:{pay['payment_id']}", f"ledger:{ev_row['row_id']}",
                                     "policy:period"],
                    "explanation": f"платёж {pay['payment_id']} внутри периода, "
                                   f"строки леджера — вне периода",
                })

            ar_row = next((r for r in pay_rows if r["account"].startswith("1200")), None)
            cash_row = next((r for r in pay_rows if r["account"].startswith("1010")), None)
            if ar_row is None or cash_row is None:
                continue

            # кросс-валютный платёж: документ в чужой валюте, постинг в базовой
            if ar_row["currency"] != pay["currency"]:
                expected = convert(pay["amount_minor"], pay["currency"], ar_row["currency"])
                if expected is None:
                    continue
                cash_val = int(cash_row["debit_minor"])
                ar_val = int(ar_row["credit_minor"])
                if cash_val != expected:
                    # обе строки уменьшены при полном документе -> частичное распределение
                    findings.append({
                        "type": "partial_allocation",
                        "affected_ids": sorted([pay["payment_id"], cash_row["row_id"],
                                                ar_row["row_id"]]),
                        "claimed_amount_minor": abs(expected - cash_val),
                        "currency": ar_row["currency"],
                        "evidence_ids": [f"payment:{pay['payment_id']}",
                                         f"ledger:{ar_row['row_id']}",
                                         f"invoice:{ar_row['invoice_id']}"],
                        "explanation": f"платёж {pay['payment_id']} ({pay['currency']}) "
                                       f"распределён на {cash_val} {ar_row['currency']} "
                                       f"при эквиваленте {expected}",
                    })
                elif ar_val != expected:
                    # касса сходится с policy-курсом, AR пересчитана по чужой дате
                    findings.append({
                        "type": "fx_rate",
                        "affected_ids": sorted([pay["payment_id"], cash_row["row_id"],
                                                ar_row["row_id"]]),
                        "claimed_amount_minor": abs(ar_val - expected),
                        "currency": ar_row["currency"],
                        "evidence_ids": [f"payment:{pay['payment_id']}",
                                         f"ledger:{ar_row['row_id']}",
                                         f"fx:{policy_day}:{pay['currency']}:{ar_row['currency']}",
                                         "policy:fx_rate_date"],
                        "explanation": f"платёж {pay['payment_id']} ({pay['currency']}) "
                                       f"конвертирован в {ar_row['currency']}: по policy-курсу "
                                       f"ожидалось {expected}, в журнале {ar_val}",
                    })
                continue

            # (d) частичное распределение (одна валюта: иначе это fx)
            allocated = sum(int(r["credit_minor"]) for r in pay_rows
                            if r["account"].startswith("1200"))
            if allocated != int(pay["amount_minor"]):
                findings.append({
                    "type": "partial_allocation",
                    "affected_ids": sorted([pay["payment_id"]] + [r["row_id"] for r in pay_rows]),
                    "claimed_amount_minor": abs(int(pay["amount_minor"]) - allocated),
                    "currency": pay["currency"],
                    "evidence_ids": [f"payment:{pay['payment_id']}",
                                     f"ledger:{ar_row['row_id']}",
                                     f"invoice:{ar_row['invoice_id']}"],
                    "explanation": f"платёж {pay['payment_id']} распределён на {allocated} "
                                   f"при документе на {pay['amount_minor']}",
                })
        return findings


# --------------------------------------------------------------------------- #
# 3. mark-everything attack
# --------------------------------------------------------------------------- #
class MarkAllPolicy(BasePolicy):
    """Flags the first 30 ledger rows regardless of evidence."""

    def __init__(self, max_findings: int = 30) -> None:
        super().__init__()
        self.pager = Pager()
        self.pager.start("ledger")
        self.max_findings = max_findings
        self.submitted = False

    def act(self, obs: dict) -> dict:
        if self.submitted:
            return {"action": "calculator", "expression": "1 + 1"}
        self.pager.feed(obs)
        if self.pager.pending:
            return self.pager.action()
        self.submitted = True
        rows = self.pager.rows["ledger"][: self.max_findings]
        findings = [{
            "type": "duplicate_entry",
            "affected_ids": [row["row_id"]],
            "claimed_amount_minor": int(row["debit_minor"] or row["credit_minor"]),
            "currency": row["currency"],
            "evidence_ids": [f"ledger:{row['row_id']}"],
            "explanation": "помечено всё подряд",
        } for row in rows]
        return self.submit(findings)


# --------------------------------------------------------------------------- #
# 4. oracle (evaluator side only)
# --------------------------------------------------------------------------- #
class OraclePolicy(BasePolicy):
    """Submits the ground truth with a complete allowed evidence set.

    Harness tool: it verifies the maximum reward, never the agent's ability.
    Do not ship it as part of an agent run.
    """

    def __init__(self, truth: Iterable[dict]) -> None:
        super().__init__()
        self.truth = list(truth)
        self.submitted = False

    def act(self, obs: dict) -> dict:
        if self.submitted:
            return {"action": "calculator", "expression": "1"}
        self.submitted = True
        findings = [{
            "type": issue["type"],
            "affected_ids": sorted(issue["affected_ids"]),
            "claimed_amount_minor": int(issue["expected_amount_minor"]),
            "currency": issue["currency"],
            "evidence_ids": list(issue["evidence_sets"][0]),
            "explanation": str(issue.get("statement", "оракул"))[:900],
        } for issue in self.truth]
        return self.submit(findings)
