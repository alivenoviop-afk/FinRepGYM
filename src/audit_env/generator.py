"""Synthetic company generator (spec section 02 and 03).

Pipeline, in this exact order:

1. **Hidden history** — a consistent double-entry story: invoices, payments,
   counterparties. Every business event produces a *pair* of ledger rows, so
   ``sum(debit_minor) == sum(credit_minor)`` holds by construction.
2. **Documents** — ``ledger.csv``, ``invoices.jsonl``, ``payments.jsonl``,
   ``fx.csv``, ``policy.json``, ``manifest.json``.
3. **Causal mutations** — errors are injected *after* the clean history is
   verified, and each mutation is recorded in ``provenance.json`` with a
   before/after snapshot. ``truth.json`` describes the real discrepancy and the
   available evidence, never a randomly suspicious row.

The agent never sees ``truth.json``, ``provenance.json`` or the seed.

Implemented issue types (3 of the required 4 — see docs/RULES.md):

* ``duplicate_entry``     — one side of a balanced pair is duplicated, so the
  books stop balancing by exactly the duplicated amount.
* ``period_cutoff``       — both rows of a payment are moved across the period
  boundary, so the in-period AR balance no longer agrees with the documents.
* ``partial_allocation``  — the payment document keeps its full amount while
  both ledger rows are reduced, so the in-period AR balance is inflated by the
  missing part.

``fx_rate`` is added on 20.11 (multi-currency conversion).

Distractors generated on purpose: split payments (legal), repeated identical
invoice amounts (legal), unpaid invoices (legal), invoices issued near the
boundary (legal).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "audit-rl/1.0"

ISSUE_TYPES = ("duplicate_entry", "partial_allocation", "fx_rate", "period_cutoff")

# Реализовано в этой версии; fx_rate добавляется 20.11 (см. docs/RULES.md)
IMPLEMENTED_TYPES = ("duplicate_entry", "period_cutoff", "partial_allocation")

# Доля оплаченных счетов, к которым добавляется законный возврат (обязательный
# отвлекающий случай из спеки). Возврат — пара строк, дебиторку не затрагивает.
RETURNS_RATE = 0.18

CASH = "1010_cash"
AR = "1200_accounts_receivable"
REVENUE = "4000_sales_revenue"
NULL_TOKEN = "null"  # explicit nulls in CSV (spec: отсутствующие ссылки — явные null)

LEDGER_COLUMNS = [
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
]
INVOICE_COLUMNS = ["doc_id", "invoice_id", "entity_id", "issued_at", "due_at",
                   "currency", "total_minor", "line_items"]
PAYMENT_COLUMNS = ["doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"]

DEFAULT_CONFIG: dict[str, Any] = {
    "n_invoices": 80,
    "base_currency": "USD",
    "minor_unit_digits": 2,
    "timezone": "Asia/Almaty",
    "utc_offset": "+05:00",
    "period_start": "2026-01-01",
    "period_end": "2026-03-31",
    "issue_types": ["duplicate_entry", "period_cutoff"],
    "n_issues": 2,
    "clean": False,
    "distractors": True,
}


class EpisodeError(RuntimeError):
    """Raised when the generator cannot honour its own invariants."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _csv_text(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: (NULL_TOKEN if row.get(c) is None else row.get(c)) for c in columns})
    return buf.getvalue()


def _jsonl(rows: list[dict], columns: list[str]) -> str:
    return "\n".join(json.dumps({c: r.get(c) for c in columns}, ensure_ascii=False) for r in rows) + "\n"


def _episode_id(seed: int, cfg: dict) -> str:
    blob = f"{seed}|{json.dumps(cfg, sort_keys=True, default=str)}".encode()
    return "ep_" + hashlib.sha1(blob).hexdigest()[:12]


def _split_amount(total: int, parts: int, rng: random.Random) -> list[int]:
    """Split a minor-unit amount into ``parts`` integers that sum exactly to it."""
    if parts <= 1:
        return [total]
    cuts = sorted(rng.randrange(0, total + 1) for _ in range(parts - 1))
    out, prev = [], 0
    for c in cuts + [total]:
        out.append(c - prev)
        prev = c
    return [x for x in out if x > 0] or [total]


# --------------------------------------------------------------------------- #
# invariants (checked before AND after mutation)
# --------------------------------------------------------------------------- #
def _invariants(ledger: list[dict], invoices: list[dict], payments: list[dict],
                period_start: datetime, period_end: datetime) -> dict[str, Any]:
    sum_debits = sum(int(r["debit_minor"]) for r in ledger)
    sum_credits = sum(int(r["credit_minor"]) for r in ledger)

    def in_period(ts: str) -> bool:
        t = datetime.fromisoformat(ts)
        return period_start <= t <= period_end

    ar_balance = sum(int(r["debit_minor"]) - int(r["credit_minor"])
                     for r in ledger if r["account"] == AR and in_period(r["timestamp"]))
    invoiced = sum(int(i["total_minor"]) for i in invoices if in_period(i["issued_at"]))
    paid = sum(int(p["amount_minor"]) for p in payments if in_period(p["timestamp"]))
    return {
        "sum_debits": sum_debits,
        "sum_credits": sum_credits,
        "books_balanced": sum_debits == sum_credits,
        "ar_balance_in_period": ar_balance,
        "invoiced_in_period": invoiced,
        "paid_in_period": paid,
        "ar_matches_documents": ar_balance == invoiced - paid,
    }


def _verify_declared(before: dict, after: dict, mutations: list[dict]) -> dict[str, Any]:
    """Independent check that every declared discrepancy is really there.

    This is what the jury re-computes from the records alone:

    * ``sum(debit) - sum(credit)`` must equal the total of injected duplicates;
    * in-period AR balance minus ``invoiced - paid`` must equal the total of
      injected period-cutoff amounts **plus** injected partial allocations
      (both inflate the in-period AR balance);
    * a clean episode must satisfy both with zero.
    """
    imbalance = after["sum_debits"] - after["sum_credits"]
    ar_diff = after["ar_balance_in_period"] - (after["invoiced_in_period"] - after["paid_in_period"])

    dup_total = sum(int(m["expected_amount_minor"]) for m in mutations
                    if m["type"] == "duplicate_entry")
    cutoff_total = sum(int(m["expected_amount_minor"]) for m in mutations
                       if m["type"] == "period_cutoff")
    partial_total = sum(int(m["expected_amount_minor"]) for m in mutations
                        if m["type"] == "partial_allocation")

    problems = []
    # duplicated credit row -> sum(debit) - sum(credit) is negative by that amount
    if imbalance != -dup_total:
        problems.append(f"books imbalance {imbalance} != -declared duplicates {-dup_total}")
    # duplicate does not touch AR; cutoff removes an AR credit from the period;
    # partial allocation shrinks an AR credit inside the period
    if ar_diff != cutoff_total + partial_total:
        problems.append(f"AR discrepancy {ar_diff} != declared cutoffs {cutoff_total} "
                        f"+ partials {partial_total}")
    if problems:
        raise EpisodeError("declared discrepancy does not match records: " + "; ".join(problems))

    return {
        "books_imbalance": imbalance,
        "declared_duplicate_total": dup_total,
        "ar_period_discrepancy": ar_diff,
        "declared_period_cutoff_total": cutoff_total,
        "declared_partial_allocation_total": partial_total,
        "verified": True,
    }


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate(seed: int, config: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """Build one episode. Returns {"open": {...}, "closed": {...}} of file texts.

    Deterministic: same seed + config -> byte-identical files.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg["clean"]:
        issue_types: list[str] = []
        n_issues = 0
    else:
        issue_types = [t for t in cfg["issue_types"] if t in IMPLEMENTED_TYPES]
        n_issues = min(int(cfg["n_issues"]), len(issue_types))
    if n_issues and not issue_types:
        raise EpisodeError("no implemented issue type requested")

    rng = random.Random(seed)
    tz = timezone(timedelta(hours=int(cfg["utc_offset"][1:3])) * (1 if cfg["utc_offset"][0] == "+" else -1))
    period_start = datetime.fromisoformat(cfg["period_start"]).replace(tzinfo=tz)
    period_end = ((datetime.fromisoformat(cfg["period_end"]) + timedelta(days=1))
                  .replace(tzinfo=tz) - timedelta(seconds=1))
    n_days = max(1, (period_end.date() - period_start.date()).days + 1)
    currency = cfg["base_currency"]

    # ---- stage 1: hidden history ---------------------------------------- #
    n_entities = rng.randint(4, 8)
    entities = [f"ENT-{i:03d}" for i in range(1, n_entities + 1)]

    # amount pool -> legitimate repeated sums on purpose (distractor)
    pool = [rng.randint(50_00, 4_000_00) * 100 for _ in range(6)]

    invoices: list[dict] = []
    payments: list[dict] = []
    ledger: list[dict] = []
    row_no = 0

    def add_row(ts: datetime, account: str, debit: int, credit: int,
                entity: str, invoice_id: str | None, payment_id: str | None) -> dict:
        nonlocal row_no
        row_no += 1
        row = {
            "row_id": f"row_{row_no:05d}",
            "timestamp": _iso(ts),
            "account": account,
            "debit_minor": int(debit),
            "credit_minor": int(credit),
            "currency": currency,
            "entity_id": entity,
            "invoice_id": invoice_id,
            "payment_id": payment_id,
        }
        ledger.append(row)
        return row

    for i in range(int(cfg["n_invoices"])):
        day = rng.randrange(0, n_days)
        issued = period_start + timedelta(days=day, hours=rng.randrange(8, 18), minutes=rng.randrange(60))
        due = issued + timedelta(days=14)
        amount = rng.choice(pool)
        entity = rng.choice(entities)
        inv_id = f"INV-2026-{i + 1:04d}"
        parts = rng.randint(1, 3) if cfg["distractors"] else 1
        shares = _split_amount(amount, parts, rng)
        invoices.append({
            "doc_id": f"inv_doc_{i + 1:04d}",
            "invoice_id": inv_id,
            "entity_id": entity,
            "issued_at": _iso(issued),
            "due_at": _iso(due),
            "currency": currency,
            "total_minor": amount,
            # line items sum exactly to total_minor (document-level invariant)
            "line_items": [
                {"sku": f"SKU-{rng.randrange(100, 999)}", "qty": 1, "unit_price_minor": s}
                for s in shares
            ],
        })
        # sale: AR debit / revenue credit (balanced pair)
        add_row(issued, AR, amount, 0, entity, inv_id, None)
        add_row(issued, REVENUE, 0, amount, entity, inv_id, None)

    pay_no = 0
    returns_made = 0
    for inv in invoices:
        r = rng.random()
        if r < 0.25:
            continue  # unpaid invoice: legitimate
        issued = datetime.fromisoformat(inv["issued_at"])
        pay_dt = issued + timedelta(days=rng.randrange(1, 21), hours=rng.randrange(1, 8))
        if pay_dt > period_end:
            pay_dt = period_end - timedelta(hours=rng.randrange(1, 20))
        total = inv["total_minor"]
        if cfg["distractors"] and r < 0.50:
            shares = _split_amount(total, 2, rng)  # split payment: legitimate distractor
        else:
            shares = [total]
        for share in shares:
            pay_no += 1
            pay_id = f"PAY-2026-{pay_no:04d}"
            payments.append({
                "doc_id": f"pay_doc_{pay_no:04d}",
                "payment_id": pay_id,
                "timestamp": _iso(pay_dt),
                "currency": currency,
                "amount_minor": share,
                "invoice_refs": [inv["invoice_id"]],
            })
            entity = inv["entity_id"]
            add_row(pay_dt, CASH, share, 0, entity, inv["invoice_id"], pay_id)
            add_row(pay_dt, AR, 0, share, entity, inv["invoice_id"], pay_id)

        # Законный возврат: пара строк (дебет выручки / кредит кассы).
        # Баланс книг сохраняется, дебиторка не затрагивается, truth не попадает.
        # Первый оплаченный счёт получает возврат гарантированно: спека требует
        # возвраты как обязательный отвлекающий случай.
        if cfg["distractors"] and (returns_made == 0 or rng.random() < RETURNS_RATE):
            returns_made += 1
            refund = max(1, int(total * rng.randrange(15, 50) / 100))
            refund_dt = pay_dt + timedelta(days=rng.randrange(1, 10))
            if refund_dt > period_end:
                refund_dt = period_end - timedelta(minutes=rng.randrange(1, 30))
            add_row(refund_dt, REVENUE, refund, 0, inv["entity_id"], inv["invoice_id"], None)
            add_row(refund_dt, CASH, 0, refund, inv["entity_id"], inv["invoice_id"], None)

    # ---- verify the clean history --------------------------------------- #
    before = _invariants(ledger, invoices, payments, period_start, period_end)
    if not before["books_balanced"] or not before["ar_matches_documents"]:
        raise EpisodeError(f"clean books violate invariants: {before}")

    provenance_stages = [
        {"stage": "history", "detail": "balanced double-entry books generated and verified",
         "invariants": before},
    ]

    # ---- stage 3: causal mutations --------------------------------------- #
    truth: list[dict] = []
    mutations: list[dict] = []

    if n_issues:
        chosen = issue_types[:n_issues]
        used_payments: set[str] = set()
        for kind in chosen:
            if kind == "duplicate_entry":
                issue, mutation = _inject_duplicate(rng, ledger, invoices)
            elif kind == "period_cutoff":
                issue, mutation = _inject_period_cutoff(rng, ledger, payments,
                                                        period_start, period_end, used_payments)
                used_payments.update(x for x in issue["affected_ids"] if x.startswith("PAY-"))
            elif kind == "partial_allocation":
                issue, mutation = _inject_partial_allocation(rng, ledger, payments,
                                                             period_start, period_end,
                                                             used_payments)
                used_payments.update(x for x in issue["affected_ids"] if x.startswith("PAY-"))
            else:  # pragma: no cover - guarded above
                raise EpisodeError(f"issue type not implemented: {kind}")
            truth.append(issue)
            mutations.append(mutation)
            provenance_stages.append({"stage": "mutation", "issue_id": issue["issue_id"],
                                      "type": kind, "causal_change": mutation["causal_change"]})

    after = _invariants(ledger, invoices, payments, period_start, period_end)
    discrepancies = _verify_declared(before, after, mutations)
    provenance_stages.append({"stage": "verify", "invariants_after": after,
                              "declared_discrepancies": discrepancies})

    # ---- files ------------------------------------------------------------ #
    n_docs = len(invoices) + len(payments)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": _episode_id(seed, cfg),
        "base_currency": currency,
        "minor_unit_digits": int(cfg["minor_unit_digits"]),
        "period_start": period_start.isoformat(timespec="seconds"),
        "period_end": period_end.isoformat(timespec="seconds"),
        "timezone": cfg["timezone"],
        "utc_offset": cfg["utc_offset"],
        "n_records": len(ledger),
        "n_documents": n_docs,
        "tables": {"ledger": len(ledger), "invoices": len(invoices),
                   "payments": len(payments), "fx": "dates x pairs"},
        "rounding": "integer minor units; no rounding applied inside an episode",
        "null_token": NULL_TOKEN,
        "issue_type_vocabulary": list(ISSUE_TYPES),
        "page_limit": 50,
        "max_steps": 100,
    }

    policy = {
        "fx_rate_date": "period_end",
        "period_boundary": "inclusive: period_start 00:00:00 .. period_end 23:59:59 in episode timezone",
        "rounding": "all amounts are integers in minor units; FX rates are decimal strings",
        "comparison_rule": "amounts in different currencies must be converted with fx.csv before comparison",
        "evidence_rule": "a finding must cite record ids that exist; at least one full allowed evidence set is required by the evaluator",
    }

    fx_rows = []
    for offset in range(0, n_days, 7):
        d = (period_start + timedelta(days=offset)).date().isoformat()
        fx_rows.append({"date": d, "source_currency": "EUR", "target_currency": currency,
                        "rate": f"{0.9 + rng.randrange(0, 20) / 100:.4f}"})
        fx_rows.append({"date": d, "source_currency": "USD", "target_currency": currency,
                        "rate": "1.0000"})
    fx_csv = _csv_text(fx_rows, ["date", "source_currency", "target_currency", "rate"])

    open_files = {
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "ledger.csv": _csv_text(ledger, LEDGER_COLUMNS),
        "invoices.jsonl": _jsonl(invoices, INVOICE_COLUMNS),
        "payments.jsonl": _jsonl(payments, PAYMENT_COLUMNS),
        "fx.csv": fx_csv,
        "policy.json": json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
    }
    closed_files = {
        "truth.json": json.dumps({"episode_id": manifest["episode_id"], "issues": truth},
                                 ensure_ascii=False, indent=2) + "\n",
        "provenance.json": json.dumps(
            {"generator": "audit-rl", "schema_version": SCHEMA_VERSION, "seed": int(seed),
             "config": {k: v for k, v in cfg.items()}, "episode_id": manifest["episode_id"],
             "stages": provenance_stages, "mutations": mutations,
             "invariants_before": before, "invariants_after": after},
            ensure_ascii=False, indent=2, default=str) + "\n",
    }
    return {"open": open_files, "closed": closed_files}


# --------------------------------------------------------------------------- #
# the two implemented mutations
# --------------------------------------------------------------------------- #
def _inject_duplicate(rng: random.Random, ledger: list[dict],
                      invoices: list[dict]) -> tuple[dict, dict]:
    """Duplicate one revenue-credit row.

    The two rows of every business event stay balanced, but this copy adds an
    extra credit, so ``sum(debit) - sum(credit)`` becomes ``-amount``. Revenue in
    the period is double counted against the invoice it belongs to.
    """
    candidates = [r for r in ledger if r["account"] == REVENUE and int(r["credit_minor"]) > 0]
    src = rng.choice(candidates)
    dup = dict(src, row_id=f"row_{len(ledger) + 1:05d}")
    ledger.append(dup)
    amount = int(dup["credit_minor"])
    issue_id = "iss_duplicate_0001"
    issue = {
        "issue_id": issue_id,
        "type": "duplicate_entry",
        "affected_ids": [dup["row_id"]],
        "expected_amount_minor": amount,
        "currency": dup["currency"],
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"ledger:{dup['row_id']}", f"ledger:{src['row_id']}", f"invoice:{dup['invoice_id']}"],
            [f"ledger:{dup['row_id']}", f"ledger:{src['row_id']}"],
        ],
        "statement": "строка выручки продублирована: два идентичных поля записи на одну "
                     "счёт-фактуру, равенство Σдебет=Σкредит нарушено на сумму строки",
    }
    mutation = {
        "issue_id": issue_id,
        "type": "duplicate_entry",
        "causal_change": f"copied ledger row {src['row_id']} -> {dup['row_id']} (credit side only)",
        "before": {"sum_debits_minus_credits": 0},
        "after": {"sum_debits_minus_credits": -amount},
        "expected_amount_minor": amount,
        "affected_ids": [dup["row_id"]],
    }
    return issue, mutation


def _inject_period_cutoff(rng: random.Random, ledger: list[dict], payments: list[dict],
                          period_start: datetime, period_end: datetime,
                          exclude: set[str] | None = None) -> tuple[dict, dict]:
    """Re-date the *ledger rows* of one in-period payment across period_end.

    The payment document keeps its true timestamp, so documents and ledger
    disagree by exactly the payment amount *inside* the period. That is the
    declared discrepancy, and it is provable from records the agent can read
    (payment doc + ledger row + policy boundary) — no hidden fact is required.
    """
    by_payment: dict[str, list[dict]] = {}
    for row in ledger:
        if row["payment_id"]:
            by_payment.setdefault(row["payment_id"], []).append(row)

    eligible = []
    for pay in payments:
        rows = by_payment.get(pay["payment_id"], [])
        ts = datetime.fromisoformat(pay["timestamp"])
        if (len(rows) == 2 and pay["payment_id"] not in (exclude or set())
                and period_start < ts < period_end - timedelta(days=2)):
            eligible.append((pay, rows))
    if not eligible:
        raise EpisodeError("no eligible payment for period_cutoff")
    pay, rows = eligible[rng.randrange(len(eligible))]

    old_ts = pay["timestamp"]
    new_ts = period_end + timedelta(days=1, hours=12)
    for row in rows:
        row["timestamp"] = _iso(new_ts)

    amount = int(pay["amount_minor"])
    ar_row = next(r for r in rows if r["account"] == AR)
    cash_row = next(r for r in rows if r["account"] == CASH)
    issue_id = "iss_period_0001"
    issue = {
        "issue_id": issue_id,
        "type": "period_cutoff",
        "affected_ids": sorted([pay["payment_id"], ar_row["row_id"], cash_row["row_id"]]),
        "expected_amount_minor": amount,
        "currency": pay["currency"],
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"payment:{pay['payment_id']}", f"ledger:{ar_row['row_id']}", "policy:period"],
            [f"payment:{pay['payment_id']}", f"ledger:{cash_row['row_id']}",
             f"ledger:{ar_row['row_id']}", "policy:period"],
        ],
        "statement": f"строки платежа {pay['payment_id']} отнесены к периоду после period_end "
                     f"при документе с настоящей датой; баланс AR внутри периода завышен на сумму",
    }
    mutation = {
        "issue_id": issue_id,
        "type": "period_cutoff",
        "causal_change": f"ledger rows of payment {pay['payment_id']} re-dated from {old_ts} "
                         f"to {_iso(new_ts)} (across period_end); payment document unchanged",
        "before": {"rows_timestamp": old_ts, "payment_document_timestamp": old_ts},
        "after": {"rows_timestamp": _iso(new_ts), "payment_document_timestamp": old_ts},
        "expected_amount_minor": amount,
        "affected_ids": issue["affected_ids"],
    }
    return issue, mutation


# --------------------------------------------------------------------------- #
# partial allocation: пропущенное/частичное распределение платежа
# --------------------------------------------------------------------------- #
def _inject_partial_allocation(rng: random.Random, ledger: list[dict], payments: list[dict],
                               period_start: datetime, period_end: datetime,
                               exclude: set[str] | None = None) -> tuple[dict, dict]:
    """Документ платежа остаётся с полной суммой, обе строки журнала уменьшены.

    Книги остаются сбалансированными (уменьшены и дебет кассы, и кредит
    дебеторки), но дебиторка внутри периода завышена ровно на недостающую
    сумму. Расхождение доказуемо по открытой паре «платёж ↔ строки журнала»
    и по сверке «платежи по счёту = погашения дебеторки».
    """
    by_payment: dict[str, list[dict]] = {}
    for row in ledger:
        if row["payment_id"]:
            by_payment.setdefault(row["payment_id"], []).append(row)

    excluded = exclude or set()
    eligible = []
    for pay in payments:
        rows = by_payment.get(pay["payment_id"], [])
        if pay["payment_id"] in excluded or len(rows) != 2:
            continue
        ts = datetime.fromisoformat(pay["timestamp"])
        allocated = sum(int(r["credit_minor"]) for r in rows if r["account"] == AR)
        if not (period_start <= ts <= period_end):
            continue
        if allocated != int(pay["amount_minor"]):
            continue  # платеж уже мутирован другим типом
        eligible.append((pay, rows))
    if not eligible:
        raise EpisodeError("no eligible payment for partial_allocation")
    pay, rows = eligible[rng.randrange(len(eligible))]

    cash_row = next(r for r in rows if r["account"] == CASH)
    ar_row = next(r for r in rows if r["account"] == AR)
    total = int(pay["amount_minor"])
    delta = min(max(1, total * rng.randrange(10, 45) // 100), total - 1)
    cash_row["debit_minor"] = int(cash_row["debit_minor"]) - delta
    ar_row["credit_minor"] = int(ar_row["credit_minor"]) - delta

    issue_id = "iss_partial_0001"
    issue = {
        "issue_id": issue_id,
        "type": "partial_allocation",
        "affected_ids": sorted([pay["payment_id"], cash_row["row_id"], ar_row["row_id"]]),
        "expected_amount_minor": delta,
        "currency": pay["currency"],
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"payment:{pay['payment_id']}", f"ledger:{ar_row['row_id']}",
             f"invoice:{ar_row['invoice_id']}"],
            [f"payment:{pay['payment_id']}", f"ledger:{cash_row['row_id']}",
             f"ledger:{ar_row['row_id']}"],
        ],
        "statement": f"платёж {pay['payment_id']} распределён в журнале на "
                     f"{total - delta} при документе на {total}: недостаёт {delta}",
    }
    mutation = {
        "issue_id": issue_id,
        "type": "partial_allocation",
        "causal_change": f"payment {pay['payment_id']} document keeps {total}, "
                         f"both ledger rows reduced by {delta}",
        "before": {"payment_amount_minor": total, "ledger_allocated_minor": total},
        "after": {"payment_amount_minor": total, "ledger_allocated_minor": total - delta},
        "expected_amount_minor": delta,
        "affected_ids": issue["affected_ids"],
    }
    return issue, mutation


# --------------------------------------------------------------------------- #
# loading a saved episode
# --------------------------------------------------------------------------- #
def load_episode(episode_dir: str | Path) -> dict[str, dict[str, str]]:
    """Read open + closed files from a saved episode directory."""
    root = Path(episode_dir)
    open_names = ["manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                  "fx.csv", "policy.json"]
    closed_names = ["truth.json", "provenance.json"]
    missing = [n for n in open_names if not (root / n).exists()]
    if missing:
        raise EpisodeError(f"episode directory {root} is missing: {missing}")
    return {
        "open": {n: (root / n).read_text(encoding="utf-8") for n in open_names},
        "closed": {n: (root / n).read_text(encoding="utf-8") for n in closed_names
                   if (root / n).exists()},
    }


def save_episode(files: dict[str, dict[str, str]], out_dir: str | Path) -> Path:
    """Write an episode to disk: open/ and closed/ subdirectories."""
    root = Path(out_dir)
    (root / "open").mkdir(parents=True, exist_ok=True)
    for name, text in files["open"].items():
        (root / "open" / name).write_text(text, encoding="utf-8")
    if files.get("closed"):
        (root / "closed").mkdir(parents=True, exist_ok=True)
        for name, text in files["closed"].items():
            (root / "closed" / name).write_text(text, encoding="utf-8")
    return root
