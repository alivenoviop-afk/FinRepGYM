"""Synthetic company generator (spec sections 02 and 03).

Pipeline, in this exact order:

1. **Hidden history** — a consistent double-entry story: invoices, payments,
   counterparties, returns. Every business event produces a *pair* of ledger rows.
2. **Documents** — ``ledger.csv``, ``invoices.jsonl``, ``payments.jsonl``,
   ``fx.csv``, ``policy.json``, ``manifest.json``.
3. **Causal mutations** — errors are injected *after* the clean history is
   verified; each mutation records its cause and its exact effect, and
   ``truth.json`` describes the real discrepancy plus available evidence.

Currency model (все 4 типа ошибок)
----------------------------------
* Documents keep their own currency: invoices in one of ``currencies``,
  payments in their own (possibly different) currency.
* A payment posted **in the invoice currency** keeps both ledger rows in that
  currency (no conversion, exact per-currency balance).
* A payment in a **different currency** is posted entirely in the **base
  currency**: both rows use the single converted integer, so
  ``sum(debit) == sum(credit)`` and the AR reconciliation hold *exactly* —
  no chain-rounding drift.
* Conversion always uses ``policy.fx_rate_date`` (= period_end) and the exact
  decimal rates published in ``fx.csv``.

Implemented issue types (all four required types):

* ``duplicate_entry``     — a revenue credit row is copied; books stop balancing.
* ``period_cutoff``       — a payment's ledger rows are re-dated past period_end.
* ``partial_allocation``  — the payment document keeps its amount, both ledger
  rows are reduced; in-period AR is inflated by the missing part.
* ``fx_rate``             — the converted AR row of a cross-currency payment is
  recomputed with a non-policy rate; base-currency books stop balancing by the
  conversion delta.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "finrepgym/1.0"

ISSUE_TYPES = ("duplicate_entry", "partial_allocation", "fx_rate", "period_cutoff")
IMPLEMENTED_TYPES = ISSUE_TYPES  # все четыре типа требуемых спекой реализованы

RETURNS_RATE = 0.18          # доля оплаченных счетов с законным возвратом
CROSS_CCY_RATE = 0.30        # доля платежей в другой валюте (сценарий спеки)
RATE_PRECISION = 6           # знаков после запятой в fx.csv

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

MINOR_DIGITS = {"USD": 2, "EUR": 2, "KZT": 2}
# USD за единицу; значение = сколько USD за 1 единицу валюты
USD_PER_UNIT = {"USD": Decimal("1"), "EUR": Decimal("1.08"), "KZT": Decimal("0.00185")}

DEFAULT_CONFIG: dict[str, Any] = {
    "n_invoices": 80,
    "base_currency": "USD",
    "currencies": ["USD", "EUR", "KZT"],
    "minor_unit_digits": 2,
    "timezone": "Asia/Almaty",
    "utc_offset": "+05:00",
    "period_start": "2026-01-01",
    "period_end": "2026-03-31",
    "issue_types": ["duplicate_entry", "period_cutoff"],
    "n_issues": 2,
    "scenarios": [],          # ["S1"] / ["S2"] / ["S1","S2"]: сценарии взаимодействия
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
    if parts <= 1:
        return [total]
    cuts = sorted(rng.randrange(0, total + 1) for _ in range(parts - 1))
    out, prev = [], 0
    for c in cuts + [total]:
        out.append(c - prev)
        prev = c
    return [x for x in out if x > 0] or [total]


# --------------------------------------------------------------------------- #
# FX engine: единый источник курсов и конвертации
# --------------------------------------------------------------------------- #
class FxTable:
    """Курсы всех направленных пар на наборе дат. Строки fx.csv — источник истины."""

    def __init__(self, rng: random.Random, currencies: list[str],
                 period_start: date, period_end: date) -> None:
        self.currencies = list(currencies)
        dates = []
        d = period_start
        while d <= period_end:
            dates.append(d)
            d += timedelta(days=7)
        if period_end not in dates:
            dates.append(period_end)  # policy:fx_rate_date обязан существовать в fx.csv
        self.dates = sorted(dates)

        # случайное блуждание вокруг базовых котировок
        level = {c: USD_PER_UNIT[c] for c in currencies}
        self.usd_per: dict[date, dict[str, Decimal]] = {}
        for day in self.dates:
            for c in currencies:
                if c != "USD":
                    jitter = Decimal(str(1 + rng.uniform(-0.02, 0.02)))
                    level[c] = level[c] * jitter
            self.usd_per[day] = {c: level[c] for c in currencies}

        self.rates: dict[tuple[date, str, str], Decimal] = {}
        for day in self.dates:
            for a in currencies:
                for b in currencies:
                    if a == b:
                        continue
                    raw = self.usd_per[day][a] / self.usd_per[day][b]
                    self.rates[(day, a, b)] = raw.quantize(
                        Decimal(1).scaleb(-RATE_PRECISION), rounding=ROUND_HALF_UP)

    def rate(self, day: date, src: str, tgt: str) -> Decimal:
        if src == tgt:
            return Decimal(1)
        return self.rates[(day, src, tgt)]

    def convert(self, amount_minor: int, src: str, tgt: str, day: date) -> int:
        if src == tgt:
            return int(amount_minor)
        value = (Decimal(int(amount_minor)) * self.rate(day, src, tgt))
        return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))

    def csv_rows(self) -> list[dict]:
        rows = []
        for (day, a, b) in sorted(self.rates, key=lambda k: (k[0], k[1], k[2])):
            rows.append({"date": day.isoformat(), "source_currency": a,
                         "target_currency": b, "rate": str(self.rates[(day, a, b)])})
        return rows


# --------------------------------------------------------------------------- #
# invariants (считаются в БАЗОВОЙ валюте по курсу policy.fx_rate_date)
# --------------------------------------------------------------------------- #
def _invariants(ledger: list[dict], invoices: list[dict], payments: list[dict],
                period_start: datetime, period_end: datetime, fx: FxTable,
                base: str) -> dict[str, Any]:
    policy_day = period_end.date()

    def to_base(amount: int, ccy: str) -> int:
        return fx.convert(int(amount), ccy, base, policy_day)

    def in_period(ts: str) -> bool:
        t = datetime.fromisoformat(ts)
        return period_start <= t <= period_end

    sum_debits = sum(to_base(r["debit_minor"], r["currency"]) for r in ledger)
    sum_credits = sum(to_base(r["credit_minor"], r["currency"]) for r in ledger)
    ar_balance = sum(to_base(r["debit_minor"], r["currency"]) - to_base(r["credit_minor"], r["currency"])
                     for r in ledger if r["account"] == AR and in_period(r["timestamp"]))
    invoiced = sum(to_base(i["total_minor"], i["currency"]) for i in invoices
                   if in_period(i["issued_at"]))
    paid = sum(to_base(p["amount_minor"], p["currency"]) for p in payments
               if in_period(p["timestamp"]))
    return {
        "sum_debits": sum_debits,
        "sum_credits": sum_credits,
        "books_balanced": sum_debits == sum_credits,
        "ar_balance_in_period": ar_balance,
        "invoiced_in_period": invoiced,
        "paid_in_period": paid,
        "ar_matches_documents": ar_balance == invoiced - paid,
        "base_currency": base,
    }


def _verify_declared(before: dict, after: dict, mutations: list[dict]) -> dict[str, Any]:
    """Независимая проверка: сумма заявленных эффектов == наблюдаемым расхождениям.

    Каждая мутация несёт подписанные эффекты в базовой валюте:
    ``effect_imbalance`` (Σдебет − Σкредит) и ``effect_ar_diff``
    (AR_в_периоде − (выставлено − оплачено)).
    """
    imbalance = after["sum_debits"] - after["sum_credits"]
    ar_diff = after["ar_balance_in_period"] - (after["invoiced_in_period"] - after["paid_in_period"])

    declared_imbalance = sum(int(m["effect_imbalance"]) for m in mutations)
    declared_ar = sum(int(m["effect_ar_diff"]) for m in mutations)

    problems = []
    if imbalance != declared_imbalance:
        problems.append(f"books imbalance {imbalance} != declared {declared_imbalance}")
    if ar_diff != declared_ar:
        problems.append(f"AR discrepancy {ar_diff} != declared {declared_ar}")
    if not before["books_balanced"] or not before["ar_matches_documents"]:
        problems.append("clean history was not balanced before mutation")
    if problems:
        raise EpisodeError("declared discrepancy does not match records: " + "; ".join(problems))

    return {
        "books_imbalance": imbalance,
        "declared_imbalance": declared_imbalance,
        "ar_period_discrepancy": ar_diff,
        "declared_ar_discrepancy": declared_ar,
        "verified": True,
    }


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate(seed: int, config: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """Build one episode. Deterministic: same seed + config -> byte-identical files."""
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
    sign = 1 if cfg["utc_offset"][0] == "+" else -1
    hours = int(cfg["utc_offset"][1:3])
    minutes = int(cfg["utc_offset"][4:6]) if len(cfg["utc_offset"]) >= 6 else 0
    tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    period_start = datetime.fromisoformat(cfg["period_start"]).replace(tzinfo=tz)
    period_end = ((datetime.fromisoformat(cfg["period_end"]) + timedelta(days=1))
                  .replace(tzinfo=tz) - timedelta(seconds=1))
    n_days = max(1, (period_end.date() - period_start.date()).days + 1)

    currencies = list(cfg["currencies"])
    base = cfg["base_currency"]
    if base not in currencies:
        currencies = [base] + currencies

    # Явная проверка противоречий конфигурации: тип fx_rate и сценарии S1/S2
    # требуют минимум двух валют (иначе конвертация невозможна в принципе).
    wants_conversion = (not cfg["clean"]) and (
        "fx_rate" in [t for t in cfg.get("issue_types", []) if t in ISSUE_TYPES]
        or any(s in ("S1", "S2") for s in (cfg.get("scenarios") or [])))
    if wants_conversion and len(currencies) < 2:
        raise EpisodeError(
            "противоречие конфигурации: для типа 'fx_rate' и сценариев S1/S2 "
            "нужно минимум 2 валюты в config['currencies']")

    fx = FxTable(rng, currencies, period_start.date(), period_end.date())
    policy_day = period_end.date()

    # Вариативность «имён и ID» (спека 02: «Варьируйте имена, ID, порядок...»):
    # префиксы и старт нумерации выбираются для каждого эпизода из своего seed.
    row_prefix = rng.choice(("row_", "rec_", "entry_", "ln_"))
    inv_prefix = rng.choice(("INV-", "FACT-", "BILL-", "INVC-"))
    pay_prefix = "PAY-"  # пространство платежей устойчиво; варьируется номер (см. ниже)
    ent_prefix = rng.choice(("ENT-", "CPTY-", "PARTNER-", "AGT-"))
    inv_doc_prefix = rng.choice(("inv_doc_", "bill_doc_", "fact_doc_"))
    pay_doc_prefix = rng.choice(("pay_doc_", "trf_doc_", "pmt_doc_"))
    id_offset = rng.randrange(1, 900)
    year_tag = period_start.year

    # ---- stage 1: hidden history ----------------------------------------- #
    n_entities = rng.randint(4, 8)
    entities = [f"{ent_prefix}{i:03d}" for i in range(1, n_entities + 1)]
    pool = [rng.randint(50_00, 400_00) * 100 for _ in range(6)]

    invoices: list[dict] = []
    payments: list[dict] = []
    ledger: list[dict] = []
    row_no = 0

    def add_row(ts: datetime, account: str, debit: int, credit: int, ccy: str,
                entity: str, invoice_id: str | None, payment_id: str | None) -> dict:
        nonlocal row_no
        row_no += 1
        row = {
            "row_id": f"{row_prefix}{id_offset + row_no:05d}",
            "timestamp": _iso(ts),
            "account": account,
            "debit_minor": int(debit),
            "credit_minor": int(credit),
            "currency": ccy,
            "entity_id": entity,
            "invoice_id": invoice_id,
            "payment_id": payment_id,
        }
        ledger.append(row)
        return row

    weights = ([6, 3, 2] + [2] * 8)[: len(currencies)]
    for i in range(int(cfg["n_invoices"])):
        day = rng.randrange(0, n_days)
        issued = period_start + timedelta(days=day, hours=rng.randrange(8, 18), minutes=rng.randrange(60))
        due = issued + timedelta(days=14)
        amount = rng.choice(pool)
        inv_ccy = rng.choices(currencies, weights=weights, k=1)[0]
        entity = rng.choice(entities)
        inv_id = f"{inv_prefix}{year_tag}-{i + 1:04d}"
        parts = rng.randint(1, 3) if cfg["distractors"] else 1
        shares = _split_amount(amount, parts, rng)
        invoices.append({
            "doc_id": f"{inv_doc_prefix}{i + 1:04d}",
            "invoice_id": inv_id,
            "entity_id": entity,
            "issued_at": _iso(issued),
            "due_at": _iso(due),
            "currency": inv_ccy,
            "total_minor": amount,
            "line_items": [
                {"sku": f"SKU-{rng.randrange(100, 999)}", "qty": 1, "unit_price_minor": s}
                for s in shares
            ],
        })
        add_row(issued, AR, amount, 0, inv_ccy, entity, inv_id, None)
        add_row(issued, REVENUE, 0, amount, inv_ccy, entity, inv_id, None)

    pay_no = 0
    returns_made = 0
    cross_made = 0
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
            shares = _split_amount(total, 2, rng)   # split payment: legitimate
        else:
            shares = [total]

        # платёж в ДРУГОЙ валюте — обязательный сценарий спеки; постинг в базовую.
        # Первый оплаченный счёт гарантированно получает кросс-валютный платёж.
        # Кросс-валютный платёж — НЕ отвлекающий случай, а подложка для типа
        # fx_rate и сценариев S1/S2: генерируем всегда, пока есть вторая валюта.
        # Иначе при distractors=false кандидатов не остаётся и ядро падало бы
        # на валидном сочетании параметров.
        others = [c for c in currencies if c != inv["currency"]]
        cross = bool(others) and (cross_made == 0 or rng.random() < CROSS_CCY_RATE)
        if cross:
            cross_made += 1
            # предпочитаем НЕ-базовую валюту: тогда постинг реально конвертируется,
            # и для типа fx_rate / сценариев всегда есть кандидат (иначе курс не применяется)
            foreign = [c for c in others if c != base]
            pay_ccy = rng.choice(foreign or others)
        else:
            pay_ccy = inv["currency"]

        for share in shares:
            pay_no += 1
            # префикс PAY- сохраняем устойчивым (пространство платежей), а номер
            # получает смещение эпизода -> ID варьируется от seed к seed
            pay_id = f"PAY-{year_tag}-{id_offset + pay_no:04d}"
            payments.append({
                "doc_id": f"{pay_doc_prefix}{pay_no:04d}",
                "payment_id": pay_id,
                "timestamp": _iso(pay_dt),
                "currency": pay_ccy,
                "amount_minor": share,
                "invoice_refs": [inv["invoice_id"]],
            })
            entity = inv["entity_id"]
            if pay_ccy == inv["currency"]:
                add_row(pay_dt, CASH, share, 0, pay_ccy, entity, inv["invoice_id"], pay_id)
                add_row(pay_dt, AR, 0, share, inv["currency"], entity, inv["invoice_id"], pay_id)
            else:
                posted = fx.convert(share, pay_ccy, base, policy_day)
                add_row(pay_dt, CASH, posted, 0, base, entity, inv["invoice_id"], pay_id)
                add_row(pay_dt, AR, 0, posted, base, entity, inv["invoice_id"], pay_id)

        # Законный возврат: пара строк, дебиторку не затрагивает, в truth не попадает.
        if cfg["distractors"] and (returns_made == 0 or rng.random() < RETURNS_RATE):
            returns_made += 1
            refund = max(1, int(total * rng.randrange(15, 50) / 100))
            refund_dt = pay_dt + timedelta(days=rng.randrange(1, 10))
            if refund_dt > period_end:
                refund_dt = period_end - timedelta(minutes=rng.randrange(1, 30))
            add_row(refund_dt, REVENUE, refund, 0, inv["currency"],
                    inv["entity_id"], inv["invoice_id"], None)
            add_row(refund_dt, CASH, 0, refund, inv["currency"],
                    inv["entity_id"], inv["invoice_id"], None)

    # ---- verify the clean history ---------------------------------------- #
    before = _invariants(ledger, invoices, payments, period_start, period_end, fx, base)
    if not before["books_balanced"] or not before["ar_matches_documents"]:
        raise EpisodeError(f"clean books violate invariants: {before}")
    provenance_stages = [{"stage": "history", "detail": "balanced books verified in base currency",
                          "invariants": before}]

    # ---- stage 3: causal mutations --------------------------------------- #
    truth: list[dict] = []
    mutations: list[dict] = []
    used_payments: set[str] = set()
    payment_id_set = {p["payment_id"] for p in payments}
    chosen = issue_types[:n_issues]
    for kind in chosen:
        if kind == "duplicate_entry":
            issue, mutation = _inject_duplicate(rng, ledger, invoices, fx, policy_day)
        elif kind == "period_cutoff":
            issue, mutation = _inject_period_cutoff(rng, ledger, payments,
                                                    period_start, period_end, used_payments,
                                                    fx, policy_day)
            used_payments.update(x for x in issue["affected_ids"] if x in payment_id_set)
        elif kind == "partial_allocation":
            issue, mutation = _inject_partial_allocation(rng, ledger, payments,
                                                         period_start, period_end,
                                                         used_payments, fx, policy_day, base)
            used_payments.update(x for x in issue["affected_ids"] if x in payment_id_set)
        elif kind == "fx_rate":
            issue, mutation = _inject_fx_rate(rng, ledger, payments, used_payments,
                                              fx, policy_day, base)
            used_payments.update(x for x in issue["affected_ids"] if x in payment_id_set)
        else:  # pragma: no cover - guarded above
            raise EpisodeError(f"issue type not implemented: {kind}")
        truth.append(issue)
        mutations.append(mutation)

    # ---- сценарии взаимодействия (спека: два обязательных) ---------------- #
    # clean=true главенствует: чистый эпизод остаётся пустым даже при заданных
    # сценариях (иначе флаг чистоты противоречил бы сам себе).
    for scen in ([] if cfg["clean"] else
                 (s for s in (cfg.get("scenarios") or []) if s in ("S1", "S2"))):
        if scen == "S1":
            issues, muts, pay_id = _inject_scenario_s1(rng, ledger, payments,
                                                        period_start, period_end,
                                                        used_payments, fx, policy_day, base)
        else:
            issues, muts, pay_id = _inject_scenario_s2(rng, ledger, payments,
                                                        period_start, period_end,
                                                        used_payments, fx, policy_day, base)
        used_payments.add(pay_id)
        truth.extend(issues)
        mutations.extend(muts)
        for issue, mutation in zip(issues, muts):
            provenance_stages.append({"stage": "mutation", "issue_id": issue["issue_id"],
                                      "type": issue["type"], "scenario": scen,
                                      "causal_change": mutation["causal_change"]})

    after = _invariants(ledger, invoices, payments, period_start, period_end, fx, base)
    discrepancies = _verify_declared(before, after, mutations)
    provenance_stages.append({"stage": "verify", "invariants_after": after,
                              "declared_discrepancies": discrepancies})

    # ---- files ------------------------------------------------------------ #
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": _episode_id(seed, cfg),
        "base_currency": base,
        "currencies": {c: MINOR_DIGITS.get(c, 2) for c in currencies},
        "minor_unit_digits": int(cfg["minor_unit_digits"]),
        "period_start": period_start.isoformat(timespec="seconds"),
        "period_end": period_end.isoformat(timespec="seconds"),
        "timezone": cfg["timezone"],
        "utc_offset": cfg["utc_offset"],
        "n_records": len(ledger),
        "n_documents": len(invoices) + len(payments),
        "tables": {"ledger": len(ledger), "invoices": len(invoices),
                   "payments": len(payments), "fx": len(fx.csv_rows())},
        "rounding": "integer minor units; FX rates are decimal strings, "
                    "half-up to minor unit on conversion",
        "null_token": NULL_TOKEN,
        "issue_type_vocabulary": list(ISSUE_TYPES),
        "page_limit": 50,
        "max_steps": 100,
    }

    policy = {
        "fx_rate_date": "period_end",
        "period_boundary": "inclusive: period_start 00:00:00 .. period_end 23:59:59 in episode timezone",
        "rounding": "all amounts are integers in minor units; FX rates are decimal strings; "
                    "conversion rounds half-up to the target currency's minor unit",
        "comparison_rule": "amounts in different currencies must be converted with fx.csv "
                           "before comparison",
        "evidence_rule": "a finding must cite record ids that exist; at least one full "
                         "allowed evidence set is required by the evaluator",
    }

    # Вариативность порядка (спека 02: «Варьируйте ... порядок ...»).
    # Мутации уже внесены и проверены; порядок на равенства и эталон не влияет.
    rng.shuffle(ledger)
    rng.shuffle(invoices)
    rng.shuffle(payments)

    open_files = {
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "ledger.csv": _csv_text(ledger, LEDGER_COLUMNS),
        "invoices.jsonl": _jsonl(invoices, INVOICE_COLUMNS),
        "payments.jsonl": _jsonl(payments, PAYMENT_COLUMNS),
        "fx.csv": _csv_text(fx.csv_rows(), ["date", "source_currency", "target_currency", "rate"]),
        "policy.json": json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
    }
    closed_files = {
        "truth.json": json.dumps({"episode_id": manifest["episode_id"], "issues": truth},
                                 ensure_ascii=False, indent=2) + "\n",
        "provenance.json": json.dumps(
            {"generator": "FinRepGYM", "schema_version": SCHEMA_VERSION, "seed": int(seed),
             "config": {k: v for k, v in cfg.items()}, "episode_id": manifest["episode_id"],
             "stages": provenance_stages, "mutations": mutations,
             "invariants_before": before, "invariants_after": after},
            ensure_ascii=False, indent=2, default=str) + "\n",
    }
    return {"open": open_files, "closed": closed_files}


# --------------------------------------------------------------------------- #
# the four mutations (каждая несёт подписанные эффекты в базовой валюте)
# --------------------------------------------------------------------------- #
def _inject_duplicate(rng: random.Random, ledger: list[dict], invoices: list[dict],
                      fx: FxTable, policy_day: date) -> tuple[dict, dict]:
    """Копия кредитной строки выручки: Σдебет − Σкредит = −сумма (в базовой)."""
    candidates = [r for r in ledger if r["account"] == REVENUE and int(r["credit_minor"]) > 0]
    src = rng.choice(candidates)
    # новый id в том же стиле, что и у источника, но гарантированно не занятый
    # (без подсказывающих слов вроде "dup" — иначе дубль читается по имени)
    stem = src["row_id"].rstrip("0123456789") or "row_"
    digits = src["row_id"][len(stem):]
    dup = dict(src, row_id=f"{stem}{int(digits) + 5000:0{len(digits)}d}")
    ledger.append(dup)
    amount = int(dup["credit_minor"])
    amount_base = fx.convert(amount, dup["currency"], "USD", policy_day)
    issue_id = "iss_duplicate_0001"
    issue = {
        "issue_id": issue_id,
        "type": "duplicate_entry",
        # Затронуты ОБЕ строки пары: после перестановки/переименования из данных
        # невозможно определить, какая из идентичных строк была «копией»,
        # поэтому эталон указывает на пару — это и делает ответ инвариантным.
        "affected_ids": sorted([src["row_id"], dup["row_id"]]),
        "expected_amount_minor": amount,
        "currency": dup["currency"],
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"ledger:{dup['row_id']}", f"ledger:{src['row_id']}", f"invoice:{dup['invoice_id']}"],
            [f"ledger:{dup['row_id']}", f"ledger:{src['row_id']}"],
        ],
        "statement": "строка выручки продублирована: две идентичные записи на одну "
                     "счёт-фактуру, равенство дебет=кредит нарушено на сумму строки",
    }
    mutation = {
        "issue_id": issue_id,
        "type": "duplicate_entry",
        "causal_change": f"copied ledger row {src['row_id']} -> {dup['row_id']} (credit side)",
        "before": {"sum_debits_minus_credits": 0},
        "after": {"sum_debits_minus_credits": -amount_base},
        "expected_amount_minor": amount,
        "currency": dup["currency"],
        "effect_imbalance": -amount_base,
        "effect_ar_diff": 0,
        "affected_ids": sorted([src["row_id"], dup["row_id"]]),
    }
    return issue, mutation


def _inject_period_cutoff(rng: random.Random, ledger: list[dict], payments: list[dict],
                          period_start: datetime, period_end: datetime,
                          exclude: set[str] | None, fx: FxTable,
                          policy_day: date) -> tuple[dict, dict]:
    """Строки платежа переезжают за period_end; документ остаётся с настоящей датой."""
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

    ar_row = next(r for r in rows if r["account"] == AR)
    amount = int(pay["amount_minor"])
    ar_base = fx.convert(int(ar_row["credit_minor"]), ar_row["currency"], "USD", policy_day)
    issue_id = "iss_period_0001"
    issue = {
        "issue_id": issue_id,
        "type": "period_cutoff",
        "affected_ids": sorted([pay["payment_id"], ar_row["row_id"]] +
                               [r["row_id"] for r in rows if r is not ar_row]),
        "expected_amount_minor": amount,
        "currency": pay["currency"],
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"payment:{pay['payment_id']}", f"ledger:{ar_row['row_id']}", "policy:period"],
            [f"payment:{pay['payment_id']}",
             f"ledger:{next(r for r in rows if r is not ar_row)['row_id']}",
             f"ledger:{ar_row['row_id']}", "policy:period"],
        ],
        "statement": f"строки платежа {pay['payment_id']} отнесены к периоду после "
                     f"period_end при документе с настоящей датой",
    }
    mutation = {
        "issue_id": issue_id,
        "type": "period_cutoff",
        "causal_change": f"ledger rows of payment {pay['payment_id']} re-dated from {old_ts} "
                         f"to {_iso(new_ts)}; payment document unchanged",
        "before": {"rows_timestamp": old_ts, "payment_document_timestamp": old_ts},
        "after": {"rows_timestamp": _iso(new_ts), "payment_document_timestamp": old_ts},
        "expected_amount_minor": amount,
        "currency": pay["currency"],
        "effect_imbalance": 0,
        "effect_ar_diff": ar_base,  # кредит дебеторки вышел из периода -> AR +amount
        "affected_ids": issue["affected_ids"],
    }
    return issue, mutation


def _inject_partial_allocation(rng: random.Random, ledger: list[dict], payments: list[dict],
                               period_start: datetime, period_end: datetime,
                               exclude: set[str] | None, fx: FxTable,
                               policy_day: date, base: str) -> tuple[dict, dict]:
    """Документ платежа полон, обе строки журнала уменьшены на недостающую сумму."""
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
        cash_row = next((r for r in rows if r["account"] == CASH), None)
        ar_row = next((r for r in rows if r["account"] == AR), None)
        if cash_row is None or ar_row is None:
            continue
        # частичное распределение живёт только внутри одной валюты (иначе путается с fx)
        if not (cash_row["currency"] == ar_row["currency"] == pay["currency"]):
            continue
        ts = datetime.fromisoformat(pay["timestamp"])
        if not (period_start <= ts <= period_end):
            continue
        if int(ar_row["credit_minor"]) != int(pay["amount_minor"]):
            continue  # уже мутирован
        eligible.append((pay, rows, cash_row, ar_row))
    if not eligible:
        raise EpisodeError("no eligible payment for partial_allocation")
    pay, rows, cash_row, ar_row = eligible[rng.randrange(len(eligible))]

    total = int(pay["amount_minor"])
    delta = min(max(1, total * rng.randrange(10, 45) // 100), total - 1)
    ar_before = int(ar_row["credit_minor"])
    cash_before = int(cash_row["debit_minor"])
    cash_row["debit_minor"] = cash_before - delta
    ar_row["credit_minor"] = ar_before - delta
    # Эффект считаем по НАБЛЮДАЕМОЙ разнице сконвертированных сумм:
    # conv(a) - conv(b) != conv(a - b) из-за независимого округления каждой
    # конвертации (иначе заявленное расхождение расходится на единицу).
    delta_base = (fx.convert(ar_before, pay["currency"], base, policy_day)
                  - fx.convert(int(ar_row["credit_minor"]), pay["currency"], base, policy_day))

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
        "statement": f"платёж {pay['payment_id']} распределён в журнале на {total - delta} "
                     f"при документе на {total}: недостаёт {delta}",
    }
    mutation = {
        "issue_id": issue_id,
        "type": "partial_allocation",
        "causal_change": f"payment {pay['payment_id']} document keeps {total}, "
                         f"both ledger rows reduced by {delta}",
        "before": {"payment_amount_minor": total, "ledger_allocated_minor": total},
        "after": {"payment_amount_minor": total, "ledger_allocated_minor": total - delta},
        "expected_amount_minor": delta,
        "currency": pay["currency"],
        "effect_imbalance": 0,          # обе стороны уменьшены одинаково
        "effect_ar_diff": delta_base,   # кредит уменьшен -> AR в периоде завышен
        "affected_ids": issue["affected_ids"],
    }
    return issue, mutation


def _inject_fx_rate(rng: random.Random, ledger: list[dict], payments: list[dict],
                    exclude: set[str] | None, fx: FxTable, policy_day: date,
                    base: str) -> tuple[dict, dict]:
    """Конвертированная строка кросс-валютного платежа посчитана не по policy-курсу."""
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
        ar_row = next((r for r in rows if r["account"] == AR), None)
        if ar_row is None:
            continue
        # кросс-валютный = документ в чужой валюте, постинг в базовой
        if pay["currency"] == base or ar_row["currency"] != base:
            continue
        if int(ar_row["credit_minor"]) != fx.convert(int(pay["amount_minor"]),
                                                     pay["currency"], base, policy_day):
            continue  # уже мутирован
        eligible.append((pay, rows, ar_row))
    if not eligible:
        raise EpisodeError("no eligible cross-currency payment for fx_rate")
    pay, rows, ar_row = eligible[rng.randrange(len(eligible))]

    correct = int(ar_row["credit_minor"])
    other_days = [d for d in fx.dates if d != policy_day]
    rng.shuffle(other_days)
    wrong = correct
    used_day = None
    for day in other_days:
        candidate = fx.convert(int(pay["amount_minor"]), pay["currency"], base, day)
        if candidate != correct:
            wrong, used_day = candidate, day
            break
    if used_day is None:  # котировки совпали -> фиксируем отклонение явно
        wrong = correct + max(1, correct // 200)
        used_day = other_days[0]
    delta = wrong - correct  # подписанный эффект в базовой валюте
    ar_row["credit_minor"] = wrong

    cash_row = next(r for r in rows if r["account"] == CASH)
    policy_date = policy_day.isoformat()
    src, tgt = pay["currency"], base
    issue_id = "iss_fx_0001"
    issue = {
        "issue_id": issue_id,
        "type": "fx_rate",
        "affected_ids": sorted([pay["payment_id"], cash_row["row_id"], ar_row["row_id"]]),
        "expected_amount_minor": abs(delta),
        "currency": base,
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"payment:{pay['payment_id']}", f"ledger:{ar_row['row_id']}",
             f"fx:{policy_date}:{src}:{tgt}", "policy:fx_rate_date"],
            [f"payment:{pay['payment_id']}", f"ledger:{cash_row['row_id']}",
             f"ledger:{ar_row['row_id']}", "policy:fx_rate_date"],
        ],
        "statement": f"платёж {pay['payment_id']} ({src}) конвертирован по курсу на "
                     f"{used_day.isoformat()} вместо policy-даты {policy_date}: "
                     f"расхождение {abs(delta)} {base}",
    }
    mutation = {
        "issue_id": issue_id,
        "type": "fx_rate",
        "causal_change": f"AR row {ar_row['row_id']} of payment {pay['payment_id']} "
                         f"recomputed with rate of {used_day.isoformat()} instead of "
                         f"policy date {policy_date}",
        "before": {"ar_credit_minor": correct, "rate_date": policy_date},
        "after": {"ar_credit_minor": wrong, "rate_date": used_day.isoformat()},
        "expected_amount_minor": abs(delta),
        "currency": base,
        "effect_imbalance": -delta,   # лишний/недостающий кредит
        "effect_ar_diff": -delta,     # кредит изменил баланс AR в периоде
        "affected_ids": issue["affected_ids"],
    }
    return issue, mutation


# --------------------------------------------------------------------------- #
# Сценарии взаимодействия (спека 02): две ошибки на ОДНОМ кросс-валютном
# платеже у границы периода. Обе ошибки делят одни записи (общие affected_ids),
# поэтому наивный детектор «только даты» или «только суммы» обязан терять вторую.
# --------------------------------------------------------------------------- #
def _cross_candidates(ledger: list[dict], payments: list[dict], period_start: datetime,
                      period_end: datetime, exclude: set[str] | None, fx: FxTable,
                      policy_day: date, base: str) -> list[tuple]:
    by_payment: dict[str, list[dict]] = {}
    for row in ledger:
        if row["payment_id"] and row["payment_id"] != NULL_TOKEN:
            by_payment.setdefault(row["payment_id"], []).append(row)
    out = []
    for pay in payments:
        if pay["payment_id"] in (exclude or set()):
            continue
        rows = by_payment.get(pay["payment_id"], [])
        if len(rows) != 2 or pay["currency"] == base:
            continue
        ar = next((r for r in rows if r["account"] == AR), None)
        cash = next((r for r in rows if r["account"] == CASH), None)
        if ar is None or cash is None or ar["currency"] != base:
            continue
        ts = datetime.fromisoformat(pay["timestamp"])
        if not (period_start < ts <= period_end):
            continue
        correct = fx.convert(int(pay["amount_minor"]), pay["currency"], base, policy_day)
        if int(cash["debit_minor"]) != correct or int(ar["credit_minor"]) != correct:
            continue  # платеж уже мутирован другим типом
        out.append((pay, rows, cash, ar, ts))
    # ближе к границе периода — в приоритете; подстраховка: любой подходящий
    out.sort(key=lambda t: abs((period_end - t[4]).total_seconds()))
    return out[:3]


def _wrong_amount(rng: random.Random, fx: FxTable, amount_minor: int, src: str, base: str,
                  policy_day: date, correct: int) -> tuple[int, date]:
    """Сумма по курсу чужой даты (или явное отклонение, если котировки совпали)."""
    other = [d for d in fx.dates if d != policy_day]
    rng.shuffle(other)
    for day in other:
        candidate = fx.convert(amount_minor, src, base, day)
        if candidate != correct:
            return candidate, day
    return correct + max(1, correct // 200), other[0]


def _cutoff_issue(pay: dict, rows: list[dict], issue_id: str) -> dict:
    ar_row = next(r for r in rows if r["account"] == AR)
    other = next(r for r in rows if r is not ar_row)
    return {
        "issue_id": issue_id,
        "type": "period_cutoff",
        "affected_ids": sorted([pay["payment_id"], ar_row["row_id"], other["row_id"]]),
        "expected_amount_minor": int(pay["amount_minor"]),
        "currency": pay["currency"],
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"payment:{pay['payment_id']}", f"ledger:{ar_row['row_id']}", "policy:period"],
            [f"payment:{pay['payment_id']}", f"ledger:{other['row_id']}",
             f"ledger:{ar_row['row_id']}", "policy:period"],
        ],
        "statement": f"строки платежа {pay['payment_id']} отнесены к периоду после "
                     f"period_end при документе с настоящей датой",
    }


def _inject_scenario_s1(rng: random.Random, ledger: list[dict], payments: list[dict],
                        period_start: datetime, period_end: datetime, exclude: set[str] | None,
                        fx: FxTable, policy_day: date, base: str) -> tuple[list, list, str]:
    """S1 «курс + граница»: fx_rate (деньги) И period_cutoff (время) на одном платеже.

    Поля не пересекаются: fx меняет сумму AR-строки, cutoff — только метки времени.
    """
    cand = _cross_candidates(ledger, payments, period_start, period_end, exclude,
                             fx, policy_day, base)
    if not cand:
        raise EpisodeError("S1: no eligible cross-currency payment near period boundary")
    pay, rows, cash, ar, ts = cand[rng.randrange(len(cand))]
    pid = pay["payment_id"]

    # (1) fx_rate: конвертированная AR-строка посчитана по курсу чужой даты
    correct = int(ar["credit_minor"])
    wrong, used_day = _wrong_amount(rng, fx, int(pay["amount_minor"]), pay["currency"],
                                    base, policy_day, correct)
    delta = wrong - correct
    ar["credit_minor"] = wrong
    fx_issue = {
        "issue_id": "iss_s1_fx_0001",
        "type": "fx_rate",
        "affected_ids": sorted([pid, cash["row_id"], ar["row_id"]]),
        "expected_amount_minor": abs(delta),
        "currency": base,
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"payment:{pid}", f"ledger:{ar['row_id']}",
             f"fx:{policy_day.isoformat()}:{pay['currency']}:{base}", "policy:fx_rate_date"],
            [f"payment:{pid}", f"ledger:{cash['row_id']}", f"ledger:{ar['row_id']}",
             "policy:fx_rate_date"],
        ],
        "statement": f"S1: платёж {pid} ({pay['currency']}) у границы периода конвертирован "
                     f"по курсу на {used_day.isoformat()} вместо {policy_day.isoformat()}: "
                     f"расхождение {abs(delta)} {base}",
    }
    fx_mut = {
        "issue_id": fx_issue["issue_id"], "type": "fx_rate",
        "causal_change": f"scenario S1: AR row {ar['row_id']} of payment {pid} recomputed "
                         f"with rate of {used_day.isoformat()} instead of policy date "
                         f"{policy_day.isoformat()}",
        "before": {"ar_credit_minor": correct, "rate_date": policy_day.isoformat()},
        "after": {"ar_credit_minor": wrong, "rate_date": used_day.isoformat()},
        "expected_amount_minor": abs(delta), "currency": base,
        "effect_imbalance": -delta, "effect_ar_diff": -delta,
        "affected_ids": fx_issue["affected_ids"],
    }

    # (2) period_cutoff: строки переезжают за period_end, документ остаётся внутри
    new_ts = period_end + timedelta(days=1, hours=12)
    for row in rows:
        row["timestamp"] = _iso(new_ts)
    cutoff_issue = _cutoff_issue(pay, rows, "iss_s1_cutoff_0001")
    ar_current = int(ar["credit_minor"])  # строка уже в базовой валюте
    cutoff_mut = {
        "issue_id": cutoff_issue["issue_id"], "type": "period_cutoff",
        "causal_change": f"scenario S1: rows of payment {pid} re-dated from {ts.isoformat()} "
                         f"to {_iso(new_ts)}; document unchanged",
        "before": {"rows_timestamp": ts.isoformat()},
        "after": {"rows_timestamp": _iso(new_ts)},
        "expected_amount_minor": int(pay["amount_minor"]), "currency": pay["currency"],
        "effect_imbalance": 0, "effect_ar_diff": ar_current,
        "affected_ids": cutoff_issue["affected_ids"],
    }
    return [fx_issue, cutoff_issue], [fx_mut, cutoff_mut], pid


def _inject_scenario_s2(rng: random.Random, ledger: list[dict], payments: list[dict],
                        period_start: datetime, period_end: datetime, exclude: set[str] | None,
                        fx: FxTable, policy_day: date, base: str) -> tuple[list, list, str]:
    """S2 «частичный платёж в другой валюте на границе периода» (пример спеки):
    partial_allocation (деньги) И period_cutoff (время) на одном кросс-валютном платеже.

    Наивный детектор, сравнивающий документ и журнал БЕЗ конвертации, получает
    ложную дельту; детектор, остановившийся на cutoff, теряет частичное распределение.
    """
    cand = _cross_candidates(ledger, payments, period_start, period_end, exclude,
                             fx, policy_day, base)
    if not cand:
        raise EpisodeError("S2: no eligible cross-currency payment near period boundary")
    pay, rows, cash, ar, ts = cand[rng.randrange(len(cand))]
    pid = pay["payment_id"]

    # (1) partial_allocation: документ полон, обе строки журнала уменьшены
    cash_base = int(cash["debit_minor"])
    delta = min(max(1, cash_base * rng.randrange(10, 45) // 100), cash_base - 1)
    cash["debit_minor"] = cash_base - delta
    ar["credit_minor"] = int(ar["credit_minor"]) - delta
    partial_issue = {
        "issue_id": "iss_s2_partial_0001",
        "type": "partial_allocation",
        "affected_ids": sorted([pid, cash["row_id"], ar["row_id"]]),
        "expected_amount_minor": delta,
        "currency": base,
        "amount_tolerance_minor": 0,
        "evidence_sets": [
            [f"payment:{pid}", f"ledger:{ar['row_id']}", f"invoice:{ar['invoice_id']}"],
            [f"payment:{pid}", f"ledger:{cash['row_id']}", f"ledger:{ar['row_id']}"],
        ],
        "statement": f"S2: платёж {pid} ({pay['currency']}) распределён в журнале на "
                     f"{cash_base - delta} {base} при документе, эквивалент {cash_base} {base}: "
                     f"недостаёт {delta}",
    }
    partial_mut = {
        "issue_id": partial_issue["issue_id"], "type": "partial_allocation",
        "causal_change": f"scenario S2: both ledger rows of payment {pid} reduced by {delta} "
                         f"{base} while the document keeps its full amount",
        "before": {"ledger_allocated_minor": cash_base},
        "after": {"ledger_allocated_minor": cash_base - delta},
        "expected_amount_minor": delta, "currency": base,
        "effect_imbalance": 0, "effect_ar_diff": delta,
        "affected_ids": partial_issue["affected_ids"],
    }

    # (2) period_cutoff: те же строки переезжают за границу периода
    new_ts = period_end + timedelta(days=1, hours=12)
    for row in rows:
        row["timestamp"] = _iso(new_ts)
    cutoff_issue = _cutoff_issue(pay, rows, "iss_s2_cutoff_0001")
    ar_current = int(ar["credit_minor"])
    cutoff_mut = {
        "issue_id": cutoff_issue["issue_id"], "type": "period_cutoff",
        "causal_change": f"scenario S2: rows of payment {pid} re-dated from {ts.isoformat()} "
                         f"to {_iso(new_ts)} after partial allocation; document unchanged",
        "before": {"rows_timestamp": ts.isoformat()},
        "after": {"rows_timestamp": _iso(new_ts)},
        "expected_amount_minor": int(pay["amount_minor"]), "currency": pay["currency"],
        "effect_imbalance": 0, "effect_ar_diff": ar_current,
        "affected_ids": cutoff_issue["affected_ids"],
    }
    return [partial_issue, cutoff_issue], [partial_mut, cutoff_mut], pid


# --------------------------------------------------------------------------- #
# loading / saving
# --------------------------------------------------------------------------- #
def load_episode(episode_dir: str | Path) -> dict[str, dict[str, str]]:
    """Читает эпизод из корня каталога либо из подкаталогов open/ и closed/."""
    root = Path(episode_dir)
    open_root = root / "open" if (root / "open").is_dir() else root
    closed_root = root / "closed" if (root / "closed").is_dir() else root
    open_names = ["manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                  "fx.csv", "policy.json"]
    closed_names = ["truth.json", "provenance.json"]
    missing = [n for n in open_names if not (open_root / n).exists()]
    if missing:
        raise EpisodeError(f"episode directory {root} is missing: {missing}")
    return {
        "open": {n: (open_root / n).read_text(encoding="utf-8") for n in open_names},
        "closed": {n: (closed_root / n).read_text(encoding="utf-8") for n in closed_names
                   if (closed_root / n).exists()},
    }


def save_episode(files: dict[str, dict[str, str]], out_dir: str | Path) -> Path:
    root = Path(out_dir)
    (root / "open").mkdir(parents=True, exist_ok=True)
    for name, text in files["open"].items():
        (root / "open" / name).write_text(text, encoding="utf-8")
    if files.get("closed"):
        (root / "closed").mkdir(parents=True, exist_ok=True)
        for name, text in files["closed"].items():
            (root / "closed" / name).write_text(text, encoding="utf-8")
    return root
