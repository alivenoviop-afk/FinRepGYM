"""Безвредные преобразования эпизодов: перестановка порядка и смена ID.

Спека (раздел 06): закрытые проверки включают «новые сочетания параметров и
безвредные преобразования»; **20 парных проверок меняют порядок и ID**, а ответ
оракула и награда обязаны сохраниться **с учётом новых ID**.

Что меняет преобразование (ровно ничего больше):
  * порядок строк в ``ledger.csv``, ``invoices.jsonl``, ``payments.jsonl``;
  * идентификаторы: row_id, doc_id, invoice_id, payment_id, entity_id, episode_id.

Что НЕ меняет: суммы, даты, валюты, курсы, структуру и состав файлов —
значит, все равенства и заявленные расхождения сохраняются как есть.

API:
  ``permute_episode(files, seed) -> (files, id_map)``
  ``remap_submission(submission, id_map) -> submission``  (находитки + episode_id)
"""

from __future__ import annotations

import csv
import io
import json
import random
from typing import Any

ROW_PREFIX = ("row", "rec", "entry", "ln")
INVOICE_PREFIX = ("INV", "BILL", "FACT", "INVC")
PAY_PREFIX = ("PAY", "TRF", "WIR", "PMT")
ENTITY_PREFIX = ("ENT", "CPTY", "PARTNER", "AGT")
INVOICE_DOC_PREFIX = ("inv_doc", "bill_doc", "invoice_doc")
PAYMENT_DOC_PREFIX = ("pay_doc", "trf_doc", "pmt_doc")

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor",
                  "currency", "entity_id", "invoice_id", "payment_id"]


def _new_id(prefixes: tuple[str, ...], index: int, rng: random.Random, width: int = 5) -> str:
    return f"{rng.choice(prefixes)}_{rng.randrange(10 ** width):0{width}d}_{index:03d}"


def permute_episode(files: dict[str, dict[str, str]],
                    seed: int) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Возвращает (файлы эпизода после преобразования, карта переименования)."""
    rng = random.Random(seed)
    open_files = dict(files["open"])
    closed_files = dict(files.get("closed") or {})

    # ---- читаем ---------------------------------------------------------- #
    manifest = json.loads(open_files["manifest.json"])
    ledger = list(csv.DictReader(io.StringIO(open_files["ledger.csv"])))
    invoices = [json.loads(l) for l in open_files["invoices.jsonl"].splitlines() if l]
    payments = [json.loads(l) for l in open_files["payments.jsonl"].splitlines() if l]

    # ---- карты переименования -------------------------------------------- #
    row_ids = [r["row_id"] for r in ledger]
    invoice_ids = [i["invoice_id"] for i in invoices]
    payment_ids = [p["payment_id"] for p in payments]
    entity_ids = sorted({r["entity_id"] for r in ledger} | {i["entity_id"] for i in invoices})

    row_map = dict(zip(row_ids, _shuffled_ids(row_ids, ROW_PREFIX, rng)))
    invoice_map = dict(zip(invoice_ids, _shuffled_ids(invoice_ids, INVOICE_PREFIX, rng)))
    payment_map = dict(zip(payment_ids, _shuffled_ids(payment_ids, PAY_PREFIX, rng)))
    entity_map = dict(zip(entity_ids, _shuffled_ids(entity_ids, ENTITY_PREFIX, rng, width=3)))
    invoice_doc_map = {d["doc_id"]: _new_id(INVOICE_DOC_PREFIX, n, rng, 4)
                       for n, d in enumerate(invoices)}
    payment_doc_map = {d["doc_id"]: _new_id(PAYMENT_DOC_PREFIX, n, rng, 4)
                       for n, d in enumerate(payments)}
    old_episode = manifest["episode_id"]
    new_episode = f"ep_{rng.randrange(16 ** 8):08x}"
    while new_episode == old_episode:
        new_episode = f"ep_{rng.randrange(16 ** 8):08x}"

    # ---- применяем порядок + переименование ------------------------------ #
    rng.shuffle(ledger)
    rng.shuffle(invoices)
    rng.shuffle(payments)

    for r in ledger:
        r["row_id"] = row_map[r["row_id"]]
        r["entity_id"] = entity_map[r["entity_id"]]
        r["invoice_id"] = invoice_map.get(r["invoice_id"], r["invoice_id"])
        r["payment_id"] = payment_map.get(r["payment_id"], r["payment_id"])
    for i in invoices:
        i["doc_id"] = invoice_doc_map[i["doc_id"]]
        i["invoice_id"] = invoice_map[i["invoice_id"]]
        i["entity_id"] = entity_map[i["entity_id"]]
    for p in payments:
        p["doc_id"] = payment_doc_map[p["doc_id"]]
        p["payment_id"] = payment_map[p["payment_id"]]
        p["invoice_refs"] = [invoice_map.get(x, x) for x in p["invoice_refs"]]

    manifest["episode_id"] = new_episode

    # ---- пишем открытые файлы -------------------------------------------- #
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for r in ledger:
        writer.writerow({c: r[c] for c in LEDGER_COLUMNS})
    open_files["ledger.csv"] = buf.getvalue()
    open_files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    open_files["invoices.jsonl"] = "\n".join(
        json.dumps(i, ensure_ascii=False) for i in invoices) + "\n"
    open_files["payments.jsonl"] = "\n".join(
        json.dumps(p, ensure_ascii=False) for p in payments) + "\n"

    # ---- закрытая зона: те же id, что в открытых файлах ------------------- #
    id_map = {
        "episode_id": [old_episode, new_episode],
        "rows": row_map,
        "invoices": invoice_map,
        "payments": payment_map,
        "entities": entity_map,
        "invoice_docs": invoice_doc_map,
        "payment_docs": payment_doc_map,
    }

    if "truth.json" in closed_files:
        truth = json.loads(closed_files["truth.json"])
        truth["episode_id"] = new_episode
        for issue in truth["issues"]:
            issue["affected_ids"] = sorted(_map_id(x, id_map) for x in issue["affected_ids"])
            issue["evidence_sets"] = [
                [_map_evidence(ev, id_map) for ev in s] for s in issue["evidence_sets"]
            ]
        closed_files["truth.json"] = json.dumps(truth, ensure_ascii=False, indent=2) + "\n"

    if "provenance.json" in closed_files:
        prov = json.loads(closed_files["provenance.json"])
        prov["episode_id"] = new_episode
        for mutation in prov.get("mutations", []):
            mutation["affected_ids"] = sorted(_map_id(x, id_map)
                                              for x in mutation.get("affected_ids", []))
        closed_files["provenance.json"] = json.dumps(
            prov, ensure_ascii=False, indent=2, default=str) + "\n"

    return {"open": open_files, "closed": closed_files}, id_map


def _shuffled_ids(old_ids: list[str], prefixes: tuple[str, ...], rng: random.Random,
                  width: int = 5) -> list[str]:
    """Новые ID в префиксах, перемешанные относительно исходного порядка."""
    new_ids = [_new_id(prefixes, i, rng, width) for i in range(len(old_ids))]
    rng.shuffle(new_ids)
    # гарантируем, что карта не совпала с исходными id (не identity)
    for old, new in zip(old_ids, new_ids):
        if old == new:
            new_ids[rng.randrange(len(new_ids))] = new + "x"
            break
    return new_ids


def _map_id(value: str, id_map: dict[str, Any]) -> str:
    for domain in ("rows", "invoices", "payments", "entities",
                   "invoice_docs", "payment_docs"):
        if value in id_map[domain]:
            return id_map[domain][value]
    return value


def _map_evidence(evidence: str, id_map: dict[str, Any]) -> str:
    prefix, sep, value = evidence.partition(":")
    if not sep:
        return evidence
    if prefix in ("ledger", "invoice", "payment"):
        return f"{prefix}:{_map_id(value, id_map)}"
    return evidence  # fx:... и policy:... не содержат эпизодных ID


def remap_submission(submission: dict[str, Any], id_map: dict[str, Any]) -> dict[str, Any]:
    """Переносит отправку агента (находки + episode_id) на новые ID."""
    findings = []
    for f in submission.get("findings", []):
        findings.append({
            **f,
            "affected_ids": [_map_id(x, id_map) for x in f.get("affected_ids", [])],
            "evidence_ids": [_map_evidence(x, id_map) for x in f.get("evidence_ids", [])],
        })
    return {"action": "submit", "episode_id": id_map["episode_id"][1], "findings": findings}
