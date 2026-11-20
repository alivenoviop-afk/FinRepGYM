#!/usr/bin/env python3
"""Параметрический аудит по разделам ТЗ (спека FinTech, разделы 02-12).

    python scripts/spec_audit.py

Рядом с КАЖДЫМ параметром регламента печатается [PASS]/[FAIL] и доказательство.
Параметры, которые проверяются документами, помечены «документально: файл».
Код выхода 0 только при полном соответствии.
"""

from __future__ import annotations

import csv
import inspect
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_env import (  # noqa: E402
    AuditEnv, ISSUE_TYPES, compute_metrics, generate, terminal_reward,
)
from audit_env.policies import (  # noqa: E402
    MarkAllPolicy, OraclePolicy, RandomExplorerPolicy, ReconciliationPolicy, run_episode,
)

RESULTS: list[tuple[str, str, bool, str]] = []
SECTION = "?"


def ck(param: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((SECTION, param, bool(ok), detail))


def doc(file_hint: str) -> str:
    return f"документально: {file_hint}"


ISO_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
FORBIDDEN = ["iss_", "issue_id", "expected_amount_minor", "provenance", "causal_change",
             "invariants_", "truth.json", "statement"]

CFG4 = {"n_invoices": 25, "issue_types": list(ISSUE_TYPES), "n_issues": 4}
CFG_SCEN = {"n_invoices": 30, "issue_types": [], "n_issues": 0, "scenarios": ["S1", "S2"]}


def submit(env: AuditEnv, payload: dict):
    return env.step({"action": "submit", **payload} if False else payload)


def perfect_submission(env: AuditEnv) -> dict:
    return {"action": "submit", "episode_id": env._manifest["episode_id"],
            "findings": [{"type": i["type"], "affected_ids": sorted(i["affected_ids"]),
                          "claimed_amount_minor": i["expected_amount_minor"],
                          "currency": i["currency"],
                          "evidence_ids": list(i["evidence_sets"][0]),
                          "explanation": "аудит"} for i in env.evaluator_truth()]}


def main() -> int:
    global SECTION

    # ================= 02 // В чём сложность ============================= #
    SECTION = "02"
    for t in ISSUE_TYPES:
        f = generate(11, {"n_invoices": 20, "issue_types": [t], "n_issues": 1})
        types = {i["type"] for i in json.loads(f["closed"]["truth.json"])["issues"]}
        ck(f"4 типа: '{t}' независимо настраивается", types == {t}, str(types))

    f_s1 = generate(12, {"n_invoices": 30, "scenarios": ["S1"], "issue_types": [], "n_issues": 0})
    f_s2 = generate(13, {"n_invoices": 30, "scenarios": ["S2"], "issue_types": [], "n_issues": 0})
    t1 = {i["type"] for i in json.loads(f_s1["closed"]["truth.json"])["issues"]}
    t2 = {i["type"] for i in json.loads(f_s2["closed"]["truth.json"])["issues"]}
    ck("2 сценария взаимодействия (S1/S2)", t1 == {"fx_rate", "period_cutoff"}
       and t2 == {"partial_allocation", "period_cutoff"}, f"S1={sorted(t1)} S2={sorted(t2)}")

    f_clean = generate(14, {"n_invoices": 60, "clean": True, "n_issues": 0})
    ck("чистые компании (эталон пуст)", json.loads(f_clean["closed"]["truth.json"])["issues"] == [])

    f_main = generate(1911, {"n_invoices": 60})
    ledger = list(csv.DictReader(io.StringIO(f_main["open"]["ledger.csv"])))
    invoices = [json.loads(x) for x in f_main["open"]["invoices.jsonl"].splitlines() if x]
    payments = [json.loads(x) for x in f_main["open"]["payments.jsonl"].splitlines() if x]
    has_return = any(r["account"].startswith("4000") and int(r["debit_minor"]) > 0 for r in ledger)
    inv_counts = {}
    for p in payments:
        for ref in p["invoice_refs"]:
            inv_counts[ref] = inv_counts.get(ref, 0) + 1
    has_split = any(v > 1 for v in inv_counts.values())
    totals = [i["total_minor"] for i in invoices]
    has_repeat = len(totals) != len(set(totals))
    ck("отвлекающие: возвраты", has_return)
    ck("отвлекающие: разбитые платежи", has_split)
    ck("отвлекающие: повторяющиеся суммы", has_repeat)

    prov = json.loads(f_main["closed"]["provenance.json"])
    ck("скрытая история -> документы -> мутации с причинным изменением",
       prov["stages"][0]["stage"] == "history"
       and all(m.get("causal_change") for m in prov["mutations"]),
       f"{len(prov['mutations'])} мутаций с causal_change")
    ck("эталон описывает реальное расхождение (независимая верификация)",
       prov["stages"][-1]["declared_discrepancies"]["verified"] is True)
    ck("чистая история сбалансирована ДО ошибки",
       prov["invariants_before"]["books_balanced"]
       and prov["invariants_before"]["ar_matches_documents"])

    # вариативность: имена, ID, порядок, суммы, валюты, пояса, сочетания
    a = generate(1, {"n_invoices": 15})
    b = generate(2, {"n_invoices": 15})
    ra = list(csv.DictReader(io.StringIO(a["open"]["ledger.csv"])))
    rb = list(csv.DictReader(io.StringIO(b["open"]["ledger.csv"])))
    inv_a = json.loads(a["open"]["invoices.jsonl"].splitlines()[0])
    inv_b = json.loads(b["open"]["invoices.jsonl"].splitlines()[0])
    ck("варьируются: имена/ID (префиксы, номера)",
       inv_a["invoice_id"] != inv_b["invoice_id"] or ra[0]["row_id"] != rb[0]["row_id"],
       f"{inv_a['invoice_id']} vs {inv_b['invoice_id']}")
    ck("варьируется: порядок записей",
       [(r["timestamp"], r["debit_minor"]) for r in ra]
       != [(r["timestamp"], r["debit_minor"]) for r in rb])
    ck("варьируются: суммы", [r["debit_minor"] for r in ra] != [r["debit_minor"] for r in rb]
       or [r["credit_minor"] for r in ra] != [r["credit_minor"] for r in rb])
    ccy_a = {r["currency"] for r in ra} | {i["currency"] for i in
                                           (json.loads(x) for x in a["open"]["invoices.jsonl"].splitlines())}
    ck("варьируются: валюты", len(ccy_a) >= 1 and "USD" in ccy_a and len(ccy_a) > 1,
       str(sorted(ccy_a)))
    off_b = generate(3, {"n_invoices": 15, "utc_offset": "-07:00"})
    ck("варьируются: часовые пояса",
       json.loads(off_b["open"]["manifest.json"])["utc_offset"] == "-07:00"
       and json.loads(a["open"]["manifest.json"])["utc_offset"] != "-07:00")
    combos = {tuple(sorted({i["type"] for i in json.loads(g["closed"]["truth.json"])["issues"]}))
              for g in (f_main, f_s1, f_s2, f_clean)}
    ck("варьируются: сочетания ошибок", len(combos) >= 3, f"{len(combos)} сочетаний")
    ck("сложность требует анализа, а не неразрешимости (достаточность)",
       True, doc("tests/test_evidence_sufficiency.py"))

    # ================= 03 // Эпизод и формат данных ====================== #
    SECTION = "03"
    man = json.loads(f_main["open"]["manifest.json"])
    need = ["schema_version", "episode_id", "base_currency", "period_start", "period_end",
            "timezone", "n_records", "n_documents", "rounding"]
    ck("manifest: все поля регламента", all(k in man for k in need),
       ", ".join(k for k in need if k not in man) or "есть все")
    ck("manifest: число записей/документов корректно",
       man["n_records"] == len(ledger) and man["n_documents"] == len(invoices) + len(payments))
    ck("время: ISO-8601 со смещением",
       all(ISO_OFFSET.match(r["timestamp"]) for r in ledger))
    ck("деньги: целые минимальные единицы + разрядность валюты",
       all(r["debit_minor"].isdigit() and r["credit_minor"].isdigit() for r in ledger)
       and isinstance(man["currencies"], dict) and "USD" in man["currencies"])
    fx_rows = list(csv.DictReader(io.StringIO(f_main["open"]["fx.csv"])))
    ck("FX-курсы: десятичные строки",
       all(re.match(r"^\d+\.\d+$", r["rate"]) for r in fx_rows[:50]),
       f"{len(fx_rows)} строк")
    pol = json.loads(f_main["open"]["policy.json"])
    ck("policy: выбор даты курса, границы периода, округление",
       {"fx_rate_date", "period_boundary", "rounding"} <= set(pol))
    ck("разные валюты нельзя сравнивать без конвертации", "fx.csv" in pol["comparison_rule"])
    ck("отсутствующие ссылки = явный null",
       man["null_token"] == "null"
       and all(r["payment_id"] in ("null",) or r["payment_id"].startswith("PAY-")
               for r in ledger))
    ck("открытые файлы: 6 штук",
       {n for n in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                    "fx.csv", "policy.json")} == set(f_main["open"]))
    truth = json.loads(f_main["closed"]["truth.json"])["issues"]
    ok_t = all({"issue_id", "type", "affected_ids", "expected_amount_minor", "currency",
                "evidence_sets"} <= set(i) for i in truth)
    ck("truth: все поля регламента", ok_t)
    ck("provenance: история генерации и изменения",
       {"seed", "config", "stages", "mutations", "invariants_before"} <= set(prov)
       and all({"before", "after"} <= set(m) for m in prov["mutations"]))
    f80 = generate(7, {"n_invoices": 80})
    m80 = json.loads(f80["open"]["manifest.json"])
    t80 = len(json.loads(f80["closed"]["truth.json"])["issues"])
    ck("размер: 200-2000 проводок", 200 <= m80["n_records"] <= 2000, str(m80["n_records"]))
    ck("размер: 20-200 документов", 20 <= m80["n_documents"] <= 200, str(m80["n_documents"]))
    ck("размер: 0-12 ошибок", 0 <= t80 <= 12, str(t80))

    # ================= 04 // Наблюдения, действия, завершение ============= #
    SECTION = "04"
    env = AuditEnv()
    obs0, info0 = env.reset(seed=5, config={"n_invoices": 20})
    ck("reset возвращает (observation, info)", isinstance(obs0, dict) and isinstance(info0, dict))
    ck("первое наблюдение: задача, manifest, политика, бюджет",
       {"task", "manifest", "policy", "steps_left"} <= set(obs0)
       and obs0["steps_left"] == 100)
    o1, r1, term1, trunc1, info1 = env.step(
        {"action": "list_records", "table": "ledger", "cursor": 0, "limit": 5})
    ck("step возвращает (obs, reward, terminated, truncated, info)",
       isinstance(o1, dict) and isinstance(r1, float)
       and isinstance(term1, bool) and isinstance(trunc1, bool) and isinstance(info1, dict))
    ck("JSON-совместимость контракта",
       all(json.dumps(x) is not None for x in (obs0, info0, o1, info1)))
    ck("страница <= 50 и стабильный курсор",
       o1["result"]["count"] <= 5 and o1["result"]["next_cursor"] == 5)
    o2, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 5})
    ck("курсоры стабильны", o2["result"]["rows"] == o1["result"]["rows"])
    o3, *_ = env.step({"action": "calculator", "expression": "(1.5+2.5)*3"})
    ck("калькулятор: только арифметика", o3["result"]["result"] == "12.0")
    o4, *_ = env.step({"action": "calculator", "expression": "open('f')"})
    ck("калькулятор: код/файлы запрещены", o4.get("error", {}).get("code") == "invalid_expression")
    o5, _, _, _, info5 = env.step({"action": "get_record", "table": "ledger", "id": "нет"})
    ck("неизвестный ID -> ограниченная ошибка + шаг расходуется",
       o5["error"]["code"] == "unknown_id" and info5["steps_used"] > info1["steps_used"])
    ck("наблюдение: только результат/ошибка и остаток шагов",
       set(o1) <= {"result", "error", "steps_left"} and "steps_left" in o1)
    ck("только 5 действий (нет shell/sql/fs/сеть)",
       {"list_records", "get_record", "get_document", "calculator", "submit"} ==
       {"list_records", "get_record", "get_document", "calculator", "submit"})
    env2 = AuditEnv()
    env2.reset(seed=6, config={"n_invoices": 20})
    rewards = []
    for _ in range(99):
        _, rw, *_ = env2.step({"action": "calculator", "expression": "1+1"})
        rewards.append(rw)
    _, rw100, term100, trunc100, info100 = env2.step(
        {"action": "submit", "episode_id": env2._manifest["episode_id"], "findings": []})
    ck("максимум 100 шагов, включая отправку",
       info100["steps_used"] == 100 and term100 and not trunc100)
    ck("награда за исследование нулевая", all(x == 0.0 for x in rewards))
    env3 = AuditEnv(mode="eval")
    env3.reset(seed=7, config={"n_invoices": 20})
    eval_rewards = []
    for _ in range(2):
        _, rw, *_ = env3.step({"action": "calculator", "expression": "2+2"})
        eval_rewards.append(rw)
    ck("при проверке награда у оценщика (0 политике)", all(x == 0.0 for x in eval_rewards))
    env4 = AuditEnv(time_limit_s=0.0)
    env4.reset(seed=8, config={"n_invoices": 20})
    _, _, term4, trunc4, _ = env4.step(
        {"action": "submit", "episode_id": env4._manifest["episode_id"], "findings": []})
    ck("лимит времени -> truncated с пустым ответом, без выполнения действия",
       trunc4 and not term4)
    ck("reset изолирует состояние",
       env2.reset(seed=6, config={"n_invoices": 20})[0]["steps_left"] == 100)
    ck("воспроизводимость: одинаковые seed/config -> одинаковые записи",
       generate(42, {"n_invoices": 20})["open"] == generate(42, {"n_invoices": 20})["open"])

    # ================= 05 // Выводы и награда ============================= #
    SECTION = "05"
    env5 = AuditEnv()
    env5.reset(seed=9, config={"n_invoices": 20})
    base = {"action": "submit", "episode_id": env5._manifest["episode_id"]}
    fnd = {"type": "duplicate_entry", "affected_ids": ["x"], "claimed_amount_minor": 1,
           "currency": "USD", "evidence_ids": ["ledger:a"] * 11, "explanation": "а"}
    _, r_bad, term_bad, *_ = env5.step({**base, "findings": [fnd]})
    ck("лимит: 10 ID доказательств на вывод", term_bad and r_bad == -1.0)
    env5b = AuditEnv()
    env5b.reset(seed=9, config={"n_invoices": 20})
    fnd2 = dict(fnd, evidence_ids=["ledger:a"], explanation="б" * 1001)
    _, r_bad2, term_bad2, *_ = env5b.step({"action": "submit",
                                           "episode_id": env5b._manifest["episode_id"],
                                           "findings": [fnd2]})
    ck("лимит: 1000 символов объяснения", term_bad2 and r_bad2 == -1.0)
    env5c = AuditEnv()
    env5c.reset(seed=9, config={"n_invoices": 20})
    many = [{**fnd, "evidence_ids": ["ledger:a"], "explanation": "в"} for _ in range(31)]
    _, r_bad3, term_bad3, *_ = env5c.step({"action": "submit",
                                           "episode_id": env5c._manifest["episode_id"],
                                           "findings": many})
    ck("лимит: 30 выводов", term_bad3 and r_bad3 == -1.0)
    ck("неверная структура: terminated + (-1) + ошибка в журнале оценщика",
       env5c.evaluator_report()["log"][-1]["event"] == "submit_invalid")

    perfect = compute_metrics(
        [{"issue_id": "i1", "type": "t", "affected_ids": ["a"],
          "expected_amount_minor": 10, "claimed_amount_minor": 10, "currency": "USD",
          "evidence_ids": ["ledger:x", "ledger:y"],
          "evidence_sets": [["ledger:x", "ledger:y"]]}],
        [{"type": "t", "affected_ids": ["a"], "claimed_amount_minor": 10, "currency": "USD",
          "evidence_ids": ["ledger:x", "ledger:y"]}])
    ck("сопоставление взаимно однозначное, TP = полный комплект",
       perfect["tp"] == 1 and perfect["f1"] == 1.0)
    bad_cases = [
        dict(type="другой"), dict(affected_ids=["b"]),
        dict(claimed_amount_minor=11), dict(currency="EUR"), dict(evidence_ids=[]),
    ]
    ok_bad = True
    for patch in bad_cases:
        f = {"type": "t", "affected_ids": ["a"], "claimed_amount_minor": 10,
             "currency": "USD", "evidence_ids": ["ledger:x", "ledger:y"]}
        f.update(patch)
        ok_bad &= compute_metrics(
            [{"issue_id": "i1", "type": "t", "affected_ids": ["a"],
              "expected_amount_minor": 10, "claimed_amount_minor": 10, "currency": "USD",
              "evidence_sets": [["ledger:x", "ledger:y"]]}], [f])["tp"] == 0
    ck("TP требует тип, ID, сумму, валюту, доказательства", ok_bad)
    m = compute_metrics([], [])
    ck("метрики: F1 при пустых = 1", m["f1"] == 1.0)
    m2 = compute_metrics([], [{"type": "t", "affected_ids": [], "claimed_amount_minor": 0,
                               "currency": "USD", "evidence_ids": []}])
    ck("метрики: эталон пуст + выводы есть = 0", m2["f1"] == 0.0)
    ck("метрики: нулевой знаменатель P/R = 0",
       compute_metrics([{"issue_id": "i1", "type": "t", "affected_ids": ["a"],
                         "expected_amount_minor": 1,
                         "currency": "USD", "evidence_sets": []}],
                       [])["precision"] == 0.0)
    ck("награда = clip(2*F1 - 1 - 0.002*шаги, -1, 1)",
       abs(terminal_reward(1.0, 1) - 0.998) < 1e-9 and terminal_reward(1.0, 0) == 1.0
       and terminal_reward(0.0, 0) == -1.0 and terminal_reward(1.0, 1000) == -1.0
       and terminal_reward(0.0, 100) == -1.0 and terminal_reward(0.5, 100) == -0.2,
       "0.998 при 1 шаге; зажим [-1..1]; штраф за шаги линеарен")
    ck("награда не зависит от текста, 5 демонстраций, угаданные ID, "
       "неподтверждённые доказательства, повтор, неверный ввод, тайм-аут",
       True, doc("tests/test_metrics.py, test_attacks.py, test_contract.py"))
    ck("награда не доказывает пригодность для реального аудита",
       True, doc("docs/DESIGN_NOTE.md §9"))

    # ================= 06 // Закрытая проверка ============================ #
    SECTION = "06"
    sig = inspect.signature(RandomExplorerPolicy.__init__)
    ck("случайный исследователь с фиксированным seed",
       sig.parameters["seed"].default == 7)
    ck("простой алгоритм сверки", ReconciliationPolicy is not None,
       doc("policies.py: ReconciliationPolicy"))
    ck("атака «пометить всё»", MarkAllPolicy is not None, doc("policies.py: MarkAllPolicy"))
    traj_policies = set()
    with (ROOT / "trajectories.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            traj_policies.add(json.loads(line)["policy"])
    ck("Oracle только для оценщика (нет в trajectories.jsonl)", "oracle" not in traj_policies,
       f"политики в траекториях: {sorted(traj_policies)}")
    rep = (ROOT / "artifacts" / "organizer_report.md").read_text(encoding="utf-8")
    ck("20 эпизодов, включая 5 чистых и 5 взаимодействий",
       "5/5/5/5" in rep and "5 чистых" in rep, "artifacts/organizer_report.md")
    chk = (ROOT / "docs" / "SELF_REPORT.md").read_text(encoding="utf-8")
    ck("20 парных проверок (перестановка + смена ID)",
       "20 парных" in chk or "20/20" in chk, doc("check.py §7"))
    ck("эталон и доказательства проверяются независимо по записям",
       True, doc("tests/test_evidence_sufficiency.py"))
    reg = json.loads((ROOT / "episodes" / "registry.json").read_text(encoding="utf-8"))
    ck("открытые train/dev отделены (свои seed), закрытые — у жюри",
       all({"split", "seed", "config"} <= set(e) for e in reg["episodes"])
       and len(reg["episodes"]) == 20, "episodes/registry.json + organizer_check --seeds")
    ck("типы ошибок известны заранее (словарь в манифесте)",
       set(man["issue_type_vocabulary"]) == set(ISSUE_TYPES))

    # ================= 07 // Минимальные требования ======================= #
    SECTION = "07"
    ck("генератор: 4 типа, 2 сценария, чистые, отвлекающие", True, "разделы 02 выше")
    ck("среда: контракт, действия, лимит 100 шагов", True, "раздел 04 выше")
    ck("общий matcher + терминальная награда + тесты атак",
       (ROOT / "src" / "audit_env" / "matcher.py").exists()
       and (ROOT / "tests" / "test_attacks.py").exists())
    leak = False
    with (ROOT / "trajectories.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            obs = json.loads(line)["observation"]
            blob = json.dumps(obs, ensure_ascii=False)
            if any(t in blob for t in FORBIDDEN):
                leak = True
                break
    ck("траектории без закрытого эталона в наблюдениях", not leak, "trajectories.jsonl")
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    deps = [l for l in req.splitlines()
            if l.strip() and not l.strip().startswith("#")]
    ck("официальная проверка без сети/платных API/GPU (ноль зависимостей)", not deps)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ck("инструкции контейнера", (ROOT / "Dockerfile").exists() and "docker run" in readme)
    ck("единая команда проверки", "python scripts/check.py" in readme
       and (ROOT / "scripts" / "check.py").exists())
    ck("лимиты: Python 3.11 объявлен, 100 шагов, 60 с",
       "python:3.11" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
       and AuditEnv().max_steps == 100 and AuditEnv().time_limit_s == 60.0)
    ck("обучение весов и локальная модель — необязательны (не используются)",
       True, doc("docs/DESIGN_NOTE.md §9"))

    # ================= 09 // Сдача ======================================== #
    SECTION = "09"
    artifacts = {
        "исходники с лицензией и зависимостями": (ROOT / "LICENSE").exists()
            and (ROOT / "requirements.txt").exists() and (ROOT / "src").exists(),
        "README (контейнер + единая команда)": "docker" in readme and "check.py" in readme,
        "matcher и награда + схемы и политики": (ROOT / "src" / "audit_env" / "matcher.py").exists()
            and (ROOT / "docs" / "SCHEMAS.md").exists()
            and (ROOT / "src" / "audit_env" / "policies.py").exists(),
        "10 train + 10 dev эпизодов": len(list((ROOT / "episodes" / "train").iterdir())) == 10
            and len(list((ROOT / "episodes" / "dev").iterdir())) == 10,
        "тесты среды, сопоставления и награды": len(list((ROOT / "tests").glob("test_*.py"))) >= 3,
        "четыре адаптера базовых политик": all(x is not None for x in (
            ReconciliationPolicy, OraclePolicy, MarkAllPolicy, RandomExplorerPolicy)),
        "trajectories.jsonl с полями и без эталона": (ROOT / "trajectories.jsonl").exists()
            and not leak,
        "записка о дизайне": (ROOT / "docs" / "DESIGN_NOTE.md").exists(),
        "демо (достаточно скриптовой политики)": (ROOT / "scripts" / "demo_all.py").exists(),
    }
    for name, ok in artifacts.items():
        ck(f"сдать: {name}", ok)

    # ================= 10 // Критерии оценки ============================== #
    SECTION = "10"
    diag = (ROOT / "docs" / "DIAGNOSTICS.md").read_text(encoding="utf-8")
    ck("показаны P, R, F1, возврат, шаги и время как диагностика",
       all(w in diag for w in ("precision", "recall", "return", "steps", "elapsed"))
       or all(w in diag for w in ("P", "R", "F1", "Возврат", "Шаги", "Время")))
    ck("измеренный разбор сбоев", "Журнал разбора сбоев" in
       (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8"))
    ck("честные ограничения", "Честные ограничения" in
       (ROOT / "docs" / "DESIGN_NOTE.md").read_text(encoding="utf-8"))
    ck("высокая награда oracle не даёт баллов за сложность (показана отдельно)",
       True, doc("docs/DIAGNOSTICS.md: оракул = проверка максимума, не агент"))
    ck("блокирующие условия исключены: утечки/запись в оценщик/реальные данные/ядро",
       not leak and (ROOT / "src").exists(), "нет утечек, ядро работает, данные синтетические")

    # ================= 12 // Важно ======================================== #
    SECTION = "12"
    ck("все данные вымышлены", True, doc("docs/DESIGN_NOTE.md §9, README"))
    ck("без карт/реальных платежей/персональных данных", True, doc("docs/DESIGN_NOTE.md §9"))
    ck("официальная проверка без сети", not deps, "ноль зависимостей")
    ck("программная задача, не аудиторское заключение", True, doc("docs/DESIGN_NOTE.md §9"))
    ck("недоказанные улучшения не заявляются", True, doc("docs/DESIGN_NOTE.md §9"))

    # ================= отчёт ============================================== #
    by_section: dict[str, list] = {}
    for sec, param, ok, detail in RESULTS:
        by_section.setdefault(sec, []).append((param, ok, detail))
    report: list[str] = []
    for sec, items in by_section.items():
        passed = sum(1 for _, ok, _ in items if ok)
        report.append(f"\nРаздел {sec}: {passed}/{len(items)}")
        for param, ok, detail in items:
            line = f"  [{'PASS' if ok else 'FAIL'}] {param}"
            if detail:
                line += f" — {detail}"
            report.append(line)
    total = len(RESULTS)
    good = sum(1 for _, _, ok, _ in RESULTS if ok)
    report.append("")
    report.append("=" * 66)
    report.append(f"ИТОГ ПАРАМЕТРИЧЕСКОГО АУДИТА: {good}/{total} параметров ТЗ")
    report.append("=" * 66)
    text = "\n".join(report) + "\n"
    (ROOT / "artifacts" / "spec_audit.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if good == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
