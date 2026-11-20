#!/usr/bin/env python3
"""Марафон: N оценок жюри ПОДРЯД на одной машине.

    python scripts/jury_marathon.py            # 10 оценок подряд
    python scripts/jury_marathon.py --runs 3   # быстрый прогон

Каждая «оценка» = полный протокол раздела 06 на СВЕЖИХ seed'ах (20 эпизодов
5/5/5/5, которых не было ни раньше, ни в прошлых прогонах):
  4 политики организаторов + 20 парных проверок + независимый пересчёт эталона
  + равенства до ошибки + размеры + утечки + лимит 60 с/эпизод.

«Железо жюри» из регламента: процесс привязывается к 2 ядрам, память меряется
и сверяется с лимитом 4 ГБ, каждый эпизод — с лимитом 60 с.

Отчёт: artifacts/jury_marathon.md. Код выхода 0 только если ВСЕ оценки прошли.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jury_conditions as jc  # noqa: E402  (наш слепой протокол оценщика)

ARTIFACTS = ROOT / "artifacts"
REPORT = ARTIFACTS / "jury_marathon.md"
RAM_LIMIT_MB = 4096          # спека 07: 4 ГБ RAM
EPISODE_LIMIT_S = 60.0       # спека 07: 60 секунд на эпизод
TARGET_CORES = 2              # спека 07: 2 CPU-ядра


def pin_to_two_cores() -> str:
    """Привязываем процесс к 2 ядрам (спека 07: 2 CPU-ядра)."""
    if sys.platform != "win32":
        try:
            import os
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {0, 1})
                return "sched_setaffinity: ядра 0-1"
        except Exception:  # noqa: BLE001
            return "привязка недоступна на этой ОС"
        return "привязка недоступна на этой ОС"
    try:
        import ctypes
        mask = (1 << TARGET_CORES) - 1          # первые 2 ядра
        ok = ctypes.windll.kernel32.SetProcessAffinityMask(
            ctypes.windll.kernel32.GetCurrentProcess(), mask)
        return f"SetProcessAffinityMask: ядра 0-{TARGET_CORES - 1}" if ok \
            else "привязка отклонена ОС"
    except Exception as exc:  # noqa: BLE001
        return f"привязка недоступна: {exc}"


def memory_mb() -> tuple[float | None, float | None]:
    """(пик RSS, текущий RSS) в МБ — через ctypes (без сторонних пакетов)."""
    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes as wt

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            handle = kernel.GetCurrentProcess()
            for lib, fn in (("kernel32", "K32GetProcessMemoryInfo"),
                            ("psapi", "GetProcessMemoryInfo")):
                try:
                    func = getattr(ctypes.windll[lib], fn)
                    # без явных типов ctypes передаёт HANDLE 32-битно -> API отвечает 0
                    func.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wt.DWORD]
                    func.restype = wt.BOOL
                    counters = PMC()
                    counters.cb = ctypes.sizeof(PMC)   # поле cb обязано быть размером структуры
                    if func(handle, ctypes.byref(counters), counters.cb):
                        return (counters.PeakWorkingSetSize / 2 ** 20,
                                counters.WorkingSetSize / 2 ** 20)
                except Exception:  # noqa: BLE001
                    continue
            return None, None
        except Exception:  # noqa: BLE001
            return None, None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, None)
    except Exception:  # noqa: BLE001
        return None, None


def one_jury_evaluation(batch_seed: int, idx: int):
    """Полная оценка на своём пакете из 20 свежих seed'ов. -> (ok, detail, problems, seconds)"""
    eps = [(group, batch_seed + i, cfg) for i, (group, _, cfg) in enumerate(jc.JURY)]
    problems: list[str] = []
    counts: dict[str, int] = {"clean": 0, "single": 0, "pair": 0, "interacting": 0}
    times: list[float] = []
    leaks: list[str] = []
    recomputed = size_ok = before_ok = 0

    for group, seed, cfg in eps:
        counts[group] += 1
        files = jc.generate_for(cfg, seed)

        # 1) размеры в рамках спеки
        import json
        man = json.loads(files["open"]["manifest.json"])
        n_issues = len(json.loads(files["closed"]["truth.json"])["issues"])
        if not (200 <= man["n_records"] <= 2000 and 20 <= man["n_documents"] <= 200
                and 0 <= n_issues <= 12):
            problems.append(f"seed={seed}: размер вне рамок")

        # 2) равенства до ошибки + 3) независимый пересчёт эталона
        r_imb, r_ar, r_before, detail = jc.independent_recompute(files)
        recomputed += int(r_imb and r_ar)
        before_ok += int(r_before)
        if not (r_imb and r_ar):
            problems.append(f"seed={seed}: пересчёт {detail}")
        if not r_before:
            problems.append(f"seed={seed}: равенства до ошибки не подтверждены")

        # 4) четыре политики
        runs_before = len(times)
        _, _ = jc.run_policies(seed, cfg, [], problems, leaks, times)
        if len(times) > runs_before and max(times[runs_before:]) >= EPISODE_LIMIT_S:
            problems.append(f"seed={seed}: эпизод дольше {EPISODE_LIMIT_S:.0f} с")

    if not all(v == 5 for v in counts.values()):
        problems.append(f"состав не 5/5/5/5: {counts}")
    if recomputed != 20:
        problems.append(f"пересчёт {recomputed}/20")
    if before_ok != 20:
        problems.append(f"равенства до {before_ok}/20")
    if leaks:
        problems.append(f"утечки: {sorted(set(leaks))[:2]}")

    # 5) двадцать парных проверок
    pairs_ok = jc.paired_checks(eps, problems)
    if pairs_ok != 20:
        problems.append(f"парные {pairs_ok}/20")

    ok = not problems
    detail = (f"эпизодов 20, прогонов {len(times)}, пересчёт {recomputed}/20, "
              f"парные {pairs_ok}/20, макс. эпизод {max(times, default=0):.3f} с")
    return ok, detail, problems, max(times, default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    affinity_note = pin_to_two_cores()
    started_total = time.monotonic()
    rows = []
    all_problems: list[str] = []

    print("=" * 74)
    print(f"  МАРАФОН ОЦЕНОК ЖЮРИ: {args.runs} подряд, свежие seed на каждом прогоне")
    print(f"  Привязка: {affinity_note}")
    print("=" * 74)

    for k in range(1, args.runs + 1):
        batch_seed = 190_000 + k * 1_000        # нигде раньше не использовался
        t0 = time.monotonic()
        ok, detail, problems, max_ep = one_jury_evaluation(batch_seed, k)
        wall = time.monotonic() - t0
        peak, current = memory_mb()
        rows.append({"run": k, "seed_base": batch_seed, "ok": ok, "wall": wall,
                     "max_episode": max_ep, "peak_mb": peak, "current_mb": current,
                     "detail": detail})
        mark = "PASS" if ok else "FAIL"
        print(f"  [прогон {k:>2}/{args.runs}] {mark}  {wall:5.1f} с  "
              f"макс.эпизод {max_ep:.3f} с  RSS {current or 0:.0f} МБ  "
              f"(seed {batch_seed}..{batch_seed + 19})")
        if not ok:
            for p in problems[:5]:
                print(f"      ! {p}")
        all_problems.extend(f"прогон {k}: {p}" for p in problems)

    total_wall = time.monotonic() - started_total
    peak, current = memory_mb()

    passed = sum(1 for r in rows if r["ok"])
    checks = [
        (f"все {args.runs} оценок жюри пройдены", passed == args.runs, f"{passed}/{args.runs}"),
        ("память < 4 ГБ (спека 07)",
         peak is not None and peak < RAM_LIMIT_MB,
         f"пик {peak:.0f} МБ из 4096" if peak is not None else "ЗАМЕР НЕДОСТУПЕН (не считается пройденным)"),
        ("каждый эпизод < 60 с (спека 07)",
         all(r["max_episode"] < EPISODE_LIMIT_S for r in rows),
         f"макс {max(r['max_episode'] for r in rows):.3f} с"),
        ("привязка к 2 ядрам (спека 07)", "недоступна" not in affinity_note, affinity_note),
    ]
    good = sum(1 for _, ok, _ in checks if ok)

    report = [
        "# Марафон оценок жюри",
        "",
        f"- Прогонов: **{args.runs}** подряд, на каждом **20 свежих seed'ов** "
        f"(191000+ ; не использовались раньше)",
        f"- Привязка CPU: {affinity_note} (спека: 2 ядра)",
        f"- Пик памяти: **{peak:.0f} МБ** из 4096 (спека: 4 ГБ)" if peak else "- Память: замер недоступен",
        f"- Общее время: {total_wall:.1f} с; медленнее всего эпизод: "
        f"{max(r['max_episode'] for r in rows):.3f} с из 60",
        "",
        "| № | seed-пакет | статус | время, с | макс. эпизод, с | RSS, МБ |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for r in rows:
        report.append(f"| {r['run']} | {r['seed_base']}..+19 | "
                      f"{'PASS' if r['ok'] else 'FAIL'} | {r['wall']:.1f} | "
                      f"{r['max_episode']:.3f} | {r['current_mb'] or 0:.0f} |")
    report += ["", "## Проверки", ""]
    for name, ok, detail in checks:
        report.append(f"- {'[PASS]' if ok else '[FAIL]'} {name} — {detail}")
    if all_problems:
        report += ["", "## Проблемы", ""] + [f"- {p}" for p in all_problems]
    report.append("")
    ARTIFACTS.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(report), encoding="utf-8")

    print("-" * 74)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    print(f"\n  ИТОГ МАРАФОНА: {good}/{len(checks)} групп проверок; "
          f"оценок пройдено {passed}/{args.runs}; {total_wall:.1f} с")
    print(f"  Отчёт: {REPORT.relative_to(ROOT)}")
    return 0 if good == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
