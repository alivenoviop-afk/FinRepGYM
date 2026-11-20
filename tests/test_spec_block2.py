"""Зеркало блока 2 определений: матрица D01-D20 и шесть условий зачёта.

Двадцать эпизодов на СВЕЖИХ seed (640001+) собираются документированными
настройками генератора (docs/SCHEMAS.md §8) и проверяются по условиям
определений 20.11:

  1. до внесения ошибки книги согласованы (provenance: invariants_before);
  2. инъекция создаёт именно заявленное расхождение — НЕЗАВИСИМЫЙ пересчёт
     оценщика (jury_conditions.independent_recompute), не наш matcher;
  3. тип/ID/сумма/валюта в truth соответствуют реальному изменению;
  4. хотя бы один полный набор доказательств разрешим через открытые данные;
  5. форматы раздела 03 (смещение времени, целые суммы, десятичные FX, null);
  6. нет противоречий, размеры в пределах 200-2000 / 20-200 / 0-12;
  плюс: оракул F1 = 1.0 и сверка FP = FN = 0 на каждом эпизоде.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jury_conditions as jc  # noqa: E402
from audit_env import ISSUE_TYPES, load_episode, save_episode, generate  # noqa: E402
from support import AuditEnv, OraclePolicy, ReconciliationPolicy, run_with  # noqa: E402

# Конфиги соответствуют таблице раскладки в docs/SCHEMAS.md §8
MATRIX = [
    ("D01", {"clean": True, "distractors": False}),                 # обычная сверка
    ("D02", {"clean": True, "distractors": True}),                  # разбитый платёж
    ("D03", {"clean": True, "distractors": True}),                  # возврат
    ("D04", {"clean": True, "distractors": True}),                  # повторяющиеся суммы
    ("D05", {"clean": True, "distractors": True}),                  # граница по policy
    ("D06", {"issue_types": ["duplicate_entry"], "n_issues": 1}),
    ("D07", {"issue_types": ["duplicate_entry"], "n_issues": 1}),
    ("D08", {"issue_types": ["duplicate_entry"], "n_issues": 1}),
    ("D09", {"issue_types": ["partial_allocation"], "n_issues": 1,
             "partial_variant": "miss"}),                           # ПРОПУСК
    ("D10", {"issue_types": ["partial_allocation"], "n_issues": 1,
             "partial_variant": "partial"}),                        # ЧАСТИЧНЫЙ
    ("D11", {"issue_types": ["partial_allocation"], "n_issues": 1,
             "partial_variant": "partial"}),
    ("D12", {"issue_types": ["fx_rate"], "n_issues": 1}),
    ("D13", {"issue_types": ["fx_rate"], "n_issues": 1}),
    ("D14", {"issue_types": ["period_cutoff"], "n_issues": 1}),
    ("D15", {"issue_types": ["period_cutoff"], "n_issues": 1}),
    ("D16", {"scenarios": ["S1"]}),
    ("D17", {"scenarios": ["S1"]}),
    ("D18", {"scenarios": ["S1"]}),
    ("D19", {"scenarios": ["S2"]}),
    ("D20", {"scenarios": ["S2"]}),
]
SEED0 = 640_001
TS_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
FX_RATE = re.compile(r"^\d+\.\d{6}$")


class TestDMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="spec_b2_"))
        cls.cases = []
        for i, (name, cfg) in enumerate(MATRIX):
            seed = SEED0 + i
            episode_dir = cls.tmp / name
            files = generate(seed, dict(cfg))
            save_episode(files, str(episode_dir))
            cls.cases.append((name, seed, cfg, episode_dir))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_matrix_covers_the_required_distribution(self):
        self.assertEqual(len(self.cases), 20)
        configs = [cfg for _, _, cfg, _ in self.cases]
        self.assertEqual(sum(1 for c in configs if c.get("clean")), 5, "D01-D05")
        self.assertEqual(sum(1 for c in configs
                             if c.get("issue_types") == ["duplicate_entry"]), 3)
        partials = [c for c in configs
                    if c.get("issue_types") == ["partial_allocation"]]
        self.assertEqual(len(partials), 3, "D09-D11")
        self.assertEqual({c.get("partial_variant") for c in partials},
                         {"miss", "partial"}, "нужны и пропуск, и частичный")
        self.assertEqual(sum(1 for c in configs
                             if c.get("issue_types") == ["fx_rate"]), 2)
        self.assertEqual(sum(1 for c in configs
                             if c.get("issue_types") == ["period_cutoff"]), 2)
        self.assertEqual(sum(1 for c in configs if c.get("scenarios") == ["S1"]), 3)
        self.assertEqual(sum(1 for c in configs if c.get("scenarios") == ["S2"]), 2)

    def test_every_episode_passes_all_conditions(self):
        for name, seed, cfg, episode_dir in self.cases:
            with self.subTest(episode=name, seed=seed):
                files = load_episode(str(episode_dir))
                manifest = json.loads(files["open"]["manifest.json"])
                truth = json.loads(files["closed"]["truth.json"])["issues"]
                prov = json.loads(files["closed"]["provenance.json"])

                # -- условие 6: размеры и отсутствие противоречий ---------- #
                self.assertTrue(200 <= manifest["n_records"] <= 2000,
                                f"{name}: {manifest['n_records']} строк")
                self.assertTrue(20 <= manifest["n_documents"] <= 200)
                self.assertTrue(0 <= len(truth) <= 12)

                # -- условие 1: книги согласованы ДО ошибки ----------------- #
                before = prov["invariants_before"]
                self.assertTrue(before["books_balanced"],
                                f"{name}: до мутации книги не сбалансированы")
                self.assertEqual(before["sum_debits"], before["sum_credits"])

                # -- условие 2: независимый пересчёт оценщика --------------- #
                r_imb, r_ar, r_before, detail = jc.independent_recompute(files)
                self.assertTrue(r_before, f"{name}: равенства до ошибки ({detail})")
                self.assertTrue(r_imb, f"{name}: дисбаланс книг ({detail})")
                self.assertTrue(r_ar, f"{name}: AR-период ({detail})")
                verified = prov["stages"][-1]["declared_discrepancies"]["verified"]
                self.assertTrue(verified, f"{name}: заявленное не сходится")

                # -- условие 3: truth соответствует реальному изменению ----- #
                if cfg.get("clean"):
                    self.assertEqual(truth, [], f"{name}: чистый эпизод с эталоном")
                else:
                    self.assertGreaterEqual(len(truth), 1, name)
                env = AuditEnv(mode="train")
                env.reset(config={"episode_dir": str(episode_dir)})
                row_ids = {r["row_id"] for r in env._tables["ledger"]}
                pay_ids = {p["payment_id"] for p in env._tables["payments"]}
                for issue in truth:
                    self.assertIn(issue["type"], ISSUE_TYPES)
                    self.assertGreater(issue["expected_amount_minor"], 0)
                    self.assertIn(issue["currency"], manifest["currencies"])
                    for affected in issue["affected_ids"]:
                        self.assertTrue(affected in row_ids or affected in pay_ids,
                                        f"{name}: {affected} нет в открытых данных")

                # -- условие 4: полный набор доказательств разрешим --------- #
                valid = env._valid_evidence_ids
                for issue in truth:
                    self.assertTrue(
                        any(set(s) <= valid for s in issue["evidence_sets"]),
                        f"{name}: набор доказательств {issue['issue_id']} недоступен")

                # -- условие 5: форматы раздела 03 -------------------------- #
                for line in files["open"]["ledger.csv"].strip().splitlines()[1:]:
                    ts = line.split(",")[1]
                    self.assertRegex(ts, TS_OFFSET, f"{name}: время без смещения")
                for line in files["open"]["fx.csv"].strip().splitlines()[1:]:
                    self.assertRegex(line.split(",")[3], FX_RATE,
                                     f"{name}: курс не десятичная строка")

    def test_oracle_perfect_and_reconciliation_silent_on_every_episode(self):
        for name, seed, cfg, episode_dir in self.cases:
            with self.subTest(episode=name):
                config = {"episode_dir": str(episode_dir)}
                oracle = run_with(lambda e: OraclePolicy(e.evaluator_truth()),
                                  seed, config)
                om = oracle["report"]["metrics"]
                self.assertEqual(om["f1"], 1.0, f"{name}: оракул F1={om['f1']}")
                recon = run_with(lambda e: ReconciliationPolicy(), seed, config)
                rm = recon["report"]["metrics"]
                self.assertEqual((rm["fp"], rm["fn"]), (0, 0),
                                 f"{name}: сверка FP/FN={rm['fp']}/{rm['fn']}")
                self.assertEqual(rm["f1"], 1.0, f"{name}: сверка F1={rm['f1']}")


if __name__ == "__main__":
    unittest.main()
