"""Зеркало блока 4 определений: 20 парных проверок с раскладкой 5/5/10 (15 баллов).

  P01-P05 — только перестановка порядка записей;
  P06-P10 — только взаимно однозначная замена ID;
  P11-P20 — и порядок, и замена ID вместе.

Для каждой пары (определения 20.11, «Pass/fail пары»):
  1. исходный эпизод воспроизводится: повтор тех же действий даёт те же
     метрики и награду;
  2. на преобразованной копии оракул отправляет ТЕ ЖЕ находки, перенесённые
     через обратное отображение ID (remap_submission);
  3. TP/FP/FN/P/R/F1 и терминальная награда совпадают;
  4. новый канал утечки не появился (сканер закрытых токенов в наблюдениях).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_env import (  # noqa: E402
    AuditEnv,
    generate,
    permute_episode,
    remap_submission,
    save_episode,
)

SEED0 = 660_001
CFG = {"n_invoices": 70, "issue_types": ["duplicate_entry", "period_cutoff",
                                         "partial_allocation", "fx_rate"],
       "n_issues": 4}
LEAK_TOKENS = ("truth", "provenance", "seed", "iss_", "expected_amount",
               "causal_change", "invariants_", "evidence_sets")


def correct_findings(truth):
    return [{"type": i["type"], "affected_ids": sorted(i["affected_ids"]),
             "claimed_amount_minor": i["expected_amount_minor"],
             "currency": i["currency"],
             "evidence_ids": list(i["evidence_sets"][0]),
             "explanation": "oracle"} for i in truth]


class TestPairedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="spec_b4_"))
        cls.pairs = []
        for i in range(20):
            seed = SEED0 + i
            kind = "order" if i < 5 else ("rename" if i < 10 else "both")
            cls.pairs.append((f"P{i + 1:02d}", seed, kind))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def oracle_on(self, files, tag, findings_from=None):
        episode_dir = self.tmp / tag
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        save_episode(files, str(episode_dir))
        env = AuditEnv(mode="train")
        env.reset(config={"episode_dir": str(episode_dir)})
        truth = env.evaluator_truth()
        findings = (correct_findings(truth) if findings_from is None
                    else findings_from)
        env.step({"action": "submit", "episode_id": env._manifest["episode_id"],
                  "findings": findings})
        report = env.evaluator_report()
        # сканер утечек на наблюдениях преобразованного эпизода
        blob = json.dumps(env._manifest, ensure_ascii=False)
        for token in LEAK_TOKENS:
            self.assertNotIn(token, blob, f"{tag}: утечка {token} в манифесте")
        return report["metrics"], report["true_reward"], env._manifest["episode_id"]

    def test_all_twenty_pairs_keep_metrics_and_reward(self):
        checked = {"order": 0, "rename": 0, "both": 0}
        for name, seed, kind in self.pairs:
            with self.subTest(pair=name, kind=kind):
                files = generate(seed, dict(CFG))
                order = kind in ("order", "both")
                rename = kind in ("rename", "both")

                # 1) исходный эпизод: повтор даёт идентичный результат
                m1a, r1a, ep1 = self.oracle_on(files, f"{name}_orig_a")
                m1b, r1b, _ = self.oracle_on(files, f"{name}_orig_b")
                self.assertEqual((m1a["tp"], m1a["fp"], m1a["fn"], m1a["f1"]),
                                 (m1b["tp"], m1b["fp"], m1b["fn"], m1b["f1"]),
                                 f"{name}: повтор исходного эпизода не совпал")
                self.assertEqual(r1a, r1b)
                self.assertEqual(m1a["f1"], 1.0, f"{name}: оракул обязан быть идеален")

                # 2) преобразование применяется заявленным образом
                transformed, id_map = permute_episode(
                    files, seed=seed + 77_777, order=order, rename=rename)

                def order_signature(ledger_csv: str) -> list[str]:
                    # порядок строк БЕЗ row_id: (timestamp, account, debit, credit)
                    return [",".join(line.split(",")[1:5])
                            for line in ledger_csv.strip().splitlines()[1:]]

                if order:
                    self.assertNotEqual(
                        order_signature(transformed["open"]["ledger.csv"]),
                        order_signature(files["open"]["ledger.csv"]),
                        f"{name}: порядок записей не изменился")
                else:
                    self.assertEqual(
                        order_signature(transformed["open"]["ledger.csv"]),
                        order_signature(files["open"]["ledger.csv"]),
                        f"{name}: порядок записей не должен меняться")
                if rename:
                    self.assertNotEqual(id_map["episode_id"][0],
                                        id_map["episode_id"][1],
                                        f"{name}: episode_id не изменился")
                    old_rows = {l.split(",")[0]
                                for l in files["open"]["ledger.csv"]
                                .strip().splitlines()[1:]}
                    new_rows = {l.split(",")[0]
                                for l in transformed["open"]["ledger.csv"]
                                .strip().splitlines()[1:]}
                    self.assertNotEqual(old_rows, new_rows,
                                        f"{name}: row_id не изменились")
                else:
                    self.assertEqual(id_map["episode_id"][0],
                                     id_map["episode_id"][1])
                    for domain in ("rows", "invoices", "payments"):
                        self.assertEqual(id_map[domain],
                                         {k: k for k in id_map[domain]},
                                         f"{name}: ID не должны меняться")

                # 3) те же находки с новыми ID -> те же метрики и награда
                remapped = remap_submission({"findings": correct_findings(
                    json.loads(files["closed"]["truth.json"])["issues"])},
                    id_map)["findings"]
                m2, r2, ep2 = self.oracle_on(transformed, f"{name}_xformed",
                                             findings_from=remapped)
                self.assertEqual(ep2, id_map["episode_id"][1])
                self.assertEqual((m2["tp"], m2["fp"], m2["fn"]),
                                 (m1a["tp"], m1a["fp"], m1a["fn"]),
                                 f"{name}: метрики после преобразования разошлись")
                self.assertAlmostEqual(m2["f1"], m1a["f1"], places=9)
                self.assertEqual(r2, r1a,
                                 f"{name}: награда после преобразования разошлась")
                checked[kind] += 1

        self.assertEqual(checked, {"order": 5, "rename": 5, "both": 10},
                         "раскладка парных проверок должна быть 5/5/10")


if __name__ == "__main__":
    unittest.main()
