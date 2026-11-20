# Слепая проверка на условиях оценщиков

Seed'ы 97001–97020 (не использовались нигде) и **новые сочетания параметров**; отчёт сформирован автоматически.

- [clean 1/5] seed=97001 rows=280 docs=132 issues=0 :: n_invoices=65,clean=True,scenarios=[],utc_offset=+00:00
- [clean 2/5] seed=97002 rows=306 docs=153 issues=0 :: n_invoices=90,clean=True,scenarios=[],distractors=False
- [clean 3/5] seed=97003 rows=306 docs=140 issues=0 :: n_invoices=70,clean=True,scenarios=[],currencies=['USD']
- [clean 4/5] seed=97004 rows=280 docs=133 issues=0 :: n_invoices=65,clean=True,scenarios=['S1', 'S2']
- [clean 5/5] seed=97005 rows=300 docs=138 issues=0 :: n_invoices=65,clean=True,scenarios=[],utc_offset=-03:00
- [single 1/5] seed=97006 rows=236 docs=118 issues=1 :: n_invoices=70,issue_types=['fx_rate'],distractors=False
- [single 2/5] seed=97007 rows=261 docs=126 issues=1 :: n_invoices=65,issue_types=['duplicate_entry']
- [single 3/5] seed=97008 rows=264 docs=120 issues=1 :: n_invoices=60,issue_types=['partial_allocation']
- [single 4/5] seed=97009 rows=328 docs=149 issues=1 :: n_invoices=75,issue_types=['period_cutoff'],utc_offset=+06:00
- [single 5/5] seed=97010 rows=392 docs=181 issues=1 :: n_invoices=90,issue_types=['fx_rate']
- [pair 1/5] seed=97011 rows=295 docs=137 issues=2 :: n_invoices=65,issue_types=['duplicate_entry', 'fx_rate']
- [pair 2/5] seed=97012 rows=292 docs=139 issues=2 :: n_invoices=70,issue_types=['partial_allocation', 'period_cutoff']
- [pair 3/5] seed=97013 rows=254 docs=116 issues=2 :: n_invoices=60,issue_types=['fx_rate', 'period_cutoff']
- [pair 4/5] seed=97014 rows=407 docs=183 issues=2 :: n_invoices=90,issue_types=['duplicate_entry', 'partial_allocation']
- [pair 5/5] seed=97015 rows=278 docs=128 issues=2 :: n_invoices=65,issue_types=['partial_allocation', 'fx_rate']
- [interacting 1/5] seed=97016 rows=240 docs=120 issues=2 :: n_invoices=70,issue_types=[],scenarios=['S1'],distractors=False
- [interacting 2/5] seed=97017 rows=276 docs=130 issues=2 :: n_invoices=70,issue_types=[],scenarios=['S2']
- [interacting 3/5] seed=97018 rows=308 docs=141 issues=4 :: n_invoices=65,issue_types=[],scenarios=['S1', 'S2']
- [interacting 4/5] seed=97019 rows=337 docs=156 issues=3 :: n_invoices=80,issue_types=['duplicate_entry'],scenarios=['S1']
- [interacting 5/5] seed=97020 rows=278 docs=128 issues=3 :: n_invoices=65,issue_types=['fx_rate'],scenarios=['S2']

## Проверки

- [PASS] состав 5/5/5/5 (чистые/одиночные/парные/взаимодействия) — {'clean': 5, 'single': 5, 'pair': 5, 'interacting': 5}
- [PASS] размеры 200-2000 / 20-200 / 0-12 на всех 20 — 20/20
- [PASS] равенства бухгалтерии ДО внесения ошибки (provenance) — 20/20
- [PASS] НЕЗАВИСИМЫЙ пересчёт эталона из открытых записей — 20/20
- [PASS] оракул достигает максимума на всех эпизодах — F1 = 1.0
- [PASS] сверка без FP/FN на всех эпизодах — FP = 0, FN = 0
- [PASS] каждый прогон < 60 секунд — max=0.032 с, прогонов 80
- [PASS] нет закрытого эталона в наблюдениях — чисто
- [PASS] 20 парных проверок: ответ и награда сохраняются — 20/20

## Сводка политик (среднее по 20 эпизодам)

| Политика | F1 | Возврат | Шаги |
|---|---:|---:|---:|
| oracle | 1.000 | +1.000 | 1.0 |
| reconciliation | 1.000 | +0.980 | 11.2 |
| mark_all | 0.000 | -1.000 | 7.4 |
| random | 0.000 | -1.000 | 21.0 |

Время: 5.3 с. Прогонов: 80.
