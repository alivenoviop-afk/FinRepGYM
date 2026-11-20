# Самоотчёт: карта покрытия регламента

> **Это самоотчёт команды, а не проверки организаторов.** Их определения проверок
> зафиксированы и публикуются отдельно (спека, раздел 10: «25 × пройдено/20» и т. д.);
> данный документ **не заменяет их и не подменяет** — он помогает быстро найти
> доказательство по каждой теме регламента.
>
> Все ссылки — на реальные тесты/скрипты репозитория. Обновлено: 20.11.2026.

Как читать: **тема** → **где доказано** (тест `tests/…` или скрипт/документ).

---

## A. Контракт и изоляция — 25 (20 тем)

| # | Тема | Доказательство |
|---|---|---|
| 1 | `reset` возвращает (observation, info) | `test_reset_returns_observation_and_info` |
| 2 | Первое наблюдение: задача, manifest, политика, бюджет шагов | он же |
| 3 | `step` возвращает 5 значений в стиле Gymnasium | `test_unknown_action_costs_a_step` и все шаговые тесты |
| 4 | Ровно 5 действий, лишнее = ошибка | `test_unknown_action_costs_a_step`, `test_action_must_be_object` |
| 5 | Страница ≤ 50 строк, курсоры стабильны | `test_pagination_limit_and_stable_cursor` |
| 6 | `limit > 50` и `limit < 1` = ошибка + шаг | `test_page_over_limit_…`, `test_zero_limit_rejected` |
| 7 | Неизвестный id / doc = ограниченная ошибка + шаг | `test_unknown_id_costs_a_step`, `test_get_document_roundtrip` |
| 8 | Калькулятор: только арифметика, без кода | `test_calculator_accepts_arithmetic_only` |
| 9 | Деление на ноль = ошибка | `test_calculator_accepts_arithmetic_only` |
| 10 | 100 шагов → `truncated` + пустой итоговый ответ | `test_truncation_at_100_steps_with_empty_final_answer` |
| 11 | `submit` → `terminated` (в т. ч. на 100-м шаге) | `test_submit_terminates_…`, `test_submit_allowed_on_the_last_budget_step` |
| 12 | Лимит времени 60 с → обрезка ДО действия | `test_time_limit_truncates_with_empty_final_answer` |
| 13 | `reset` изолирует состояние эпизода | `test_reset_isolates_episode_state` |
| 14 | Повтор seed+действий → идентичный эпизод | `test_same_seed_same_records` |
| 15 | Промежуточная награда всегда 0 (train) | `test_train_mode_returns_terminal_reward_only` |
| 16 | eval: награда 0 везде, диагностика у оценщика | `test_eval_mode_withholds…`, `test_eval_report_is_not_in_info` |
| 17 | Нет эталона/зерна в наблюдениях | `test_observations_carry_no_closed_truth` + сканер в `build_trajectories`/`organizer_check` |
| 18 | Закрытые файлы вне открытой зоны | `test_closed_files_are_not_in_the_open_set` |
| 19 | Ошибки ограничены (без списка правильных ответов и эталона) | код ошибок в `docs/SCHEMAS.md` §7, тесты неверного ввода |
| 20 | Наследование `gym.Env` не требуется; схемы опубликованы | `docs/SCHEMAS.md` §1–2 |

## B. Данные и эталон — 25 (20 тем)

| # | Тема | Доказательство |
|---|---|---|
| 1 | Баланс книг ДО внесения ошибки | `test_books_balance_before_the_mutation` |
| 2 | Заявленное расхождение ПОСЛЕ = независимый пересчёт из открытых файлов | `test_declared_discrepancy_matches_open_records` |
| 3 | Подписанные эффекты мутаций сходятся с наблюдениями | он же + `_verify_declared` в генераторе |
| 4 | Каждая мутация имеет причинную запись «до/после» | `test_mutation_is_recorded_with_cause_and_effect` |
| 5 | Все 4 типа детектируются сверкой без FP/FN | `test_reconciliation_finds_all_four_implemented_types` |
| 6 | Сценарий S1 (курс + граница) = 2 ошибки на одном платеже | `test_s1_is_fx_plus_cutoff_on_one_payment` |
| 7 | Сценарий S2 (частичный платёж в другой валюте у границы) | `test_s2_is_partial_plus_cutoff_on_one_payment`, `test_s2_payment_is_really_cross_currency` |
| 8 | Наивный детектор теряет ошибку взаимодействия | `test_naive_date_only_detector_misses_the_other_errors` |
| 9 | Без конвертации дельта считается неверно | `test_naive_amount_comparison_without_conversion_is_wrong` |
| 10 | Чистые эпизоды: эталон пуст, 0 FP | `test_reconciliation_stays_silent_on_clean_books` |
| 11 | Возвраты законны и не флагаются | `test_returns_are_legal_and_never_flagged` |
| 12 | Отвлекающие: разбитые платежи, повторы, неоплаченные | свипы `check.py` (чистые конфиги), `test_attacks` |
| 13 | Мультивалюта: документы в своей валюте | `test_documents_use_several_currencies` |
| 14 | Кросс-платежи постятся в базовую (без дрейфа округления) | `test_cross_currency_payment_posts_in_base` |
| 15 | fx.csv: курс всех пар на policy-дату | `test_fx_rates_cover_policy_date_for_every_pair` |
| 16 | Сравнение валют без конвертации запрещено | `policy.json` (comparison_rule), `test_fx_error_is_recomputed_from_open_records` |
| 17 | Каждый evidence id разрешим в открытых файлах | `test_every_evidence_id_resolves_in_open_files` |
| 18 | **Каждый полный набор доказателен: сумма пересчитывается** | `test_evidence_sufficiency` (все наборы, 4 конфига) |
| 19 | Размеры и форматы: 200–2000 / 20–200 / 0–12, ISO, целые суммы, decimal fx, явный null | `test_episode_size…`, `test_formats`, `test_missing_references…` |
| 20 | Вариативность: имена, ID, порядок, валюты, пояса от seed | `test_offsets_are_varying_and_consistent`, свип поясов в `check.py`, вариативность префиксов (см. `docs/RULES.md` оси) |

## C. Надёжность награды — 25 (20 тем)

| # | Тема | Доказательство |
|---|---|---|
| 1 | Один общий matcher для всех (код сопоставления) | `src/audit_env/matcher.py` |
| 2 | Нормализация опубликована | `docs/SCHEMAS.md` §4, README «Открытые решения» |
| 3 | Неоднозначное совпадение: жадное правило опубликовано | README §«Сопоставление», `match()` |
| 4 | TP требует тип + ID + валюту + сумму + полный набор | `test_perfect_match` и отрицательные тесты |
| 5 | Порядок ID и регистр валюты не важны | `test_id_order_and_currency_case_do_not_matter` |
| 6 | Неверная сумма/тип/валюта ≠ TP | `test_wrong_amount…`, `test_wrong_type…`, `test_wrong_currency…` |
| 7 | Частичные/пустые доказательства ≠ TP | `test_partial_evidence…`, `test_missing_evidence…` |
| 8 | Угаданный ID ≠ TP | `test_guessing_an_id_is_not_enough`, `test_guessed_ids_score_nothing` |
| 9 | Повтор находки → FP, выгода нулевая | `test_duplicate_submission_gains_nothing` |
| 10 | Назначение взаимно однозначное (1-к-1) | `test_matching_is_one_to_one` |
| 11 | Краевые случаи F1 (оба пусты / эталон пуст / нулевой знаменатель) | `TestMetricEdgeCases` (4 теста) |
| 12 | Форма награды и зажимы | `TestReward.test_formula`, `test_clip_bounds` |
| 13 | Штраф за шаги мал | `test_step_cost_is_small` |
| 14 | Текст объяснения не влияет на награду | `test_reward_ignores_explanation_text` |
| 15 | Атака «пометить всё» → F1=0, −1 | `test_mark_all_is_bad` + `check.py` |
| 16 | Полный ответ > молчание на грязных; тишина > обвинений на чистых | `test_correct_answer_beats_silence…`, `test_silence_beats_false_accusations…` |
| 17 | Неверная структура → −1 (объяснение, лимиты, типы полей, episode_id) | `test_malformed_…`, `test_too_long_explanation…`, `test_too_many_evidence_ids…`, `test_non_integer_amount…`, `test_wrong_episode_id…`, `test_too_many_findings…` |
| 18 | Повторная отправка отклонена | `test_resubmission_is_rejected` |
| 19 | Максимум награды достижим (оракул) | `check.py` (свипы 13×6), `organizer_check.py` (20×4): F1=1.0 везде |
| 20 | Награда/диагностика скрыты в eval; возвраты отдельно показаны | `test_eval_mode…`, `docs/DIAGNOSTICS.md` |

## D. Воспроизводимость и обобщение — 15 (20 тем)

| # | Тема | Доказательство |
|---|---|---|
| 1 | Повтор seed+config → байт-в-байт те же файлы | `test_generation_is_deterministic`, `check.py` §4 |
| 2 | Повтор эпизода → идентичная траектория и награда | `test_same_seed_same_actions_same_rewards` |
| 3 | Перестановка строк → ответ сохранён | `test_transform_changes_order_and_ids` |
| 4 | Смена ID → находки переносятся (remap) | `test_oracle_answer_survives_id_renaming` |
| 5 | Награда сохраняется до 12 знаков после преобразования | он же |
| 6 | Сверка верна после преобразования (FP=FN=0) | `test_reconciliation_generalizes_to_permuted_episode` |
| 7 | Состав/суммы/даты/валюты преобразование не меняет | `test_amounts_dates_and_currencies_are_untouched` |
| 8 | Старые ID не остаются в открытых файлах | `test_no_old_ids_survive_in_open_files` |
| 9 | Эталон ссылается только на существующие записи после смены ID | `test_truth_ids_exist_in_permuted_open_files` |
| 10 | В наблюдениях перестановки нет эталона | `test_no_leak_in_permuted_trajectory` |
| 11 | **20 парных проверок** проходят | `check.py` §7: 20/20 (спека: «Ещё 20 парных проверок») |
| 12 | Ось seed/конфиг: повторяемая генерация | `episodes/registry.json`, свипы `check.py` |
| 13 | Ось `clean` (чистые эпизоды) | свип «чистые» в `check.py` |
| 14 | Ось `issue_types` (4 типа) | свипы «одиночные/парные/четыре типа» |
| 15 | Ось `n_issues` (0–12) | `test_issue_count_between_zero_and_twelve` |
| 16 | Ось `scenarios` (S1/S2) | свип «сценарии S1+S2», «одиночные + сценарии» |
| 17 | Ось размер (200–2000 строк) | `test_episode_size_within_spec` |
| 18 | Ось валюты (USD/EUR/KZT) и кросс-платежи | `TestMultiCurrency` |
| 19 | Ось часовые пояса (+03:00, −07:00, +09:30) | `check.py` §8, `TestTimeZoneAxis` |
| 20 | Открытые train/dev отделены от закрытых (их seed + их сочетания) | `episodes/registry.json`, `organizer_check.py --seeds`, README «Эпизоды» |

## E. Инженерия и объяснение — 10 (5 критериев × 0/1/2)

| Критерий | Доказательство | Оценка |
|---|---|---|
| Запуск одной командой | `python scripts/check.py` (32/32) и `docker run --rm finrepgym`; CI гоняет обе | 2 |
| Версии конфигов | `schema_version: finrepgym/1.0`, `audit_env.__version__`, Python/платформа в отчёте `check.py`, реестр seed/конфигов `episodes/registry.json`, Python 3.11 в CI и контейнере | 2 |
| Журналы траекторий | `trajectories.jsonl` (60 прогонов, 839 записей; оракул — у оценщика), `artifacts/diagnostics.csv` (80 прогонов), `artifacts/organizer_runs.csv`, `docs/DIAGNOSTICS.md` | 2 |
| Измеренный разбор сбоев | `docs/ARCHITECTURE.md` §12: 14 записей (как поймано → метрика → фикс); замеры: время прогона (max 0.016 с), размер решения (≈9 МБ), память (шаг «Замер памяти и размера» в CI) | 2 |
| Честные ограничения | `docs/DESIGN_NOTE.md` §9 (7 пунктов), `docs/SUMMARY.md`; заявлений об улучшении модели нет | 2 |

---

## Дополнительные доказательства (доп-пакет, 20.11)

| Доказательство | Спека | Как проверить |
|---|---|---|
| **Эталонный отпечаток эпизода**: seed=42 → SHA-256 `05eb2eb1…1ebf8`, одинаковый на Windows/3.12 и в CI на Linux/3.11 | 04: «одинаковые seed, config, **версия** … воспроизводят записи» | `python scripts/golden_hash.py` и `tests/test_golden.py` |
| **Среда ничего не пишет**: полный прогон (все таблицы, калькулятор, вредоносные команды, submit) не меняет ни один байт папки эпизода | 10: «запись в оценщик» — блокирующее условие | `tests/test_isolation_writes.py` |
| **Фаззинг**: 500 вредоносных/случайных действий — среда не падает, всегда JSON, бюджет шагов не превышен | 05: «проверьте … неверный ввод» | `tests/test_fuzz.py` (зерно фиксировано) |
| **Пограничные даты**: счёт/платёж в первый день и ровно в последний момент периода; на границе FP=FN=0 (inclusive) | 02/04: правдоподобие + границы `policy.json` включительно | `tests/test_data.py::TestBoundaryDates` |
| **Порядок находок в `submit` не влияет** на P/R/F1 и награду (прямой/обратный/перемешанный) | 05: сопоставление взаимно однозначное, порядок не важен | `tests/test_metrics.py::test_finding_order_does_not_change_metrics_or_reward` |
| **Баланс по каждой валюте отдельно** на чистых эпизодах (USD/EUR/KZT: Σдебет = Σкредит) | 02: эталон описывает реальное расхождение | `tests/test_data.py::test_each_currency_balanced_separately_on_clean` |

---

## Итог самоотчёта

- Тем покрыто: **A 20 + B 20 + C 20 + D 20 = 80**; инженерия — 5 критериев.
- Автоматизировано: 117 юнит-тестов, `check.py` 32/32, `organizer_check.py` 5/5 (20 эпизодов × 4 политики),
  **параметрический аудит `spec_audit.py` — 100/100 параметров разделов 02–12**,
  **слепая проверка `jury_conditions.py` — 9/9 на условиях оценщиков** (свежие seed,
  новые сочетания параметров, независимый пересчёт эталона 20/20),
  **марафон `jury_marathon.py` — 10/10 оценок подряд** (200 эпизодов, привязка к 2 ядрам,
  пик памяти 28 МБ из 4096, макс. эпизод 0.031 с из 60)
  (отчёты: `artifacts/spec_audit.txt`, `artifacts/jury_report.md`),
  CI на Python 3.11 + сборка контейнера + замер ресурсов + офлайн-режим `python -S`.
- Чего **не делаем сознательно**: обучение весов (не даёт баллов), пятый тип ошибок (типы известны заранее), реальные данные (всё вымышленное), заявления об улучшении модели (нет контролируемого сравнения — и мы ничего не заявляем).
