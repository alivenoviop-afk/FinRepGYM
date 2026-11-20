# Схемы FinRepGYM

> Явное описание контрактов (пункт сдачи: «Matcher и награда — общий код сопоставления
> и награды, **схемы** и политики»). Версия схемы: `finrepgym/1.0`.
> Реализация: `src/audit_env/`. Python >= 3.11, внешних зависимостей нет.

## 1. Среда: reset / step

```
reset(seed: int|null, config: object|null) -> (observation: object, info: object)

step(action: object) -> (observation: object,
                         reward: float,
                         terminated: bool,
                         truncated: bool,
                         info: object)
```

**Первое наблюдение (после reset):**

```json
{
  "task": "текст задачи для агента",
  "manifest": { ...manifest.json эпизода... },
  "policy":   { ...policy.json эпизода + "brief": ["строки правил"] },
  "steps_left": 100
}
```

**Наблюдение на каждом шаге** — ровно одна из трёх форм + `steps_left`:

```json
{ "result": { ... }, "steps_left": 99 }          // успех
{ "error":  { "code": "...", "message": "..." }, "steps_left": 99 }  // ограниченная ошибка
{ "result": { "accepted": true, "findings_count": 2 }, "steps_left": 98 }  // после submit
```

Через `judge_adapter.py` то же наблюдение приходит в каноническом виде
определений: `{"result", "error", "remaining_steps"}` (незаполненное поле —
`null`), а `submit` отдаёт ровно `{"accepted": true}` (без `findings_count`).

`info` в `reset`: `{"mode": "train"|"eval", "steps_used": 0, "schema_version": "finrepgym/1.0"}`;
`info` в `step`: `{"mode": "train"|"eval", "steps_used": int}` — закрытых данных нет никогда.

**Флаги завершения:**

| Ситуация | terminated | truncated | reward (train) |
|---|---|---|---|
| `submit` принят | `true` | `false` | `clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1)` |
| `submit` отклонён (плохая структура) | `true` | `false` | `-1.0` |
| израсходовано 100 шагов | `false` | `true` | награда за **пустой** ответ |
| вышло время (60 с) — действие не выполняется | `false` | `true` | награда за **пустой** ответ |
| все промежуточные шаги | `false` | `false` | `0.0` |
| шаг после завершения → ошибка `episode_finished` | `true` | `false` | `0.0` (состояние не меняется) |

`inspection_steps` — число действий **до** `submit` (включая ошибочные): сам submit
входит в лимит 100 действий, но в штраф за шаги не входит.

В режиме `eval` политика получает `reward = 0.0` на всех шагах; истинная награда и
метрики доступны только через `evaluator_report()` (вне контракта).

## 2. Действия (5 штук, только чтение)

| Действие | Поля | Лимиты |
|---|---|---|
| `list_records` | `table`, `cursor` (int >= 0), `limit` (int) | `table` ∈ {ledger, invoices, payments, fx}; `1 <= limit <= 50` |
| `get_record` | `table`, `id` (непустая строка) | id должен существовать |
| `get_document` | `doc_id` (непустая строка) | doc_id ∈ invoices/payments |
| `calculator` | `expression` (строка) | только цифры, `.`, `+ - * /`, скобки; длина <= 200 |
| `submit` | `action="submit"`, `episode_id`, `findings` (см. §3) | `episode_id` = manifest.episode_id |

Результат `list_records`:

```json
{ "table": "ledger", "cursor": 0, "count": 50, "total": 342,
  "next_cursor": 50, "rows": [ ... ] }   // next_cursor: null на последней странице
```

## 3. Ответ агента: findings.json

```json
{
  "episode_id": "ep_691f01b2b0e3",
  "findings": [
    {
      "type": "duplicate_entry | partial_allocation | fx_rate | period_cutoff",
      "affected_ids": ["row_00017"],
      "claimed_amount_minor": 13743500,
      "currency": "USD",
      "evidence_ids": ["ledger:row_00017", "ledger:row_00016", "invoice:INV-2026-0008"],
      "explanation": "короткое обоснование"
    }
  ]
}
```

**Лимиты (превышение = отклонение всей отправки, награда −1):**

| Параметр | Лимит |
|---|---|
| находок в `findings` | <= 30 |
| `evidence_ids` на находку | <= 10 |
| `explanation` | строка длиной <= 1000 символов |
| `episode_id` | обязан совпадать с manifest эпизода |

Обязательные поля находки: `type`, `affected_ids`, `claimed_amount_minor` (целое),
`currency`, `evidence_ids`, `explanation`.

Лимиты проверяются **до** нормализации множеств; длина `explanation` считается в
Unicode code points; `claimed_amount_minor` — JSON-целое, `bool` не принимается за
денежную сумму; `affected_ids`/`evidence_ids` обязаны быть списками строк.

**Префиксы `evidence_ids`:** `ledger:<row_id>`, `invoice:<invoice_id>`,
`payment:<payment_id>`, `fx:<date>:<source>:<target>`, `policy:<rule>`.
Ссылка на несуществующий ID → находка не сопоставляется (FP), а не авария.

## 4. Источник истины (закрытая зона, агенту недоступна)

```json
// truth.json
{ "episode_id": "...", "issues": [
  { "issue_id": "iss_duplicate_0001",
    "type": "duplicate_entry",
    "affected_ids": ["row_00016", "row_00017"],
    "expected_amount_minor": 13743500,
    "currency": "USD",
    "amount_tolerance_minor": 0,
    "evidence_sets": [ ["ledger:row_00017", "ledger:row_00016", "invoice:INV-..."] ],
    "statement": "описание реального расхождения" } ] }
```

`provenance.json`: seed, config, стадии генерации, для каждой мутации —
`causal_change` (что изменено), снимки `before`/`after`, подписанные эффекты
`effect_imbalance` / `effect_ar_diff` в базовой валюте, флаг `verified`.

## 5. Открытые файлы эпизода

| Файл | Схема |
|---|---|
| `manifest.json` | `schema_version`, `episode_id`, `base_currency`, `currencies` {валюта: разрядность}, `minor_unit_digits`, `period_start`, `period_end` (ISO-8601 со смещением), `timezone`, `utc_offset`, `n_records`, `n_documents`, `tables`, `rounding`, `null_token`, `issue_type_vocabulary`, `page_limit` (50), `max_steps` (100) |
| `ledger.csv` | `row_id, timestamp, account, debit_minor, credit_minor, currency, entity_id, invoice_id, payment_id` — суммы целые (минимальные единицы), отсутствующие ссылки = `null` |
| `invoices.jsonl` | `doc_id, invoice_id, entity_id, issued_at, due_at, currency, total_minor, line_items[]` (сумма позиций = `total_minor`) |
| `payments.jsonl` | `doc_id, payment_id, timestamp, currency, amount_minor, invoice_refs[]` |
| `fx.csv` | `date, source_currency, target_currency, rate` — `rate` десятичная строка (6 знаков) |
| `policy.json` | `fx_rate_date` (= `period_end`), `period_boundary` (включительно), `rounding` (half-up), `comparison_rule` (без конвертации сравнивать нельзя), `evidence_rule` |

## 6. Метрики и награда

```
P = TP / (TP + FP)   при нулём знаменателе → 0
R = TP / (TP + FN)   при нулём знаменателе → 0
F1 = 2·TP / (2·TP + FP + FN)
  эталон и находки пусты          → F1 = 1
  эталон пуст, находки есть       → F1 = 0
награда = clip(2·F1 − 1 − 0.002 · inspection_steps, −1, 1)   // только терминальная
```

**Условие TP (все одновременно):** тип равен **точно** (без trim/синонимов);
множества `affected_ids` равны **точно** (лишний существующий ID убивает совпадение);
валюта равна **точно** (регистр и пробелы не склеиваются);
`|claimed − expected| <= amount_tolerance_minor` (τ по умолчанию 0);
`evidence_ids` содержит **полностью** хотя бы один допустимый набор из `evidence_sets`
и **только существующие** в открытых файлах ID (несуществующий → FP, не авария).
Назначение — **максимальное паросочетание** с детерминированным выбором
(эталоны по возрастанию `issue_id`, каждой находке — наименьший возможный индекс);
повтор по занятому эталону → FP.

## 7. Коды ошибок

| Код | Когда |
|---|---|
| `invalid_action` | действие не является объектом |
| `unknown_action` | имя действия вне списка |
| `unknown_table` | таблица вне списка |
| `unknown_id` / `unknown_document` | записи/документа нет |
| `invalid_cursor` / `invalid_limit` / `invalid_id` | нарушение ограничений полей |
| `limit_too_large` | `limit > 50` |
| `invalid_expression` | в калькулятор передано не арифметика |
| `division_by_zero` | деление на ноль |
| `time_limit` | истекли 60 с — действие не выполнено, `truncated=true` |
| `submit_invalid` | нарушена схема findings.json → `terminated=true`, reward = −1 |
| `episode_finished` | шаг после завершения: состояние и награда не меняются |

Сообщения ограничены и не содержат ни эталона, ни списка корректных ответов.

## 8. Стандартные конфигурации (оси)

```jsonc
{ "n_invoices": 60,                    // размер: ~4.4 строки на счёт
                                       // рекомендуемый диапазон 65..150: гарантирует
                                       // >=200 строк при ЛЮБОМ seed; без distractors >=70
  "base_currency": "USD",
  "currencies": ["USD", "EUR", "KZT"],
  "utc_offset": "+05:00",              // поддерживается и смещение в минутах
  "period_start": "2026-01-01", "period_end": "2026-03-31",
  "issue_types": ["duplicate_entry"],  // подмножество из 4 типов
  "n_issues": 1,                       // 0..12
  "scenarios": ["S1"],                 // [] | ["S1"] | ["S2"] | ["S1","S2"]
  "partial_variant": "partial",        // "partial" — недостаёт часть;
                                       // "miss" — распределение пропущено полностью
                                       //   (определения D09-D11: «пропуск» и «частичный»)
  "clean": false,                      // true → эталон пуст
  "distractors": true }                // возвраты, разбитые платежи, кросс-платежи,
                                       // повторяющиеся суммы, границы периода
```

**Раскладка для определений D01-D20** (жюри подбирает конфиги по этим осям):

| Эпизоды | Конфигурация |
|---|---|
| D01 (обычная сверка) | `{"clean": true, "distractors": false}` |
| D02-D05 (разбитый платёж / возврат / повторяющиеся суммы / граница) | `{"clean": true, "distractors": true}` + разные seed: все четыре признака присутствуют в данных |
| D06-D08 (дубль счёта) | `{"issue_types": ["duplicate_entry"], "n_issues": 1}` |
| D09-D11 (пропуск/частичное распределение) | `{"issue_types": ["partial_allocation"], "n_issues": 1, "partial_variant": "miss"}` и `"partial_variant": "partial"` |
| D12-D13 (дата/курс) | `{"issue_types": ["fx_rate"], "n_issues": 1}` |
| D14-D15 (период) | `{"issue_types": ["period_cutoff"], "n_issues": 1}` |
| D16-D18 (сценарий 1) | `{"scenarios": ["S1"]}` (+ нужные типы) |
| D19-D20 (сценарий 2) | `{"scenarios": ["S2"]}` (+ нужные типы) |
