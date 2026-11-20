# Схемы audit-rl-env

> Явное описание контрактов (пункт сдачи: «Matcher и награда — общий код сопоставления
> и награды, **схемы** и политики»). Версия схемы: `audit-rl/1.0`.
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

`info` всегда содержит только: `{"mode": "train"|"eval", "steps_used": int, "schema_version": "audit-rl/1.0"}`.

**Флаги завершения:**

| Ситуация | terminated | truncated | reward (train) |
|---|---|---|---|
| `submit` принят | `true` | `false` | `clip(2·F1 − 1 − 0.002·шаги, −1, 1)` |
| `submit` отклонён (плохая структура) | `true` | `false` | `-1.0` |
| израсходовано 100 шагов | `false` | `true` | награда за **пустой** ответ |
| вышло время (60 с) — действие не выполняется | `false` | `true` | награда за **пустой** ответ |
| все промежуточные шаги | `false` | `false` | `0.0` |

В режиме `eval` политика получает `reward = 0.0` на всех шагах; истинная награда и
метрики доступны только через `evaluator_report()` (вне контракта).

## 2. Действия (5 штук, только чтение)

| Действие | Поля | Лимиты |
|---|---|---|
| `list_records` | `table`, `cursor` (int >= 0), `limit` (int) | `table` ∈ {ledger, invoices, payments, fx}; `1 <= limit <= 50` |
| `get_record` | `table`, `id` (непустая строка) | id должен существовать |
| `get_document` | `doc_id` (непустая строка) | doc_id ∈ invoices/payments |
| `calculator` | `expression` (строка) | только цифры, `.`, `+ - * /`, скобки; длина <= 200 |
| `submit` | `episode_id`, `findings` (см. §3) | `episode_id` = manifest.episode_id |

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

**Префиксы `evidence_ids`:** `ledger:<row_id>`, `invoice:<invoice_id>`,
`payment:<payment_id>`, `fx:<date>:<source>:<target>`, `policy:<rule>`.

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

**Условие TP (все одновременно):** тип равен; канонические `affected_ids` равны
(множество строк); валюта равна (верхний регистр); `|claimed − expected| <= amount_tolerance_minor`;
`evidence_ids` содержит **полностью** хотя бы один допустимый набор из `evidence_sets`.
Назначение взаимно однозначное, жадное в порядке подачи; повтор по занятому эталону → FP.

## 7. Коды ошибок

| Код | Когда |
|---|---|
| `unknown_action` | действие не из списка / не объект |
| `unknown_table` | таблица вне списка |
| `unknown_id` / `unknown_document` | записи/документа нет |
| `invalid_cursor` / `invalid_limit` / `invalid_id` | нарушение ограничений полей |
| `limit_too_large` | `limit > 50` |
| `invalid_expression` | в калькулятор передано не арифметика |
| `division_by_zero` | деление на ноль |
| `time_limit` | истекли 60 с — действие не выполнено, `truncated=true` |
| `submit_invalid` | нарушена схема findings.json → `terminated=true`, reward = −1 |

Сообщения ограничены и не содержат ни эталона, ни списка корректных ответов.

## 8. Стандартные конфигурации (оси)

```jsonc
{ "n_invoices": 60,                    // размер: ~4.4 строки на счёт
  "base_currency": "USD",
  "currencies": ["USD", "EUR", "KZT"],
  "utc_offset": "+05:00",              // поддерживается и смещение в минутах
  "period_start": "2026-01-01", "period_end": "2026-03-31",
  "issue_types": ["duplicate_entry"],  // подмножество из 4 типов
  "n_issues": 1,                       // 0..12
  "scenarios": ["S1"],                 // [] | ["S1"] | ["S2"] | ["S1","S2"]
  "clean": false,                      // true → эталон пуст
  "distractors": true }                // возвраты, разбитые платежи, кросс-платежи
```
