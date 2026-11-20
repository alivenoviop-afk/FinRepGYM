"""Shared matcher: normalisation, maximum matching, P/R/F1, submit validation.

Everything here is published with the submission (spec section 05: «Опубликуйте код
нормализации, допусков и выбора при неоднозначном совпадении»).

Matching rules (определения проверок от 20.11.2026, раздел «Точные соглашения matcher»)
-------------------------------------------------------------------------------
A submission finding is a true positive only if ALL hold:

* ``type`` equals the documented issue type exactly (no trimming, no synonyms);
* ``affected_ids`` as a SET equals the canonical set of the issue exactly
  (order and in-list repeats do not matter; an extra id kills the match);
* ``currency`` equals the issue currency EXACTLY (no case folding, no trimming);
* ``|claimed_amount_minor - expected_amount_minor| <= amount_tolerance_minor``
  (default tau = 0 when the field is absent);
* claimed ``evidence_ids`` form a set that (a) contains only ids that exist in the
  open episode data and (b) includes at least one COMPLETE allowed evidence set
  (``set(allowed) <= set(claimed)`` — extra existing evidence is allowed).

Assignment is a MAXIMUM bipartite matching (greedy first-fit can lose a TP because
of list order). Among maximum matchings the choice is deterministic: issues are
visited in lexicographic order of ``issue_id`` (Unicode code points) and each issue
takes the lowest input index of a finding that still allows a maximum matching.

Metric edge cases (spec section 05)
-----------------------------------
* denominator of P or R is zero -> 0
* truth empty AND findings empty -> F1 = 1
* truth empty AND findings present -> F1 = 0
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

DEFAULT_TOLERANCE_MINOR = 0

MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def canonical_ids(values: Any) -> list[str]:
    """Canonical id list: elements must be strings (spec: no number->string
    coercion), de-duplicated, sorted. Raises ValueError for structural errors."""
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("ids must be a list of strings")
    out: set[str] = set()
    for v in values:
        if not isinstance(v, str):
            raise ValueError("ids must be strings (no implicit coercion)")
        out.add(v)
    return sorted(out)


def normalize_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """Canonical form of a submitted finding (idempotent)."""
    amount = finding.get("claimed_amount_minor")
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise ValueError("claimed_amount_minor must be an integer number of minor units")
    currency = finding.get("currency")
    if not isinstance(currency, str):
        raise ValueError("currency must be a string")
    ftype = finding.get("type")
    if not isinstance(ftype, str):
        raise ValueError("type must be a string")
    return {
        # Точное совпадение: без strip/upper (определения 20.11, «currency», «type»)
        "type": ftype,
        "affected_ids": canonical_ids(finding.get("affected_ids")),
        "claimed_amount_minor": amount,
        "currency": currency,
        "evidence_ids": canonical_ids(finding.get("evidence_ids")),
    }


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Canonical form of a ground-truth issue."""
    return {
        "issue_id": str(issue["issue_id"]),
        "type": str(issue["type"]),
        "affected_ids": canonical_ids(issue.get("affected_ids")),
        "expected_amount_minor": int(issue["expected_amount_minor"]),
        "currency": str(issue["currency"]),
        "amount_tolerance_minor": int(issue.get("amount_tolerance_minor",
                                                 DEFAULT_TOLERANCE_MINOR)),
        "evidence_sets": [canonical_ids(s) for s in issue.get("evidence_sets", [])],
    }


def _satisfies(finding: dict[str, Any], issue: dict[str, Any],
               valid_evidence_ids: set[str] | None = None) -> bool:
    if finding["type"] != issue["type"]:
        return False
    if finding["affected_ids"] != issue["affected_ids"]:   # множества: точное равенство
        return False
    if finding["currency"] != issue["currency"]:           # точное совпадение, регистр важен
        return False
    if abs(finding["claimed_amount_minor"]
           - issue["expected_amount_minor"]) > issue["amount_tolerance_minor"]:
        return False
    claimed = set(finding["evidence_ids"])
    # M04: несуществующий/недоступный evidence_id -> вывод не сопоставляется (FP)
    if valid_evidence_ids is not None and not claimed <= valid_evidence_ids:
        return False
    return any(set(allowed) <= claimed for allowed in issue["evidence_sets"])


# --------------------------------------------------------------------------- #
# validation of the submit payload
# --------------------------------------------------------------------------- #
def validate_submission(payload: Any, expected_episode_id: str) -> tuple[bool, str | None, dict | None]:
    """Return (ok, error_code, normalised_payload).

    On failure ``error_code`` is a short, non-leaking string for the evaluator log.
    Лимиты проверяются ДО нормализации множеств (определения 20.11, «Лимиты»).
    """
    if not isinstance(payload, dict):
        return False, "submit_not_object", None
    if payload.get("episode_id") != expected_episode_id:
        return False, "episode_id_mismatch", None
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return False, "findings_not_list", None
    if len(findings) > MAX_FINDINGS:
        return False, "too_many_findings", None

    normalised = []
    for i, raw in enumerate(findings):
        if not isinstance(raw, dict):
            return False, f"finding_{i}_not_object", None
        evidence = raw.get("evidence_ids")
        if not isinstance(evidence, list):
            return False, f"finding_{i}_bad_evidence_ids", None
        if len(evidence) > MAX_EVIDENCE_IDS:
            return False, f"finding_{i}_too_many_evidence_ids", None
        affected = raw.get("affected_ids")
        if not isinstance(affected, list):
            return False, f"finding_{i}_bad_affected_ids", None
        explanation = raw.get("explanation")
        if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION_CHARS:
            # длина считается в Unicode code points (len() после декодирования JSON)
            return False, f"finding_{i}_bad_explanation", None
        for field in ("type", "affected_ids", "claimed_amount_minor", "currency",
                      "evidence_ids", "explanation"):
            if field not in raw:
                return False, f"finding_{i}_missing_{field}", None
        try:
            normalised.append(normalize_finding(raw))
        except ValueError:
            return False, f"finding_{i}_bad_structure", None
    return True, None, {"episode_id": expected_episode_id, "findings": normalised}


# --------------------------------------------------------------------------- #
# matching + metrics
# --------------------------------------------------------------------------- #
def _kuhn_max(adjacency: Sequence[Sequence[int]], n_slots: int) -> int:
    """Максимальное паросочетание (Кузн): adjacency[вывод] = слоты (индексы issues)."""
    match_slot = [-1] * n_slots

    def dfs(fi: int, seen: set[int]) -> bool:
        for slot in adjacency[fi]:
            if slot in seen:
                continue
            seen.add(slot)
            if match_slot[slot] == -1 or dfs(match_slot[slot], seen):
                match_slot[slot] = fi
                return True
        return False

    size = 0
    for fi in range(len(adjacency)):
        if dfs(fi, set()):
            size += 1
    return size


def match(truth: Sequence[dict], findings: Sequence[dict],
          valid_evidence_ids: set[str] | None = None) -> dict[str, Any]:
    """Maximum bipartite matching with the deterministic tie-break of the spec.

    * кардинальность — максимальная (жадный выбор первого совпадения запрещён);
    * при нескольких максимальных — лексикографически наименьшая последовательность
      пар ``(issue_id, индекс вывода)``: issues идут по возрастанию ``issue_id``,
      каждый берёт наименьший возможный индекс вывода.
    """
    issues = [normalize_issue(t) for t in truth]
    subs = [normalize_finding(f) for f in findings]

    adjacency: list[list[int]] = [
        [ii for ii, issue in enumerate(issues) if _satisfies(f, issue, valid_evidence_ids)]
        for f in subs
    ]
    total = _kuhn_max(adjacency, len(issues))

    # ---- детерминированный выбор среди максимальных паросочетаний ------------ #
    issue_order = sorted(range(len(issues)), key=lambda i: issues[i]["issue_id"])
    used: set[int] = set()
    assigned: dict[int, int] = {}          # issue index -> finding index
    decided: set[int] = set()              # issues, уже рассмотренные (взяты или пропущены)

    def max_over_remaining(skip_finding: int | None) -> int:
        rest_issues = [i for i in range(len(issues)) if i not in decided and i not in assigned]
        rest_findings = [fi for fi in range(len(subs))
                         if fi not in used and fi != skip_finding]
        if not rest_issues or not rest_findings:
            return 0
        slot = {ii: s for s, ii in enumerate(rest_issues)}
        sub_adj = [[slot[ii] for ii in adjacency[fi] if ii in slot]
                   for fi in rest_findings]
        return _kuhn_max(sub_adj, len(rest_issues))

    for ii in issue_order:
        decided.add(ii)
        chosen = None
        for fi in range(len(subs)):            # по возрастанию индекса вывода
            if fi in used or ii not in adjacency[fi]:
                continue
            # берём fi, только если общая кардинальность останется максимальной
            if len(assigned) + 1 + max_over_remaining(skip_finding=fi) >= total:
                chosen = fi
                break
        if chosen is not None:
            used.add(chosen)
            assigned[ii] = chosen

    tp_pairs = sorted((fi, ii) for ii, fi in assigned.items())
    fp_indices = [fi for fi in range(len(subs)) if fi not in used]
    fn_indices = [i for i in range(len(issues)) if i not in assigned]
    return {
        "tp": len(tp_pairs),
        "fp": len(fp_indices),
        "fn": len(fn_indices),
        "tp_pairs": tp_pairs,
        "fp_finding_indices": fp_indices,
        "fn_issue_indices": fn_indices,
    }


def compute_metrics(truth: Sequence[dict], findings: Sequence[dict],
                    valid_evidence_ids: set[str] | None = None) -> dict[str, Any]:
    """P, F1, recall with every edge case from the spec."""
    issues = [normalize_issue(t) for t in truth]
    m = match(truth, findings, valid_evidence_ids)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0

    if not truth and not findings:
        f1 = 1.0
    elif not truth and findings:
        f1 = 0.0
    else:
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_truth": len(truth),
        "n_findings": len(findings),
        "matched_truth_ids": sorted(issues[i]["issue_id"] for _, i in m["tp_pairs"]),
    }
