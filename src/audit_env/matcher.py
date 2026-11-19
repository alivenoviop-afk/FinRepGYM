"""Shared matcher: normalisation, one-to-one matching, P/R/F1, submit validation.

Everything here is published with the submission (spec section 05: «Опубликуйте код
нормализации, допусков и выбора при неоднозначном совпадении»).

Matching rules
--------------
A submission finding is a true positive only if ALL hold:

* ``type`` equals the issue type exactly;
* canonical ``affected_ids`` (set of strings, sorted) equal the canonical issue ids;
* ``currency`` equals the issue currency after upper-casing;
* ``|claimed_amount_minor - expected_amount_minor| <= amount_tolerance_minor``;
* claimed ``evidence_ids`` contain at least one *complete* allowed evidence set
  (set containment: ``set(allowed) <= set(claimed)``).

Greedy assignment in submission order: each finding takes the first not-yet-taken
issue that passes every check. Findings that match nothing (including a second
finding for an already taken issue) are FP. Issues nobody claimed are FN.

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
def canonical_ids(values: Iterable[Any] | None) -> list[str]:
    """Canonical affected/evidence ids: strings, de-duplicated, sorted."""
    if not values:
        return []
    return sorted({str(v) for v in values})


def normalize_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """Canonical form of a submitted finding (idempotent)."""
    amount = finding.get("claimed_amount_minor")
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise ValueError("claimed_amount_minor must be an integer number of minor units")
    return {
        "type": str(finding.get("type", "")).strip(),
        "affected_ids": canonical_ids(finding.get("affected_ids")),
        "claimed_amount_minor": amount,
        "currency": str(finding.get("currency", "")).strip().upper(),
        "evidence_ids": canonical_ids(finding.get("evidence_ids")),
    }


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Canonical form of a ground-truth issue."""
    return {
        "issue_id": str(issue["issue_id"]),
        "type": str(issue["type"]).strip(),
        "affected_ids": canonical_ids(issue.get("affected_ids")),
        "expected_amount_minor": int(issue["expected_amount_minor"]),
        "currency": str(issue["currency"]).strip().upper(),
        "amount_tolerance_minor": int(issue.get("amount_tolerance_minor", DEFAULT_TOLERANCE_MINOR)),
        "evidence_sets": [canonical_ids(s) for s in issue.get("evidence_sets", [])],
    }


def _satisfies(finding: dict[str, Any], issue: dict[str, Any]) -> bool:
    if finding["type"] != issue["type"]:
        return False
    if finding["affected_ids"] != issue["affected_ids"]:
        return False
    if finding["currency"] != issue["currency"]:
        return False
    if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > issue["amount_tolerance_minor"]:
        return False
    claimed = set(finding["evidence_ids"])
    return any(set(allowed) <= claimed for allowed in issue["evidence_sets"])


# --------------------------------------------------------------------------- #
# validation of the submit payload
# --------------------------------------------------------------------------- #
def validate_submission(payload: Any, expected_episode_id: str) -> tuple[bool, str | None, dict | None]:
    """Return (ok, error_code, normalised_payload).

    On failure ``error_code`` is a short, non-leaking string for the evaluator log.
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
        if len(raw.get("evidence_ids") or []) > MAX_EVIDENCE_IDS:
            return False, f"finding_{i}_too_many_evidence_ids", None
        explanation = raw.get("explanation")
        if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION_CHARS:
            return False, f"finding_{i}_bad_explanation", None
        for field in ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids"):
            if field not in raw:
                return False, f"finding_{i}_missing_{field}", None
        try:
            normalised.append(normalize_finding(raw))
        except ValueError:
            return False, f"finding_{i}_bad_amount", None
    return True, None, {"episode_id": expected_episode_id, "findings": normalised}


# --------------------------------------------------------------------------- #
# matching + metrics
# --------------------------------------------------------------------------- #
def match(truth: Sequence[dict], findings: Sequence[dict]) -> dict[str, Any]:
    """One-to-one greedy matching. Returns index lists and the raw counts."""
    issues = [normalize_issue(t) for t in truth]
    subs = [normalize_finding(f) for f in findings]

    taken: set[int] = set()
    tp_pairs: list[tuple[int, int]] = []
    fp_indices: list[int] = []

    for fi, finding in enumerate(subs):
        matched_at = None
        for ii, issue in enumerate(issues):
            if ii in taken:
                continue
            if _satisfies(finding, issue):
                matched_at = ii
                break
        if matched_at is None:
            fp_indices.append(fi)
        else:
            taken.add(matched_at)
            tp_pairs.append((fi, matched_at))

    fn_indices = [i for i in range(len(issues)) if i not in taken]
    return {
        "tp": len(tp_pairs),
        "fp": len(fp_indices),
        "fn": len(fn_indices),
        "tp_pairs": tp_pairs,
        "fp_finding_indices": fp_indices,
        "fn_issue_indices": fn_indices,
    }


def compute_metrics(truth: Sequence[dict], findings: Sequence[dict]) -> dict[str, Any]:
    """P, R, F1 with every edge case from the spec."""
    issues = [normalize_issue(t) for t in truth]
    m = match(truth, findings)
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
