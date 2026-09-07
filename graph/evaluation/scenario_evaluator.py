# graph/evaluation/scenario_evaluator.py
"""Scenario Coverage Evaluator for Test Cases.
Derives expected scenario categories dynamically from graph/workflow structure
and evaluates coverage for Positive, Negative, Edge, Alternate, and Failure/Retry scenarios.
Classification reads from the tc 'type' / 'scenario_type' field first, then falls back to keyword signals.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

NEGATIVE_CUE_WORDS = {
    "invalid", "error", "fail", "failed", "failure", "incorrect", "denied",
    "reject", "rejected", "unauthorized", "expired", "missing", "forbidden",
    "exception", "wrong", "bad", "unavailable", "cannot", "blocked",
    "not found", "incorrect password", "wrong otp", "invalid payment"
}

EDGE_CUE_WORDS = {
    "limit", "maximum", "max", "minimum", "min", "boundary", "threshold",
    "lock", "locked", "timeout", "empty", "overflow", "exceed", "zero",
    "exactly", "upper", "lower", "extreme", "bulk", "large", "capacity"
}

FAILURE_RETRY_CUE_WORDS = {
    "retry", "recover", "recovery", "loop", "reattempt", "resubmit",
    "fallback", "re-attempt", "try again", "reconnect", "resume"
}

ALTERNATE_CUE_WORDS = {
    "alternate", "alternative", "branch", "option", "bypassed", "skip",
    "another", "different", "secondary", "additional path", "other way"
}

# Type field canonical values that map to categories
TYPE_CANONICAL = {
    "positive": "positive",
    "happy": "positive",
    "valid": "positive",
    "success": "positive",
    "negative": "negative",
    "invalid": "negative",
    "error": "negative",
    "failure": "negative",
    "edge": "edge",
    "boundary": "edge",
    "limit": "edge",
    "corner": "edge",
    "alternate": "alternate",
    "alternative": "alternate",
    "branch": "alternate",
    "retry": "failure_retry",
    "recovery": "failure_retry",
    "fault": "failure_retry",
}


def _classify_test_case(tc: Dict[str, Any]) -> str:
    """Classify a test case into positive/negative/edge/alternate/failure_retry."""
    # First: check explicit type/scenario_type/category fields exactly
    for field in ["type", "scenario_type", "category"]:
        raw = str(tc.get(field) or "").lower().strip()
        if raw in {"positive", "happy", "valid", "success"}:
            return "positive"
        if raw in {"negative", "invalid", "error", "failure"}:
            return "negative"
        if raw in {"edge", "boundary", "limit", "corner"}:
            return "edge"
        if raw in {"alternate", "alternative", "branch"}:
            return "alternate"
        if raw in {"failure_retry", "retry", "recovery"}:
            return "failure_retry"

    # Second: check title prefix (e.g. "Positive - Successful Login", "Negative - Invalid Email")
    title_lower = str(tc.get("title") or tc.get("name") or "").lower()
    if title_lower.startswith("positive") or "positive" in title_lower:
        return "positive"
    if title_lower.startswith("negative") or "negative" in title_lower:
        return "negative"
    if title_lower.startswith("edge") or "edge" in title_lower:
        return "edge"

    # Third: full-text keyword signal
    tc_text = " ".join([
        str(tc.get("title") or tc.get("name") or ""),
        str(tc.get("precondition") or tc.get("preconditions") or ""),
        str(tc.get("expected_result") or tc.get("expected_results") or ""),
        str(tc.get("objective") or ""),
        " ".join([str(s) for s in tc.get("steps", [])])
    ]).lower()

    words = set(re.findall(r'\b\w+\b', tc_text))
    bigrams = set()
    word_list = list(re.findall(r'\b\w+\b', tc_text))
    for i in range(len(word_list) - 1):
        bigrams.add(f"{word_list[i]} {word_list[i+1]}")
    all_signals = words | bigrams

    if all_signals.intersection(FAILURE_RETRY_CUE_WORDS):
        return "failure_retry"
    if all_signals.intersection(EDGE_CUE_WORDS):
        return "edge"
    if all_signals.intersection(ALTERNATE_CUE_WORDS):
        return "alternate"
    if all_signals.intersection(NEGATIVE_CUE_WORDS):
        return "negative"

    return "positive"


def evaluate_scenario_coverage(
    test_cases: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    graph_rels: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Dynamically derives expected scenarios from the evaluation contract and evaluates coverage per type."""
    from graph.evaluation.evaluation_contract import build_evaluation_contract
    
    # 1. Fetch scenario applicability from Semantic Contract
    contract = build_evaluation_contract(graph_nodes, graph_rels, use_llm=False)
    applicability = contract.get("scenario_applicability", {})
    
    # Determine global requirements based on feature applicability
    requires_negative = any(app.get("negative", False) for app in applicability.values())
    requires_edge = any(app.get("edge", False) for app in applicability.values())
    requires_alternate = any(app.get("alternate", False) for app in applicability.values())
    requires_retry = any(app.get("failure_retry", False) for app in applicability.values())

    total_tcs = len(test_cases)
    if total_tcs == 0:
        return {
            "expected_counts": {}, "covered_counts": {}, "per_tc_classification": [],
            "positive_pct": 0.0, "negative_pct": 0.0, "edge_pct": 0.0,
            "alternate_pct": 0.0, "failure_retry_pct": 0.0, "warnings": ["No test cases provided."]
        }

    # Dynamic target expectations
    expected_counts = {
        "positive": max(1, int(total_tcs * 0.40)),
        "negative": max(1, int(total_tcs * 0.30)) if requires_negative else 1,
        "edge": max(1, int(total_tcs * 0.15)) if requires_edge else 1,
        "alternate": max(1, int(total_tcs * 0.10)) if requires_alternate else 0,
        "failure_retry": max(1, int(total_tcs * 0.05)) if requires_retry else 0
    }

    # 2. Classify all test cases
    covered_counts = {
        "positive": 0,
        "negative": 0,
        "edge": 0,
        "alternate": 0,
        "failure_retry": 0
    }
    per_tc_classification = []

    for tc in test_cases:
        category = _classify_test_case(tc)
        covered_counts[category] = covered_counts.get(category, 0) + 1
        per_tc_classification.append({
            "id": tc.get("id") or tc.get("tc_id") or tc.get("name"),
            "title": tc.get("title") or tc.get("name"),
            "classified_as": category
        })

    # 3. Compute coverage percentages and generate warnings
    percentages = {}
    warnings = []
    
    for cat in ["positive", "negative", "edge", "alternate", "failure_retry"]:
        exp = expected_counts.get(cat, 0)
        cov = covered_counts.get(cat, 0)
        if exp > 0:
            percentages[cat] = round(min(1.0, cov / max(exp, 1)) * 100, 1)
            if cov == 0:
                warnings.append(f"Missing '{cat}' scenarios: Features in this graph semantically support '{cat}' paths, but 0 test cases cover them.")
        else:
            percentages[cat] = 100.0 if cov > 0 else 0.0

    return {
        "expected_counts": expected_counts,
        "covered_counts": covered_counts,
        "per_tc_classification": per_tc_classification,
        "positive_pct": percentages["positive"],
        "negative_pct": percentages["negative"],
        "edge_pct": percentages["edge"],
        "alternate_pct": percentages["alternate"],
        "failure_retry_pct": percentages["failure_retry"],
        "warnings": warnings
    }
