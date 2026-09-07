# graph/evaluation/test_case_deduplicator.py
"""Test Case Deduplication and Scenario Completion Engine.
Deterministically identifies duplicate and redundant test cases, enforces complete scenario details,
and computes uniqueness metrics.
"""

import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    if not s:
        return ""
    return "".join(c.lower() for c in str(s) if c.isalnum())


def _build_test_case_signature(tc: Dict[str, Any]) -> str:
    """Builds a normalized signature string to detect duplicate test cases."""
    title_norm = _normalize(tc.get("title") or tc.get("name") or "")
    type_norm = _normalize(tc.get("type") or "positive")
    
    steps = tc.get("steps", [])
    if isinstance(steps, list):
        steps_norm = "->".join([_normalize(str(s)) for s in steps])
    else:
        steps_norm = _normalize(str(steps))

    # Include precondition if present
    precond_norm = _normalize(str(tc.get("precondition") or tc.get("preconditions") or ""))

    return f"{type_norm}::{title_norm}::{precond_norm}::{steps_norm}"


def _enrich_scenario_details(tc: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Ensures test case has complete fields for test scenario display."""
    tc_id = str(tc.get("id") or tc.get("tc_id") or tc.get("name") or f"TC-{index:03d}")
    raw_title = str(tc.get("title") or tc.get("name") or f"Test Scenario {index}")
    raw_type = str(tc.get("type") or "Positive").capitalize()

    # Determine scenario category
    type_lower = raw_type.lower()
    text_blob = " ".join([raw_title, str(tc.get("precondition") or ""), str(tc.get("expected_result") or "")]).lower()

    if "negative" in type_lower or any(w in text_blob for w in ["invalid", "fail", "error", "denied"]):
        scenario_type = "Negative"
    elif "edge" in type_lower or any(w in text_blob for w in ["limit", "max", "min", "boundary", "lock", "timeout"]):
        scenario_type = "Edge Cases"
    elif any(w in text_blob for w in ["retry", "recover", "reattempt", "loop"]):
        scenario_type = "Failure / Retry"
    elif any(w in text_blob for w in ["alternate", "option", "branch"]):
        scenario_type = "Alternate Paths"
    else:
        scenario_type = "Positive"

    precond = str(tc.get("precondition") or tc.get("preconditions") or "Valid system state and user session established.")
    steps = tc.get("steps", [])
    if not isinstance(steps, list) or not steps:
        steps = ["Navigate to target feature", "Execute action", "Verify response"]

    expected = str(tc.get("expected_result") or tc.get("expected_results") or "System processes action and displays expected response.")
    entities = tc.get("entities_used") or tc.get("graph_nodes") or tc.get("entities") or []

    return {
        "id": tc_id,
        "title": raw_title,
        "scenario_type": scenario_type,
        "objective": f"Validate {raw_title.lower()} path against knowledge graph requirements.",
        "preconditions": precond,
        "steps": [str(s) for s in steps],
        "expected_result": expected,
        "target_entities": entities
    }


def deduplicate_test_cases(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministically identifies duplicate test cases and returns uniqueness metrics."""
    if not test_cases:
        return {
            "total_test_cases": 0,
            "unique_test_cases_count": 0,
            "duplicate_test_cases_count": 0,
            "duplication_ratio": 0.0,
            "unique_test_cases": [],
            "duplicate_test_cases": []
        }

    seen_signatures = {}
    unique_tcs = []
    duplicate_tcs = []

    for idx, raw_tc in enumerate(test_cases, start=1):
        enriched = _enrich_scenario_details(raw_tc, idx)
        sig = _build_test_case_signature(enriched)

        if sig in seen_signatures:
            original_id = seen_signatures[sig]["id"]
            duplicate_info = dict(enriched)
            duplicate_info["duplicate_of_id"] = original_id
            duplicate_info["reason"] = f"Identical step sequence and scenario targets as test case '{original_id}'."
            duplicate_tcs.append(duplicate_info)
        else:
            seen_signatures[sig] = enriched
            unique_tcs.append(enriched)

    total_count = len(test_cases)
    unique_count = len(unique_tcs)
    dup_count = len(duplicate_tcs)
    dup_ratio = round((dup_count / max(total_count, 1)) * 100, 1)

    return {
        "total_test_cases": total_count,
        "unique_test_cases_count": unique_count,
        "duplicate_test_cases_count": dup_count,
        "duplication_ratio": dup_ratio,
        "unique_test_cases": unique_tcs,
        "duplicate_test_cases": duplicate_tcs
    }
