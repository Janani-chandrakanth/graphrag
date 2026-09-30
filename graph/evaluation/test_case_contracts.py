"""
graph/evaluation/test_case_contracts.py

Consolidated test case evaluation contracts, preprocessing, and deduplication:
- Semantic evaluation contract derivation (formerly evaluation_contract.py)
- Test case field normalization and text enrichment (formerly test_case_preprocess.py)
- Deterministic duplicate detection & scenario enrichment (formerly test_case_deduplicator.py)
"""

import logging
import re
from typing import Dict, List, Any, Tuple

from graph.semantic_classifier import classify_graph
from graph.feature_extractor import extract_features

logger = logging.getLogger(__name__)


# ── Semantic Evaluation Contract ──────────────────────────────────────────

def build_evaluation_contract(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Build the semantic evaluation contract directly from the source BRD and Knowledge Graph.
    Serves as the ground-truth denominator for all test coverage metrics.
    """
    # 1. Classify graph to get the true source coverage universe
    classification = classify_graph(nodes, relationships, use_llm=use_llm)
    universe = classification["coverage_universe"]

    # 2. Extract features to map scenarios
    extraction = extract_features(nodes, relationships, classification, use_llm=use_llm)
    features = extraction.get("features", [])

    # 3. Build scenario applicability for each feature
    scenario_applicability = {}
    for feature in features:
        fname = feature.get("name", "")
        if not fname:
            continue
            
        # Analyze the feature to determine valid scenarios
        has_branches = len(feature.get("branches", [])) > 0
        has_negative = len(feature.get("negative_paths", [])) > 0
        has_alternate = len(feature.get("alternate_paths", [])) > 0
        has_decision = len(feature.get("decision_node_ids", [])) > 0
        
        # Determine failure/retry logic
        has_failure_retry = False
        for end_state in feature.get("end_state_ids", []):
            end_name = end_state.lower()
            if any(term in end_name for term in ["error", "fail", "reject", "invalid"]):
                has_failure_retry = True
        if has_negative:
            has_failure_retry = True

        scenario_applicability[fname] = {
            "positive": True,  # Every feature has at least a happy path
            "negative": has_negative or has_failure_retry or has_branches,
            "edge": has_decision,  # Decision boundaries often lead to edge cases
            "alternate": has_alternate or has_branches,
            "failure_retry": has_failure_retry
        }

    return {
        "features": features,
        "workflow_nodes": universe.get("workflow_nodes", []),
        "workflow_transitions": universe.get("workflow_transitions", []),
        "requirements": universe.get("requirement_nodes", []),
        "business_rules": universe.get("business_rule_nodes", []),
        "inputs": universe.get("input_nodes", []),
        "scenario_applicability": scenario_applicability
    }


# ── Test Case Preprocessing & Normalization ───────────────────────────────

def _extract_full_text(tc: Dict[str, Any]) -> str:
    """Build a single lowercase corpus string from all textual fields of a test case."""
    parts = [
        str(tc.get("title") or tc.get("name") or ""),
        str(tc.get("objective") or ""),
        str(tc.get("precondition") or tc.get("preconditions") or ""),
        str(tc.get("expected_result") or tc.get("expected_results") or ""),
        str(tc.get("scenario_type") or tc.get("type") or tc.get("category") or ""),
    ]
    for step in tc.get("steps", []):
        parts.append(str(step))
    for ent in tc.get("target_entities", tc.get("entities_used", tc.get("entities", []))):
        if isinstance(ent, str):
            parts.append(ent)
        elif isinstance(ent, dict):
            parts.append(str(ent.get("name") or ent.get("id") or ""))
    return " ".join(parts).lower()


def preprocess_test_cases(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize and enrich test cases. Returns a new list; originals are unchanged."""
    result = []
    for tc in test_cases:
        if hasattr(tc, "items"):
            tc2 = dict(tc.items())
        elif isinstance(tc, dict):
            tc2 = dict(tc)
        else:
            continue

        # Normalize common id/title aliases
        if not tc2.get("id"):
            tc2["id"] = tc2.get("tc_id") or tc2.get("case_id") or ""
        if not tc2.get("title"):
            tc2["title"] = tc2.get("name") or tc2.get("description") or tc2["id"]

        # Build flat corpus string used by LLM bridge and metric aggregator
        tc2["_full_text"] = _extract_full_text(tc2)

        result.append(tc2)
    return result


# ── Test Case Deduplication & Uniqueness ──────────────────────────────────

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
