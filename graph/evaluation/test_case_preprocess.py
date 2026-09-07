# graph/evaluation/test_case_preprocess.py
"""Preprocess test cases before hybrid evaluation.

Normalizes field aliases, strips empty records, and enriches each test case
with a flat 'full_text' corpus string that all downstream evaluators can use.
"""

import re
from typing import List, Dict, Any


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
        if not isinstance(tc, dict):
            continue
        # Shallow copy so we don't mutate caller's data
        tc2 = dict(tc)

        # Normalize common id/title aliases
        if not tc2.get("id"):
            tc2["id"] = tc2.get("tc_id") or tc2.get("case_id") or ""
        if not tc2.get("title"):
            tc2["title"] = tc2.get("name") or tc2.get("description") or tc2["id"]

        # Build flat corpus string used by LLM bridge and metric aggregator
        tc2["_full_text"] = _extract_full_text(tc2)

        result.append(tc2)
    return result
