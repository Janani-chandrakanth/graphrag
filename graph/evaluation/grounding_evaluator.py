# graph/evaluation/grounding_evaluator.py
"""Test Case Grounding and Unsupported Step Evaluator.
Evaluates whether generated test cases are supported by the Neo4j Knowledge Graph.
A step is only flagged as unsupported when an explicit entity reference in the step
cannot be traced back to any node name, id, or description in the graph.
"""

import logging
import re
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

# Universal UI/system action words that are always valid and should never be flagged
ALWAYS_VALID_WORDS = {
    "user", "system", "button", "click", "enter", "select", "page", "screen",
    "display", "dashboard", "input", "submit", "valid", "invalid", "error",
    "message", "success", "field", "form", "view", "verify", "navigate",
    "login", "logout", "click", "tap", "open", "close", "check", "confirm",
    "search", "type", "upload", "download", "scroll", "the", "an", "a",
    "should", "can", "must", "then", "when", "with", "from", "into", "and",
    "test", "step", "action", "expected", "result", "scenario", "case",
    "attempt", "response", "request", "complete", "process", "provide"
}


def _normalize(s: str) -> str:
    if not s:
        return ""
    return re.sub(r'\s+', ' ', str(s).lower().strip())


def _build_graph_vocabulary(graph_nodes: List[Dict[str, Any]]) -> Set[str]:
    """Builds a comprehensive set of grounded terms from graph nodes."""
    vocab = set()
    for n in graph_nodes:
        raw_fields = [
            str(n.get("id") or ""),
            str(n.get("name") or ""),
            str(n.get("description") or ""),
            str(n.get("type") or ""),
            str(n.get("label") or "")
        ]
        aliases = n.get("aliases", [])
        if isinstance(aliases, list):
            raw_fields += [str(a) for a in aliases]

        for field in raw_fields:
            norm = _normalize(field)
            if len(norm) >= 3:
                vocab.add(norm)
            # Also add individual tokens from compound names (add_to_cart -> add, cart)
            tokens = re.split(r'[_\-\s]+', norm)
            for t in tokens:
                if len(t) >= 3 and t not in ALWAYS_VALID_WORDS:
                    vocab.add(t)

    return vocab


def _extract_domain_entities(step_text: str) -> List[str]:
    """
    Extracts meaningful domain entity references from a test step.
    Only returns multi-word phrases or capitalized nouns that look like
    domain-specific entity references (not generic action verbs).
    """
    entities = []

    # Match capitalized multi-token phrases e.g. "Shopping Cart", "Delivery Slot"
    phrases = re.findall(r'\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+)\b', step_text)
    for p in phrases:
        norm = _normalize(p)
        tokens = norm.split()
        if all(t not in ALWAYS_VALID_WORDS for t in tokens) and len(norm) >= 4:
            entities.append(norm)

    # Match CamelCase compound names e.g. ShoppingCart, FreshCartPortal
    camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', step_text)
    for c in camel:
        norm = _normalize(c)
        if norm not in ALWAYS_VALID_WORDS and len(norm) >= 5:
            entities.append(norm)

    # Match snake_case identifiers e.g. add_to_cart
    snake = re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', step_text)
    for s in snake:
        if _normalize(s) not in ALWAYS_VALID_WORDS and len(s) >= 5:
            entities.append(_normalize(s))

    return entities


def _is_entity_grounded(entity: str, vocab: Set[str]) -> bool:
    """Check whether an entity reference is supported by the graph vocabulary."""
    entity_norm = _normalize(entity)

    # Direct full-string match
    if entity_norm in vocab:
        return True

    # Substring match (entity is part of a graph term or vice versa)
    for term in vocab:
        if len(term) >= 3 and (entity_norm in term or term in entity_norm):
            return True

    # Token-level overlap: split entity into tokens, check how many appear in vocab
    entity_tokens = set(re.split(r'[_\-\s]+', entity_norm))
    meaningful = {t for t in entity_tokens if len(t) >= 3 and t not in ALWAYS_VALID_WORDS}

    if not meaningful:
        return True  # If no meaningful tokens, don't penalize

    overlap = sum(1 for t in meaningful if any(t in v or v in t for v in vocab))
    overlap_ratio = overlap / max(len(meaningful), 1)
    return overlap_ratio >= 0.5


def evaluate_test_case_grounding(
    test_cases: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluates test case grounding and identifies unsupported/hallucinated steps."""
    vocab = _build_graph_vocabulary(graph_nodes)

    unsupported_steps = []
    supported_test_cases = 0
    total_test_cases = len(test_cases)

    for tc in test_cases:
        tc_id = str(tc.get("id") or tc.get("tc_id") or tc.get("name") or "TC-UNKNOWN")
        tc_title = str(tc.get("title") or tc.get("name") or "Test Case")
        steps = tc.get("steps", [])
        has_unsupported_step = False

        for step in steps:
            step_str = str(step)
            # Extract only genuine domain entity references from this step
            entities = _extract_domain_entities(step_str)

            for entity in entities:
                if not _is_entity_grounded(entity, vocab):
                    has_unsupported_step = True
                    unsupported_steps.append({
                        "test_case_id": tc_id,
                        "title": tc_title,
                        "unsupported_step": step_str,
                        "unsupported_entity": entity,
                        "reason": (
                            f"Entity reference '{entity}' in this test step "
                            f"has no corresponding node, name, or alias in the Knowledge Graph."
                        )
                    })
                    break  # One report per step

            if has_unsupported_step:
                break  # One unsupported step report per test case

        if not has_unsupported_step:
            supported_test_cases += 1

    test_case_correctness_pct = round(
        (supported_test_cases / max(total_test_cases, 1)) * 100, 1
    )

    return {
        "test_case_correctness_pct": test_case_correctness_pct,
        "total_test_cases": total_test_cases,
        "supported_test_cases": supported_test_cases,
        "unsupported_step_count": len(unsupported_steps),
        "unsupported_test_steps": unsupported_steps
    }
