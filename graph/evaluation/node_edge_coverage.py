# graph/evaluation/node_edge_coverage.py
"""Deterministic Node and Relationship Coverage Evaluator for Test Cases.
Inspects Neo4j nodes and relationships against generated test cases to determine
node coverage %, workflow edge coverage %, and itemized missed graph elements.
Uses multi-token substring matching and token-overlap scoring for robust coverage detection.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set

from graph.evaluation.evaluation_contract import build_evaluation_contract
from graph.flow_graph_analysis import FLOW_RELATIONS

logger = logging.getLogger(__name__)

# Stop words that are too short or generic to count as meaningful matches
STOP_WORDS = {
    "the", "and", "for", "are", "not", "has", "with", "user",
    "that", "this", "from", "but", "can", "all", "was", "they",
    "when", "have", "will", "been", "each", "than", "then"
}


def _normalize(s: str) -> str:
    if not s:
        return ""
    # Strip quotes, escapes, and extra whitespace
    cleaned = str(s).replace('"', '').replace('\\', '').replace("'", "").strip()
    return re.sub(r'\s+', ' ', cleaned.lower())


def _tokenize(s: str) -> Set[str]:
    """Tokenize string into meaningful words (no stop words, length >= 3)."""
    words = re.findall(r'[a-z0-9]+', _normalize(s))
    # Filter out generic wrappers from token matching
    noise = STOP_WORDS | {"message", "feature", "portal", "system", "app"}
    return {w for w in words if len(w) >= 3 and w not in noise}


def _node_matched_in_text(node_name: str, node_id: str, full_text: str) -> bool:
    """Check whether a node is referenced in test case text using multiple flexible strategies."""
    full_text_norm = _normalize(full_text)
    name_norm = _normalize(node_name)
    id_norm = _normalize(node_id)

    if not full_text_norm:
        return False

    # Strategy 1: direct substring match of full node name / ID in text
    if len(name_norm) >= 3 and name_norm in full_text_norm:
        return True
    if len(id_norm) >= 3 and id_norm in full_text_norm:
        return True

    # Strategy 2: underscore/hyphen/camelCase split match
    # Clean up prefixes like 'feature_' or suffixes like ' message'
    clean_name = re.sub(r'^(feature_|system_|app_)', '', name_norm)
    clean_name = re.sub(r'(\s+message|\s+portal|\s+app)$', '', clean_name)
    
    clean_id = re.sub(r'^(feature_|system_|app_)', '', id_norm)
    clean_id = re.sub(r'(_message|_portal|_app)$', '', clean_id)

    if len(clean_name) >= 3 and clean_name in full_text_norm:
        return True
    if len(clean_id) >= 3 and clean_id in full_text_norm:
        return True

    name_tokens = set(re.split(r'[_\-\s]+', clean_name))
    id_tokens = set(re.split(r'[_\-\s]+', clean_id))
    all_tokens = (name_tokens | id_tokens) - STOP_WORDS - {"message", "feature", "portal", "system"}
    meaningful_tokens = {t for t in all_tokens if len(t) >= 3}

    if not meaningful_tokens:
        return False

    # Strategy 3: token overlap — if >= 40% of node tokens appear in text
    text_tokens = _tokenize(full_text)
    overlap = meaningful_tokens & text_tokens
    overlap_ratio = len(overlap) / max(len(meaningful_tokens), 1)

    if overlap_ratio >= 0.4:
        return True

    # Strategy 4: stem / core keyword match (e.g. 'login', 'delete', 'filter', 'archive', 'approve', 'audit')
    # If any core root verb/noun token (length >= 4) matches in full text
    core_words = {t for t in meaningful_tokens if len(t) >= 4 and t not in {"feature", "process", "functionality", "table", "page"}}
    if core_words:
        for cw in core_words:
            # Check stem prefix match e.g. "delete" in "deletion", "login" in "login_page", "filter" in "filters"
            stem = cw[:4]
            if any(stem in tt for tt in text_tokens) or stem in full_text_norm:
                return True

    return False


def _build_test_case_corpus(test_cases: List[Dict[str, Any]]) -> str:
    """Combines all test case text into a single searchable corpus string."""
    parts = []
    for tc in test_cases:
        nodes_used = tc.get("entities_used") or tc.get("graph_nodes") or tc.get("entities") or []
        for n in nodes_used:
            if isinstance(n, str):
                parts.append(n)
            elif isinstance(n, dict):
                parts.append(str(n.get("id") or n.get("name") or ""))

        parts.append(str(tc.get("title") or tc.get("name") or ""))
        parts.append(str(tc.get("precondition") or tc.get("preconditions") or ""))
        parts.append(str(tc.get("expected_result") or tc.get("expected_results") or ""))
        parts.extend([str(s) for s in tc.get("steps", [])])
        parts.append(str(tc.get("objective") or ""))

    return " ".join(parts)


def evaluate_node_edge_coverage(
    test_cases: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    graph_rels: List[Dict[str, Any]],
    feature_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes coverage across multiple dimensions using the Evaluation Contract.
    Separate coverage universes: Core Workflow, Transitions, Requirements, Business Rules, Inputs.
    Full Graph coverage is informational only.
    """
    # 1. Build the semantic contract (cached/fast if already run)
    # We use use_llm=False here to ensure deterministic, fast fallback if needed,
    # but normally the classification is already cached.
    contract = build_evaluation_contract(graph_nodes, graph_rels, use_llm=False)
    
    # 2. Build full-corpus string from all test cases
    tc_corpus = _build_test_case_corpus(test_cases)
    
    def evaluate_node_list(node_list: List[Dict]) -> Dict:
        covered = []
        missed = []
        for node in node_list:
            n_id = str(node.get("id") or "")
            n_name = str(node.get("name") or n_id)
            if _node_matched_in_text(n_name, n_id, tc_corpus):
                covered.append(node)
            else:
                missed.append(node)
        pct = round((len(covered) / max(len(node_list), 1)) * 100, 1) if node_list else 100.0
        return {"total": len(node_list), "covered": len(covered), "missed": len(missed), "pct": pct, "missed_items": missed}

    # 3. Core Workflow Coverage (only workflow_step and workflow_state nodes)
    workflow_cov = evaluate_node_list(contract.get("workflow_nodes", []))
    
    # 4. Requirement Coverage
    req_cov = evaluate_node_list(contract.get("requirements", []))
    
    # 5. Business Rule Coverage
    br_cov = evaluate_node_list(contract.get("business_rules", []))
    
    # 6. Input Coverage
    input_cov = evaluate_node_list(contract.get("inputs", []))

    # 7. Informational: Full Graph Context Coverage (all source nodes except artifacts)
    full_graph_cov = evaluate_node_list(contract.get("all_source_nodes", graph_nodes))

    # Issue 4: Procedural Node Coverage — only nodes that appear as endpoints
    # of at least one FLOW_RELATIONS edge. This is the coverage metric most
    # directly aligned with sequential-flow test adequacy.
    flow_node_ids: Set[str] = set()
    for rel in graph_rels:
        if rel.get("type") in FLOW_RELATIONS:
            src = rel.get("from") or rel.get("source") or ""
            tgt = rel.get("to") or rel.get("target") or ""
            if src:
                flow_node_ids.add(str(src))
            if tgt:
                flow_node_ids.add(str(tgt))
    procedural_nodes = [n for n in graph_nodes if str(n.get("id", "")) in flow_node_ids]
    procedural_cov = evaluate_node_list(procedural_nodes)

    # 8. Workflow Transition Coverage (only workflow_transition edges)
    workflow_transitions = contract.get("workflow_transitions", [])
    covered_edges = []
    missed_edges = []
    
    node_map = {str(n.get("id") or n.get("name")): n.get("name") or n.get("id") for n in graph_nodes}
    
    for rel in workflow_transitions:
        src_id = rel.get("from") or rel.get("source") or rel.get("start") or ""
        tgt_id = rel.get("to") or rel.get("target") or rel.get("end") or ""
        src_name = node_map.get(str(src_id), str(src_id))
        tgt_name = node_map.get(str(tgt_id), str(tgt_id))
        
        src_covered = _node_matched_in_text(src_name, str(src_id), tc_corpus)
        tgt_covered = _node_matched_in_text(tgt_name, str(tgt_id), tc_corpus)
        
        rel_formatted = {
            "source_id": src_id,
            "source_name": src_name,
            "type": rel.get("type", ""),
            "target_id": tgt_id,
            "target_name": tgt_name
        }
        
        if src_covered and tgt_covered:
            covered_edges.append(rel_formatted)
        else:
            missed_edges.append(rel_formatted)
            
    edge_pct = round((len(covered_edges) / max(len(workflow_transitions), 1)) * 100, 1) if workflow_transitions else 100.0
    
    # Pack legacy fields for compatibility with test_case_evaluator.py while adding new ones
    return {
        # Core Workflow
        "workflow_node_coverage_pct": workflow_cov["pct"],
        "workflow_node_total": workflow_cov["total"],
        "workflow_node_covered": workflow_cov["covered"],
        "workflow_node_missed": workflow_cov["missed_items"],

        "edge_coverage_pct": edge_pct,
        "total_workflow_edges": len(workflow_transitions),
        "covered_edge_count": len(covered_edges),
        "missed_edge_count": len(missed_edges),
        "covered_edges": covered_edges,
        "missed_edges": missed_edges,

        # Dimensions
        "requirement_coverage_pct": req_cov["pct"],
        "business_rule_coverage_pct": br_cov["pct"],
        "input_coverage_pct": input_cov["pct"],

        # Issue 4: Procedural (FLOW edges) vs Full Graph — reported separately
        "procedural_node_coverage_pct": procedural_cov["pct"],
        "procedural_node_total": procedural_cov["total"],
        "procedural_node_covered": procedural_cov["covered"],
        "procedural_node_missed_items": procedural_cov["missed_items"],

        # Legacy/Informational
        "node_coverage_pct": full_graph_cov["pct"],  # Full Graph Context Coverage
        "total_nodes": full_graph_cov["total"],
        "covered_node_count": full_graph_cov["covered"],
        "missed_node_count": full_graph_cov["missed"],
        "missed_nodes": full_graph_cov["missed_items"],

        "contract": contract
    }
