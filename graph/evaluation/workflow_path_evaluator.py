# graph/evaluation/workflow_path_evaluator.py
"""Workflow Sequence and Path Correctness Evaluator for Test Cases.
Evaluates workflow step coverage, step-to-step transitions, and deterministically
validates test case step sequences against Neo4j graph paths to detect Flow Violations.
Scopes flow validation strictly to the feature subgraph to prevent cross-feature jumps.
"""

import logging
from typing import List, Dict, Any, Optional

from graph.evaluation.evaluation_contract import build_evaluation_contract

logger = logging.getLogger(__name__)

def _normalize(s: str) -> str:
    if not s:
        return ""
    return "".join(c.lower() for c in str(s) if c.isalnum())


def _build_adjacency_map(graph_rels: List[Dict[str, Any]], graph_nodes: List[Dict[str, Any]]) -> Dict[str, set]:
    """Builds a directed adjacency graph from graph relationships."""
    node_id_map = {}
    for n in graph_nodes:
        n_id = str(n.get("id") or "")
        n_name = str(n.get("name") or n_id)
        if n_id:
            node_id_map[_normalize(n_id)] = n_id
        if n_name:
            node_id_map[_normalize(n_name)] = n_id

    adj = {}
    for r in graph_rels:
        src = str(r.get("source") or r.get("start") or r.get("from") or "")
        tgt = str(r.get("target") or r.get("end") or r.get("to") or "")
        
        src_norm = _normalize(src)
        tgt_norm = _normalize(tgt)

        src_key = node_id_map.get(src_norm, src_norm)
        tgt_key = node_id_map.get(tgt_norm, tgt_norm)

        if src_key not in adj:
            adj[src_key] = set()
        adj[src_key].add(tgt_key)

    return adj


def _is_path_reachable(start_node: str, target_node: str, adj: Dict[str, set], max_depth: int = 4) -> bool:
    """Checks if target_node is reachable from start_node within max_depth hops."""
    if not start_node or not target_node:
        return True
    if start_node == target_node:
        return True

    queue = [(start_node, 0)]
    visited = {start_node}

    while queue:
        curr, depth = queue.pop(0)
        if curr == target_node:
            return True
        if depth >= max_depth:
            continue

        for neighbor in adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return False


def _match_node_for_step(step_text: str, graph_nodes: List[Dict[str, Any]]) -> Optional[str]:
    """Matches a test step string to a node ID in the graph."""
    step_norm = _normalize(step_text)
    best_match = None
    best_len = 0

    for n in graph_nodes:
        n_id = str(n.get("id") or "")
        n_name = str(n.get("name") or n_id)
        
        norm_id = _normalize(n_id)
        norm_name = _normalize(n_name)

        if (norm_name and norm_name in step_norm) or (norm_id and norm_id in step_norm):
            if len(norm_name) > best_len:
                best_len = len(norm_name)
                best_match = n_id

    return best_match


def _assign_test_case_to_feature(tc: Dict[str, Any], features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Determine which feature this test case belongs to by finding highest overlap of entities used."""
    tc_nodes = set()
    for n in tc.get("entities_used") or tc.get("graph_nodes") or tc.get("entities") or []:
        if isinstance(n, str):
            tc_nodes.add(n)
        elif isinstance(n, dict):
            tc_nodes.add(str(n.get("id") or n.get("name") or ""))

    if not tc_nodes:
        return None

    best_feature = None
    max_overlap = 0

    for feat in features:
        feat_nodes = set(feat.get("step_node_ids", [])) | set(feat.get("supporting_node_ids", []))
        overlap = len(tc_nodes.intersection(feat_nodes))
        if overlap > max_overlap:
            max_overlap = overlap
            best_feature = feat

    return best_feature


def evaluate_workflow_paths(
    test_cases: List[Dict[str, Any]],
    workflow_model: Optional[Any],
    graph_nodes: List[Dict[str, Any]],
    graph_rels: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluates workflow coverage and deterministically validates test case flow correctness."""
    
    # 1. Fetch features and transitions from contract
    contract = build_evaluation_contract(graph_nodes, graph_rels, use_llm=False)
    features = contract.get("features", [])
    
    # We still build a global adjacency for fallback, but scoped adjacencies are better
    global_adj = _build_adjacency_map(graph_rels, graph_nodes)

    total_test_cases = len(test_cases)
    flow_violations = []
    valid_test_cases = 0

    # 2. Flow Correctness Evaluation for each test case
    for tc in test_cases:
        tc_id = str(tc.get("id") or tc.get("tc_id") or tc.get("name") or "TC-UNKNOWN")
        tc_title = str(tc.get("title") or tc.get("name") or "Test Case")
        steps = tc.get("steps", [])

        if not isinstance(steps, list) or len(steps) < 2:
            valid_test_cases += 1
            continue

        # Assign TC to a specific feature to prevent cross-feature jumping
        assigned_feature = _assign_test_case_to_feature(tc, features)
        
        adj_to_use = global_adj
        if assigned_feature:
            # Build feature-scoped adjacency
            feat_edges = assigned_feature.get("transitions", [])
            # Format feature edges to match expected shape for _build_adjacency_map
            formatted_rels = [{"source": e.get("from"), "target": e.get("to")} for e in feat_edges]
            adj_to_use = _build_adjacency_map(formatted_rels, graph_nodes)

        has_violation = False

        # Map steps to graph nodes
        mapped_step_nodes = []
        for s in steps:
            s_text = str(s)
            matched_node = _match_node_for_step(s_text, graph_nodes)
            if matched_node:
                mapped_step_nodes.append((s_text, matched_node))

        # Validate sequential step transitions
        for i in range(len(mapped_step_nodes) - 1):
            s1_text, n1_id = mapped_step_nodes[i]
            s2_text, n2_id = mapped_step_nodes[i + 1]

            if n1_id != n2_id and not _is_path_reachable(n1_id, n2_id, adj_to_use):
                has_violation = True
                
                feat_name = assigned_feature.get("name", "Unknown Feature") if assigned_feature else "Global Graph"
                
                flow_violations.append({
                    "test_case_id": tc_id,
                    "title": tc_title,
                    "reason": f"Generated step order does not follow available workflow path in feature '{feat_name}'.",
                    "expected_transition": f"Valid workflow path from '{n1_id}'",
                    "generated_transition": f"Invalid jump from '{s1_text[:40]}' to '{s2_text[:40]}'"
                })
                break

        if not has_violation:
            valid_test_cases += 1

    flow_correctness_pct = round((valid_test_cases / max(total_test_cases, 1)) * 100, 1)

    # 3. Workflow Coverage Analysis
    total_workflow_steps = 0
    covered_steps_count = 0
    missing_workflow_steps = []

    # Use Features instead of the old workflow_model
    for feat in features:
        feat_name = feat.get("name", "Workflow")
        feat_steps = feat.get("ordered_steps", [])
        
        for st_action in feat_steps:
            total_workflow_steps += 1
            st_norm = _normalize(st_action)

            # Check if covered in test cases
            is_step_covered = any(
                st_norm in _normalize(str(tc))
                for tc in test_cases
            )

            if is_step_covered:
                covered_steps_count += 1
            else:
                missing_workflow_steps.append({
                    "feature": feat_name,
                    "step_id": st_action,  # Using name as fallback for id
                    "action": st_action
                })

    workflow_coverage_pct = round((covered_steps_count / max(total_workflow_steps, 1)) * 100, 1) if total_workflow_steps > 0 else 100.0

    return {
        "workflow_coverage_pct": workflow_coverage_pct,
        "total_workflow_steps": total_workflow_steps,
        "covered_workflow_steps": covered_steps_count,
        "missing_workflow_steps": missing_workflow_steps,
        "flow_correctness_pct": flow_correctness_pct,
        "total_test_cases_validated": total_test_cases,
        "valid_flow_test_cases": valid_test_cases,
        "flow_violations": flow_violations
    }
