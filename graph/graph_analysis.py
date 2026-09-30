"""
graph/graph_analysis.py

Comprehensive graph structure analysis, topological health evaluation,
confidence scoring, sequence validation, and failure trace analysis.
"""

import json
import re
from collections import defaultdict, deque

from graph.flow_graph_analysis import analyze_flow
from graph.neo4j_manager import get_all_nodes, get_all_relationships


# =============================================================================
# PART 1: Structural Health & Sequence Evaluation (formerly graph_eval.py)
# =============================================================================

def _local_backbone_id(item_id: str) -> str:
    """Canonicalization key used solely by evaluate_sequence() to compare ground-truth
    'from'/'to' item IDs on a consistent key."""
    slug = re.sub(r'[^a-z0-9]+', '_', item_id.lower()).strip('_')
    return f"requirement_{slug}"


def evaluate_structure(
    nodes: list,
    relationships: list,
    items_with_links: list,
    hub_degree_ratio: float = 0.15,
) -> dict:
    """
    Structural health checks on the graph as-built.

    hub_degree_ratio: a node whose degree exceeds this fraction of the
    sum of all node degrees is flagged as a potential hub / over-generalized entity.
    """
    node_ids = {n["id"] for n in nodes}
    degree = defaultdict(int)
    adjacency = defaultdict(set)
    for rel in relationships:
        f, t = rel.get("from"), rel.get("to")
        if f in node_ids and t in node_ids and f != t:
            degree[f] += 1
            degree[t] += 1
            adjacency[f].add(t)
            adjacency[t].add(f)

    # 1. Isolated nodes
    isolated = sorted(n["id"] for n in nodes if degree[n["id"]] == 0)

    # 2. Connected components (undirected BFS)
    visited = set()
    components = []
    for n in nodes:
        nid = n["id"]
        if nid in visited:
            continue
        comp = set()
        queue = deque([nid])
        visited.add(nid)
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nb in adjacency[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    components.sort(key=len, reverse=True)
    largest_component_size = len(components[0]) if components else 0
    largest_component_pct = (
        round(100 * largest_component_size / len(nodes), 1) if nodes else 0
    )

    # 3. Hub concentration
    total_degree = sum(degree.values())
    hubs = []
    if total_degree:
        id_to_node = {n["id"]: n for n in nodes}
        for nid, d in degree.items():
            ratio = d / total_degree
            if ratio >= hub_degree_ratio:
                node = id_to_node.get(nid, {})
                hubs.append({
                    "id": nid,
                    "name": node.get("name", nid),
                    "type": node.get("type", "?"),
                    "degree": d,
                    "share_of_all_edges_pct": round(ratio * 100, 1),
                })
    hubs.sort(key=lambda h: h["degree"], reverse=True)

    # 4. Requirements with zero extracted content
    expected_item_ids = {it["id"] for it in items_with_links}
    sourced_item_ids = {n["source"] for n in nodes if n.get("source")}
    empty_requirements = sorted(expected_item_ids - sourced_item_ids)

    node_type_counts = defaultdict(int)
    for n in nodes:
        node_type_counts[n.get("type", "?")] += 1
    rel_type_counts = defaultdict(int)
    for r in relationships:
        rel_type_counts[r.get("type", "?")] += 1

    report = {
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "isolated_nodes": isolated,
        "isolated_node_count": len(isolated),
        "connected_components": len(components),
        "largest_component_size": largest_component_size,
        "largest_component_pct_of_graph": largest_component_pct,
        "hubs": hubs,
        "requirements_with_no_extracted_content": empty_requirements,
        "node_type_counts": dict(sorted(node_type_counts.items(), key=lambda kv: -kv[1])),
        "relationship_type_counts": dict(sorted(rel_type_counts.items(), key=lambda kv: -kv[1])),
    }
    report["flags"] = _structure_flags(report)
    return report


def _structure_flags(report: dict) -> list:
    """Human-readable pass/warn/fail lines summarizing the report."""
    flags = []

    if report["isolated_node_count"]:
        flags.append(
            f"WARN: {report['isolated_node_count']} node(s) fully isolated "
            f"(0 edges) — check whether the LLM genuinely found nothing to relate it to, "
            f"or missed a real relationship in that chunk."
        )
    else:
        flags.append("PASS: no isolated nodes.")

    if report["node_count"] and report["largest_component_pct_of_graph"] < 90:
        flags.append(
            f"WARN: graph is fragmented into {report['connected_components']} "
            f"components — only {report['largest_component_pct_of_graph']}% of "
            f"nodes are in the main connected cluster. Traceability walks will "
            f"dead-end for anything outside it."
        )
    else:
        flags.append(
            f"PASS: {report['largest_component_pct_of_graph']}% of nodes in "
            f"one connected component."
        )

    if report["hubs"]:
        top = report["hubs"][0]
        flags.append(
            f"WARN: '{top['name']}' ({top['type']}) touches "
            f"{top['share_of_all_edges_pct']}% of all edges — check whether "
            f"this is a genuinely central entity or the LLM over-generalizing "
            f"across chunks."
        )
    else:
        flags.append("PASS: no hub-dominated nodes above threshold.")

    if report["requirements_with_no_extracted_content"]:
        flags.append(
            f"WARN: {len(report['requirements_with_no_extracted_content'])} "
            f"requirement(s) produced no extracted entities at all — likely "
            f"extraction failures on those specific chunks."
        )
    else:
        flags.append("PASS: every requirement produced at least some extracted content.")

    return flags


def load_ground_truth(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data.get("expected_sequence", [])


def evaluate_sequence(
    relationships: list,
    ground_truth: list,
    sequence_types: tuple = (
        "LEADS_TO", "TRIGGERS", "CAUSES",
        "VERIFIED_BY", "VERIFIES", "REALIZES", "REALIZED_BY",
        "RELATED_TO", "RELATES_TO",
    ),
) -> dict:
    """
    Compares the graph's actual sequence-ish edges against a ground truth.
    """
    graph_pairs = set()
    for rel in relationships:
        if rel.get("type") in sequence_types:
            graph_pairs.add((rel["from"], rel["to"]))

    gt_pairs = {
        (_local_backbone_id(gt["from"]), _local_backbone_id(gt["to"]))
        for gt in ground_truth
    }

    matched = graph_pairs & gt_pairs
    missed = gt_pairs - graph_pairs
    extra = graph_pairs - gt_pairs

    precision = round(len(matched) / len(graph_pairs), 3) if graph_pairs else None
    recall = round(len(matched) / len(gt_pairs), 3) if gt_pairs else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 3)
        if precision and recall and (precision + recall) > 0 else None
    )

    return {
        "ground_truth_pairs": len(gt_pairs),
        "graph_sequence_pairs": len(graph_pairs),
        "matched": len(matched),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed_pairs": sorted(missed),
        "unverified_extra_pairs": sorted(extra),
    }


# =============================================================================
# PART 2: Deterministic Confidence Scoring (formerly confidence_scorer.py)
# =============================================================================

LOW_CONFIDENCE_THRESHOLD = 0.5

_BASELINE = 0.5
_OCCURRENCE_BONUS = {1: 0.0, 2: 0.15, 3: 0.25}
_OCCURRENCE_BONUS_CAP = 0.25
_DESCRIPTION_BONUS = 0.10
_TYPE_CONFLICT_PENALTY = 0.20
_DIRECTION_CONFLICT_PENALTY = 0.15
_LINKER_CORROBORATION_BONUS = 0.15
_ENDPOINT_PROPAGATION_WEIGHT = 0.3


def _occurrence_bonus(count: int) -> float:
    return _OCCURRENCE_BONUS.get(count, _OCCURRENCE_BONUS_CAP)


def _score_node(node: dict, type_conflict_ids: set) -> tuple:
    """Returns (score, reasons)."""
    reasons = []
    score = _BASELINE

    occ = node.get("_occurrence_count", 1)
    bonus = _occurrence_bonus(occ)
    if bonus:
        score += bonus
        reasons.append(f"independently extracted from {occ} chunk(s)")
    else:
        reasons.append("extracted from only 1 chunk")

    if node.get("description"):
        score += _DESCRIPTION_BONUS
        reasons.append("has a description")

    if node.get("id") in type_conflict_ids:
        score -= _TYPE_CONFLICT_PENALTY
        reasons.append(
            f"type was inconsistent across chunks before resolving to '{node.get('type')}'"
        )

    return max(0.0, min(1.0, round(score, 2))), reasons


def _score_relationship(
    rel: dict,
    node_confidence: dict,
    direction_conflict_keys: set,
    linked_source_ids: set,
) -> tuple:
    """Returns (score, reasons)."""
    reasons = []
    score = _BASELINE

    occ = rel.get("_occurrence_count", 1)
    bonus = _occurrence_bonus(occ)
    if bonus:
        score += bonus
        reasons.append(f"independently extracted from {occ} chunk(s)")
    else:
        reasons.append("extracted from only 1 chunk")

    from_conf = node_confidence.get(rel.get("from"), _BASELINE)
    to_conf = node_confidence.get(rel.get("to"), _BASELINE)
    endpoint_avg = (from_conf + to_conf) / 2
    score += (endpoint_avg - _BASELINE) * _ENDPOINT_PROPAGATION_WEIGHT
    reasons.append(f"endpoint node confidence averages {endpoint_avg:.2f}")

    key = (tuple(sorted([rel.get("from"), rel.get("to")])), rel.get("type"))
    if key in direction_conflict_keys:
        score -= _DIRECTION_CONFLICT_PENALTY
        reasons.append("direction was inconsistent across chunks before resolving")

    source = rel.get("source")
    if source and source in linked_source_ids:
        score += _LINKER_CORROBORATION_BONUS
        reasons.append(
            f"source item '{source}' also has a rule-based requirement "
            f"cross-reference (Requirement Linker), an independent signal"
        )

    return max(0.0, min(1.0, round(score, 2))), reasons


def score_graph(
    nodes: list,
    relationships: list,
    type_conflicts: list = None,
    direction_conflicts: list = None,
    linked_items: list = None,
) -> dict:
    """
    Score a deduplicated candidate graph.
    """
    type_conflicts = type_conflicts or []
    direction_conflicts = direction_conflicts or []
    linked_items = linked_items or []

    type_conflict_ids = {c.get("id") for c in type_conflicts}
    direction_conflict_keys = {
        (tuple(sorted([c.get("from"), c.get("to")])), c.get("type"))
        for c in direction_conflicts
    }
    linked_source_ids = {
        item.get("id") for item in linked_items if item.get("linked_ids")
    }

    node_confidence = {}
    for node in nodes:
        score, reasons = _score_node(node, type_conflict_ids)
        node["confidence"] = score
        node["confidence_reasons"] = reasons
        node_confidence[node.get("id")] = score

    for rel in relationships:
        score, reasons = _score_relationship(
            rel, node_confidence, direction_conflict_keys, linked_source_ids
        )
        rel["confidence"] = score
        rel["confidence_reasons"] = reasons

    low_confidence_nodes = [n for n in nodes if n["confidence"] < LOW_CONFIDENCE_THRESHOLD]
    low_confidence_relationships = [
        r for r in relationships if r["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]

    avg_node = round(sum(n["confidence"] for n in nodes) / len(nodes), 2) if nodes else None
    avg_rel = (
        round(sum(r["confidence"] for r in relationships) / len(relationships), 2)
        if relationships else None
    )

    return {
        "nodes": nodes,
        "relationships": relationships,
        "low_confidence_nodes": low_confidence_nodes,
        "low_confidence_relationships": low_confidence_relationships,
        "summary": {
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "low_confidence_nodes": len(low_confidence_nodes),
            "low_confidence_relationships": len(low_confidence_relationships),
            "avg_node_confidence": avg_node,
            "avg_relationship_confidence": avg_rel,
        },
    }


# =============================================================================
# PART 3: Root-Cause Failure Path Tracing (formerly failure_trace.py)
# =============================================================================

def _tc_node_id(tc_id: str) -> str:
    return f"TESTCASE::{tc_id}"


def trace_failure(tc_id: str, nodes: list = None, relationships: list = None) -> dict:
    """
    Given a failed test case's id, returns a backward trace of every
    flow step that must have executed before the point(s) it validates.
    """
    if nodes is None or relationships is None:
        nodes = get_all_nodes()
        relationships = get_all_relationships()

    node_by_id = {n["id"]: n for n in nodes}
    tc_node_id = _tc_node_id(tc_id)

    tc_node = node_by_id.get(tc_node_id)
    if not tc_node:
        return {
            "tc_id": tc_id,
            "found": False,
            "is_negative": False,
            "error_message": f"No TestCase node found for '{tc_id}' -- has it been written to the graph yet?",
            "validated_nodes": [],
            "trace": [],
        }

    tc_type = tc_node.get("attributes", {}).get("type") or tc_node.get("type", "")
    tc_name = tc_node.get("name", tc_id)

    is_negative = any(kw in tc_id.upper() or kw in tc_name.upper() or kw in str(tc_type).upper()
                      for kw in ("NEG", "FAIL", "EDGE", "NEGATIVE", "ERROR", "INVALID"))

    if not is_negative:
        return {
            "tc_id": tc_id,
            "found": True,
            "is_negative": False,
            "error_message": "Traceback operations are restricted to failure/negative/edge test cases. The selected test case appears to be a positive happy path.",
            "validated_nodes": [],
            "trace": [],
        }

    validated_nodes = [
        r["to"] for r in relationships
        if r.get("from") == tc_node_id and r.get("type") == "VALIDATES"
    ]

    flow = analyze_flow(nodes, relationships)
    dominators = flow["dominators"]
    dfn = flow["dfn"]
    in_flow_subgraph = flow["in_flow_subgraph"]

    trace = []
    for target in validated_nodes:
        target_name = node_by_id.get(target, {}).get("name", target)

        target_and_preds = (dominators.get(target, set()) | {target}) if target in in_flow_subgraph else {target}
        supporting_structural = []
        for r in relationships:
            frm, to, rtype = r.get("from"), r.get("to"), r.get("type")
            if frm in target_and_preds and to not in target_and_preds:
                other = node_by_id.get(to)
                if other and other.get("type") in {"BusinessRule", "Condition", "Constraint", "DataObject", "SystemComponent"}:
                    supporting_structural.append({
                        "id": to,
                        "name": other.get("name", to),
                        "type": other.get("type"),
                        "relation": rtype,
                    })
            elif to in target_and_preds and frm not in target_and_preds:
                other = node_by_id.get(frm)
                if other and other.get("type") in {"BusinessRule", "Condition", "Constraint", "DataObject", "SystemComponent"}:
                    supporting_structural.append({
                        "id": frm,
                        "name": other.get("name", frm),
                        "type": other.get("type"),
                        "relation": rtype,
                    })

        upstream_ids = dominators.get(target, set()) - {target} if target in in_flow_subgraph else set()
        upstream_sorted = sorted(upstream_ids, key=lambda nid: dfn.get(nid, 0))

        trace.append({
            "failing_node": target,
            "failing_node_name": target_name,
            "in_flow_subgraph": target in in_flow_subgraph,
            "upstream_steps": [
                {
                    "id": nid,
                    "name": node_by_id.get(nid, {}).get("name", nid),
                    "sequence_position": dfn.get(nid),
                }
                for nid in upstream_sorted
            ],
            "supporting_structural_nodes": supporting_structural,
        })

    return {
        "tc_id": tc_id,
        "found": True,
        "is_negative": True,
        "validated_nodes": validated_nodes,
        "trace": trace,
    }


def format_trace_report(result: dict) -> str:
    """
    Human-readable version of trace_failure()'s output.
    """
    if not result["found"]:
        return f"No TestCase node found for '{result['tc_id']}' -- has it been written to the graph yet?"

    lines = [f"Failure trace for {result['tc_id']}:"]
    if not result["trace"]:
        lines.append("  (this test case has no resolved VALIDATES edges to trace from)")

    for entry in result["trace"]:
        lines.append(f"\n  Touched: {entry['failing_node_name']} ({entry['failing_node']})")
        if not entry["in_flow_subgraph"]:
            lines.append("    -- not part of the sequential flow subgraph; no upstream steps to trace (this node is only reached via structural edges, e.g. CONTAINS/USES).")
            continue
        if not entry["upstream_steps"]:
            lines.append("    -- this IS the flow entry point; nothing runs before it.")
            continue
        lines.append("    Steps that ran before this, in order (check these first):")
        for step in entry["upstream_steps"]:
            lines.append(f"      {step['sequence_position']}. {step['name']}")

    return "\n".join(lines)
