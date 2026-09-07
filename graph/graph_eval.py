"""
graph/graph_eval.py — automated correctness checks for the built
knowledge graph, so a big document's graph can be sanity-checked
without eyeballing a visualization screenshot.

Two parts, deliberately separated because they have very different
requirements:

  PART A — evaluate_structure(): structural health. No ground truth
  needed — runs on any document, today, right after the Graph
  Validator. Answers "is this graph well-formed" (connected, not
  hub-dominated, nothing missing), not "is this graph correct."

  PART B — evaluate_sequence(): sequence/traceability correctness
  against a human-authored ground truth file. This is scaffolding —
  it needs you to mark up 1-2 representative documents with the
  sequence you know is actually correct before it can tell you
  anything. See load_ground_truth()'s docstring for the file format.
  Not wired into the UI yet; call it directly once you have a
  ground-truth file to test against.

Design note (post backbone removal): graph/structural_linker.py no
longer creates a requirement-ID backbone node per item, or a PART_OF
edge from every entity back to one (see its module docstring for why).
That means the checks below that used to lean on backbone nodes had to
change what they measure:
  - Isolated nodes are no longer treated as an automatic FAIL (a real
    entity genuinely unrelated to anything else within its own chunk
    can legitimately end up isolated now) — it's a WARN worth a look,
    not proof of a wiring bug.
  - "missing backbone nodes" no longer applies — there's no backbone
    to be missing from.
  - "requirements with no extracted content" is now computed from
    node["source"] (set by tag_extraction_source()) instead of PART_OF
    edges — same signal, different mechanism.
  - evaluate_sequence() (Part B) still converts ground-truth item IDs
    with _local_backbone_id() purely as a canonicalization key for
    comparing pairs — no actual graph node is implied by it.
"""

import json
import re
from collections import defaultdict, deque


def _local_backbone_id(item_id: str) -> str:
    """Canonicalization key only — NOT a graph node id anymore (see
    module docstring). Used solely by evaluate_sequence() (Part B) to
    compare ground-truth "from"/"to" item IDs against each other on a
    consistent key, independent of formatting differences like
    'FR-001' vs 'fr_001'."""
    slug = re.sub(r'[^a-z0-9]+', '_', item_id.lower()).strip('_')
    return f"requirement_{slug}"


# ============================================================
# PART A — Structural health (no ground truth needed)
# ============================================================

def evaluate_structure(nodes: list, relationships: list, items_with_links: list,
                        hub_degree_ratio: float = 0.15) -> dict:
    """
    Structural health checks on the graph as-built.

    hub_degree_ratio: a node whose degree exceeds this fraction of the
    sum of all node degrees is flagged as a potential hub / over-
    generalized entity (e.g. an Actor node like "User" absorbing every
    chunk's extraction). Default 0.15 = flagged once a single node
    touches more than 15% of all edge-endpoints in the graph. Lower
    this for smaller documents where a real, legitimate central actor
    can still cross 15% just by being genuinely central.
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

    # 1. Isolated nodes — should be 0 given structural_linker's PART_OF
    #    tagging; a nonzero count here means either structural_linker
    #    wasn't wired into this run, or a node type it doesn't cover
    #    slipped through.
    isolated = sorted(n["id"] for n in nodes if degree[n["id"]] == 0)

    # 2. Connected components (undirected BFS). A graph fragmented into
    #    many small components, rather than one large connected core,
    #    means requirement traceability/test-path walks will dead-end
    #    partway through the document.
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

    # 3. Hub concentration — the visualization complaint ("User"
    #    connected to everything) made measurable instead of eyeballed.
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

    # 4. Requirements with zero extracted content — items that produced
    #    no node at all (no node["source"] pointing to them). Same
    #    signal the old PART_OF-based check gave, computed differently
    #    now that there's no backbone node to check for orphaned
    #    PART_OF targets against (see module docstring).
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
    """Human-readable pass/warn/fail lines summarizing the report —
    what you'd actually want to read first, before the raw numbers."""
    flags = []

    if report["isolated_node_count"]:
        flags.append(
            f"WARN: {report['isolated_node_count']} node(s) fully isolated "
            f"(0 edges) — expected occasionally now that entities aren't "
            f"force-connected to a backbone node (see structural_linker.py's "
            f"module docstring). Worth a look, not necessarily a bug: check "
            f"whether the LLM genuinely found nothing to relate it to, or "
            f"missed a real relationship in that chunk."
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
            f"across chunks (the original hub-and-spoke problem)."
        )
    else:
        flags.append("PASS: no hub-dominated nodes above threshold.")

    if report["requirements_with_no_extracted_content"]:
        flags.append(
            f"WARN: {len(report['requirements_with_no_extracted_content'])} "
            f"requirement(s) produced no extracted entities at all — likely "
            f"extraction failures on those specific chunks, worth auditing "
            f"individually rather than assuming it's fine."
        )
    else:
        flags.append("PASS: every requirement produced at least some extracted content.")

    return flags


# ============================================================
# PART B — Sequence/traceability correctness (needs ground truth)
# ============================================================
#
# This half can't tell you anything until you author a ground-truth
# file for at least one representative document — there's no way to
# check "is the sequence correct" without knowing what correct means
# for that specific document's actual workflow.
#
# Ground truth format — save as e.g. eval/ground_truth_test_req.json:
#
# {
#   "document": "test_req.txt",
#   "expected_sequence": [
#     {"from": "FR-001", "to": "FR-002", "reason": "login happens before the dashboard is shown"},
#     {"from": "FR-002", "to": "FR-003", "reason": "dashboard is viewed before logout"}
#   ]
# }
#
# "from"/"to" are requirement-item IDs as they appear in the source
# document (e.g. "FR-001", "BR_STAY_6") — NOT internal graph node ids.
# The loader below converts them for you, so authoring a ground truth
# file never requires knowing the internal id format.

def load_ground_truth(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data.get("expected_sequence", [])


def evaluate_sequence(relationships: list, ground_truth: list,
                       sequence_types: tuple = (
                           "LEADS_TO", "TRIGGERS", "CAUSES",
                           "VERIFIED_BY", "VERIFIES", "REALIZES", "REALIZED_BY",
                           "RELATED_TO", "RELATES_TO",
                       )) -> dict:
    """
    Compares the graph's actual sequence-ish edges against a
    human-authored ground truth.

    sequence_types is deliberately permissive by default — any edge
    type listed counts as "the graph captured this pair somehow" (e.g.
    a VERIFIES edge counts as capturing a sequence pair). NEXT_IN_DOCUMENT
    is no longer in the default set since structural_linker.py stopped
    producing it (see its module docstring) — sequence/flow now has to
    come from the LLM's own functional edges, not a document-order
    backbone.

    Returns precision/recall/F1 AND the actual missed/wrong pairs named
    explicitly — a bare score doesn't tell you which requirement pair
    to go look at.

    Note "unverified_extra_pairs" isn't necessarily wrong — it's pairs
    the graph captured that aren't in your ground truth, which may just
    mean your ground truth file is incomplete, not that the graph is.
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