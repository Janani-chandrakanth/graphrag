# graph/evaluation/node_coverage.py
"""Node & Entity Coverage Evaluator for CIG Evaluation Module.
Measures entity coverage ratio, node type distribution, and orphan node detection.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def compute_node_coverage(expected_facts: List[Dict[str, Any]], graph_nodes: List[Dict[str, Any]], graph_rels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes entity coverage, node type distribution, and orphan node counts."""
    total_nodes = len(graph_nodes)
    if total_nodes == 0:
        return {
            "node_coverage_score": 0.0,
            "total_nodes": 0,
            "orphan_nodes": 0,
            "orphan_ratio": 0.0,
            "type_distribution": {},
            "covered_entity_count": 0,
            "total_expected_entities": 0
        }

    # 1. Type Distribution
    type_dist = {}
    node_ids = set()
    for n in graph_nodes:
        n_type = n.get("type", "Unknown")
        type_dist[n_type] = type_dist.get(n_type, 0) + 1
        n_id = str(n.get("id") or n.get("name"))
        if n_id:
            node_ids.add(n_id)

    # 2. Orphan Node Detection (nodes with 0 incoming or outgoing relationships)
    connected_nodes = set()
    for rel in graph_rels:
        src = str(rel.get("source") or rel.get("start") or rel.get("from") or "")
        tgt = str(rel.get("target") or rel.get("end") or rel.get("to") or "")
        if src: connected_nodes.add(src)
        if tgt: connected_nodes.add(tgt)

    orphan_nodes = [nid for nid in node_ids if nid not in connected_nodes]
    orphan_count = len(orphan_nodes)
    orphan_ratio = round(orphan_count / total_nodes, 4) if total_nodes > 0 else 0.0

    # 3. Expected Entity Coverage
    expected_entities = set()
    for fact in expected_facts:
        if fact.get("subject"): expected_entities.add(str(fact.get("subject")).lower())
        if fact.get("object"): expected_entities.add(str(fact.get("object")).lower())

    covered_count = 0
    graph_names_lower = [str(n.get("name") or n.get("id") or "").lower() for n in graph_nodes]

    for entity in expected_entities:
        if any(entity in gname or gname in entity for gname in graph_names_lower):
            covered_count += 1

    total_expected = len(expected_entities)
    coverage_score = round(covered_count / total_expected, 4) if total_expected > 0 else 1.0

    # Weight coverage with non-orphan penalty
    final_score = round(coverage_score * (1.0 - 0.5 * orphan_ratio), 4)

    return {
        "node_coverage_score": final_score,
        "raw_coverage": coverage_score,
        "total_nodes": total_nodes,
        "orphan_nodes": orphan_count,
        "orphan_ratio": orphan_ratio,
        "type_distribution": type_dist,
        "covered_entity_count": covered_count,
        "total_expected_entities": total_expected
    }
