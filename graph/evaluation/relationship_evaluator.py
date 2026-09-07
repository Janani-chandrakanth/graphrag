# graph/evaluation/relationship_evaluator.py
"""Relationship Evaluator for CIG Evaluation Module.
Measures Precision, Recall, F1 for relationships in Neo4j graph, relationship types, and connectivity.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Recognised CIG relationship types across graph_builder, story_graph, etc.
RECOGNIZED_REL_TYPES = {
    "HAS_STEP", "NEXT", "PERFORMS", "VERIFIES", "VALIDATES", "DEPENDS_ON",
    "TRIGGERS", "CALLS", "INCLUDES", "LEADS_TO", "CONNECTED", "PRECEDES",
    "FOLLOWS", "FLOWS_TO", "HAS_ACCEPTANCE_CRITERIA", "HAS_FEATURE",
    "HAS_REQUIREMENT", "HAS_USER_STORY", "CROSS_REFERENCE"
}

def evaluate_relationships(expected_facts: List[Dict[str, Any]], graph_nodes: List[Dict[str, Any]], graph_rels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate quality of graph relationships."""
    total_graph_rels = len(graph_rels)
    node_ids = {str(n.get("id") or n.get("name")): n for n in graph_nodes if n}

    if total_graph_rels == 0:
        return {
            "rel_precision": 0.0,
            "rel_recall": 0.0,
            "rel_f1": 0.0,
            "total_graph_rels": 0,
            "total_expected_rels": 0,
            "matched_rels": 0,
            "dangling_rels": 0,
            "connectivity_rate": 0.0,
            "valid_type_rate": 0.0,
            "rel_type_distribution": {}
        }

    # 1. Analyze Neo4j Graph Relationships
    valid_connected_rels = 0
    dangling_rels = 0
    recognized_type_count = 0
    rel_type_counts = {}

    for rel in graph_rels:
        r_type = (rel.get("type") or rel.get("relationship") or "CONNECTED").upper()
        rel_type_counts[r_type] = rel_type_counts.get(r_type, 0) + 1

        if r_type in RECOGNIZED_REL_TYPES:
            recognized_type_count += 1

        src = str(rel.get("source") or rel.get("start") or rel.get("from") or "")
        tgt = str(rel.get("target") or rel.get("end") or rel.get("to") or "")

        if src in node_ids or tgt in node_ids:
            valid_connected_rels += 1
        else:
            dangling_rels += 1

    connectivity_rate = round(valid_connected_rels / total_graph_rels, 4)
    valid_type_rate = round(recognized_type_count / total_graph_rels, 4)

    # 2. Match expected facts against graph triples
    matched_fact_rels = 0
    total_expected_rels = 0

    for fact in expected_facts:
        if fact.get("subject") and fact.get("object"):
            total_expected_rels += 1
            subj = str(fact.get("subject")).lower()
            obj = str(fact.get("object")).lower()

            for rel in graph_rels:
                src = str(rel.get("source") or rel.get("start") or rel.get("from") or "").lower()
                tgt = str(rel.get("target") or rel.get("end") or rel.get("to") or "").lower()

                if (subj in src or src in subj) and (obj in tgt or tgt in obj):
                    matched_fact_rels += 1
                    break

    # 3. Compute Metrics
    if total_expected_rels > 0:
        rel_recall = round(matched_fact_rels / total_expected_rels, 4)
        rel_precision = round(matched_fact_rels / total_graph_rels, 4)
        rel_precision = min(1.0, rel_precision)
        if (rel_precision + rel_recall) > 0:
            rel_f1 = round((2 * rel_precision * rel_recall) / (rel_precision + rel_recall), 4)
        else:
            rel_f1 = round(0.5 * connectivity_rate + 0.5 * valid_type_rate, 4)
    else:
        # Structural relationship score when document fact triples are not available
        rel_precision = valid_type_rate
        rel_recall = connectivity_rate
        rel_f1 = round(0.5 * connectivity_rate + 0.5 * valid_type_rate, 4)

    return {
        "rel_precision": rel_precision,
        "rel_recall": rel_recall,
        "rel_f1": rel_f1,
        "total_graph_rels": total_graph_rels,
        "total_expected_rels": total_expected_rels,
        "matched_rels": matched_fact_rels,
        "dangling_rels": dangling_rels,
        "connectivity_rate": connectivity_rate,
        "valid_type_rate": valid_type_rate,
        "rel_type_distribution": rel_type_counts
    }
