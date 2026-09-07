# graph/evaluation/fact_evaluator.py
"""Information Retention & Fact Evaluator (MINE-1 inspired).
Evaluates how accurately document facts are retained in the Neo4j Knowledge Graph.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _normalize_str(s: str) -> str:
    """Normalize string for fuzzy comparison."""
    if not s:
        return ""
    return "".join(c.lower() for c in str(s) if c.isalnum())


def _string_similarity(s1: str, s2: str) -> float:
    """Token-overlap similarity metric."""
    n1, n2 = _normalize_str(s1), _normalize_str(s2)
    if not n1 or not n2:
        return 0.0
    if n1 in n2 or n2 in n1:
        return 1.0
    
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def evaluate_fact(fact: Dict[str, Any], graph_nodes: List[Dict[str, Any]], graph_rels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate a single fact against the Neo4j graph nodes and relationships."""
    subj = fact.get("subject", "")
    pred = fact.get("predicate", "")
    obj = fact.get("object", "")
    fact_text = fact.get("text", "")

    subj_matched_node = None
    obj_matched_node = None

    for node in graph_nodes:
        node_name = str(node.get("name") or node.get("id") or "")
        node_desc = str(node.get("description") or "")
        aliases = [str(a) for a in node.get("aliases", []) if a]

        # Check name or aliases
        all_node_strings = [node_name, node_desc] + aliases
        
        if any(_string_similarity(subj, s) >= 0.6 for s in all_node_strings):
            subj_matched_node = node
        if any(_string_similarity(obj, s) >= 0.6 for s in all_node_strings):
            obj_matched_node = node

    rel_matched = False
    matched_rel_type = None

    if subj_matched_node and obj_matched_node:
        s_id = str(subj_matched_node.get("id") or subj_matched_node.get("name"))
        o_id = str(obj_matched_node.get("id") or obj_matched_node.get("name"))

        for rel in graph_rels:
            rel_src = str(rel.get("source") or rel.get("start") or rel.get("from") or "")
            rel_tgt = str(rel.get("target") or rel.get("end") or rel.get("to") or "")

            if (rel_src == s_id and rel_tgt == o_id) or (rel_src == o_id and rel_tgt == s_id):
                rel_matched = True
                matched_rel_type = rel.get("type") or rel.get("relationship", "CONNECTED")
                break

    if subj_matched_node and obj_matched_node and rel_matched:
        status = "verified"
        score = 1.0
        details = f"Fact retained: '{subj_matched_node.get('name')}' connected to '{obj_matched_node.get('name')}' via '{matched_rel_type}'"
    elif subj_matched_node or obj_matched_node:
        status = "partially_verified"
        score = 0.5
        found = []
        if subj_matched_node:
            found.append(f"Subject '{subj_matched_node.get('name')}'")
        if obj_matched_node:
            found.append(f"Object '{obj_matched_node.get('name')}'")
        details = f"Partially retained: Found {', '.join(found)} in graph, but direct relationship is missing."
    else:
        # Text matching check across node properties
        text_found = False
        if fact_text:
            fact_words = set(fact_text.lower().split())
            for node in graph_nodes:
                node_text = (str(node.get("name", "")) + " " + str(node.get("description", ""))).lower()
                node_words = set(node_text.split())
                if len(fact_words.intersection(node_words)) >= 3:
                    text_found = True
                    break
        if text_found:
            status = "partially_verified"
            score = 0.3
            details = "Fact details mentioned in graph text, but entities not distinctly linked."
        else:
            status = "missing"
            score = 0.0
            details = f"Fact missing: Neither subject '{subj}' nor object '{obj}' mapped to graph nodes."

    return {
        "fact_id": fact.get("id"),
        "fact_text": fact_text,
        "subject": subj,
        "predicate": pred,
        "object": obj,
        "status": status,
        "score": score,
        "details": details,
        "matched_subject": subj_matched_node.get("name") if subj_matched_node else None,
        "matched_object": obj_matched_node.get("name") if obj_matched_node else None,
        "matched_rel": matched_rel_type
    }


def evaluate_information_retention(facts: List[Dict[str, Any]], graph_nodes: List[Dict[str, Any]], graph_rels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate Information Retention over extracted facts."""
    if not facts:
        return {
            "fact_precision": 0.0,
            "fact_recall": 0.0,
            "fact_f1": 0.0,
            "total_facts": 0,
            "verified_count": 0,
            "partial_count": 0,
            "missing_count": 0,
            "facts_available": False,
            "fact_results": []
        }

    results = []
    total_score = 0.0
    verified_count = 0
    partial_count = 0
    missing_count = 0

    for fact in facts:
        eval_res = evaluate_fact(fact, graph_nodes, graph_rels)
        results.append(eval_res)
        total_score += eval_res["score"]

        if eval_res["status"] == "verified":
            verified_count += 1
        elif eval_res["status"] == "partially_verified":
            partial_count += 1
        else:
            missing_count += 1

    total_facts = len(facts)
    recall = round(total_score / total_facts, 4) if total_facts > 0 else 0.0
    precision = round((verified_count + 0.5 * partial_count) / max(total_facts, 1), 4)
    precision = min(1.0, precision)
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
        "total_facts": total_facts,
        "verified_count": verified_count,
        "partial_count": partial_count,
        "missing_count": missing_count,
        "facts_available": True,
        "fact_results": results
    }
