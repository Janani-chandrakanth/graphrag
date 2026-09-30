"""
graph/evaluation/graph_evaluators.py

Consolidated Graph Evaluation Suite:
1. Fact Extraction & Generation (extract_facts_with_llm, extract_facts_from_text, extract_facts_from_nodes, generate_evaluation_facts)
2. Information Retention & Fact Evaluator (evaluate_fact, evaluate_information_retention)
3. Relationship Evaluator (evaluate_relationships, RECOGNIZED_REL_TYPES)
4. Workflow & Sequence Evaluator (evaluate_workflow, WORKFLOW_NODE_TYPES, SEQUENCE_REL_TYPES)
5. Node & Entity Coverage Evaluator (compute_node_coverage)
6. Evaluation Report Builder & Persistence (build_evaluation_report, save_evaluation_run, get_evaluation_history, generate_graph_eval_narrative_summary)
"""

import logging
import json
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from parser.llm_client import call_ollama
from graph.neo4j_manager import get_stored_documents, run_cypher_query

logger = logging.getLogger(__name__)


# =========================================================
# 1. FACT EXTRACTION & GENERATION
# =========================================================

def extract_facts_with_llm(text: str) -> List[Dict[str, Any]]:
    """Extract atomic facts using LLM for deep semantic requirement coverage."""
    if not text or len(text.strip()) < 20:
        return []

    prompt = f"""You are a Software Requirements Fact Extractor. Extract all explicit business facts, user actions, system operations, and conditions from the following text into atomic (Subject, Predicate, Object) fact triples.

TEXT:
\"\"\"{text[:4000]}\"\"\"

Return ONLY a raw JSON array of fact objects with keys "subject", "predicate", "object", "text". Do not include markdown formatting or conversational text outside JSON.
JSON Output Format:
[
  {{"subject": "User", "predicate": "SUBMITS", "object": "Claim Form", "text": "User submits claim form for processing."}}
]
"""
    try:
        res = call_ollama(prompt, timeout=60)
        raw_resp = res.get("raw", "")
        if raw_resp and "[" in raw_resp and "]" in raw_resp:
            json_str = raw_resp[raw_resp.find("["):raw_resp.rfind("]")+1]
            data = json.loads(json_str)
            facts = []
            for i, item in enumerate(data, start=1):
                if isinstance(item, dict) and item.get("subject") and item.get("object"):
                    facts.append({
                        "id": f"llm_fact_{i}",
                        "text": item.get("text") or f"{item.get('subject')} {item.get('predicate')} {item.get('object')}",
                        "subject": str(item.get("subject")).strip(),
                        "predicate": str(item.get("predicate", "RELATES_TO")).strip().upper(),
                        "object": str(item.get("object")).strip(),
                        "type": "llm_fact"
                    })
            if facts:
                logger.info("Extracted %d facts via LLM", len(facts))
                return facts
    except Exception as e:
        logger.warning("LLM fact extraction error: %s", e)

    return []


def extract_facts_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract atomic facts using regex sentence pattern matching."""
    facts = []
    if not text:
        return facts

    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    fact_id = 1

    for sentence in sentences:
        clean_stmt = sentence.strip()
        if len(clean_stmt) < 8:
            continue

        match = re.search(
            r'([A-Z][a-zA-Z0-9_\s]{1,30}?)\s+(shall|must|should|can|will|submits|validates|creates|approves|rejects|sends|receives|stores|updates|deletes|processes|triggers|contains|includes|requires)\s+(.+)',
            clean_stmt, re.IGNORECASE
        )

        if match:
            subj, pred, obj = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            facts.append({
                "id": f"fact_{fact_id}",
                "text": clean_stmt[:150],
                "subject": subj,
                "predicate": pred.upper(),
                "object": obj[:60],
                "type": "requirement_fact"
            })
            fact_id += 1
        else:
            words = clean_stmt.split()
            if len(words) >= 4:
                facts.append({
                    "id": f"fact_{fact_id}",
                    "text": clean_stmt[:150],
                    "subject": words[0],
                    "predicate": "DESCRIBES",
                    "object": " ".join(words[1:5]),
                    "type": "text_fact"
                })
                fact_id += 1

    return facts


def extract_facts_from_nodes(graph_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback fact generator deriving facts from Neo4j node attributes."""
    facts = []
    fact_id = 1
    for node in graph_nodes:
        name = node.get("name") or node.get("id")
        desc = node.get("description", "")
        n_type = node.get("type", "Entity")

        if name and desc and len(desc) > 10:
            sub_facts = extract_facts_from_text(desc)
            for sf in sub_facts:
                sf["id"] = f"node_fact_{fact_id}"
                sf["subject"] = name
                facts.append(sf)
                fact_id += 1
        elif name:
            facts.append({
                "id": f"node_fact_{fact_id}",
                "text": f"{name} is a {n_type}",
                "subject": name,
                "predicate": "IS_A",
                "object": n_type,
                "type": "entity_type_fact"
            })
            fact_id += 1
    return facts


def generate_evaluation_facts(
    normalized_document: Optional[Dict[str, Any]] = None,
    raw_text: Optional[str] = None,
    graph_nodes: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Generate evaluation facts using LLM extraction first, falling back to regex/rules."""
    extracted_facts = []

    if not raw_text:
        raw_text = (
            st.session_state.get("raw_text") or
            st.session_state.get("extracted_text") or
            st.session_state.get("text")
        )

    if not normalized_document:
        normalized_document = st.session_state.get("normalized_document")

    # 1. LLM-based fact extraction (semantic)
    if raw_text and len(str(raw_text).strip()) > 20:
        extracted_facts = extract_facts_with_llm(str(raw_text))
        if not extracted_facts:
            extracted_facts = extract_facts_from_text(str(raw_text))

    # 2. Secondary: Normalized document structure
    if not extracted_facts and normalized_document and isinstance(normalized_document, dict):
        sections = normalized_document.get("sections", [])
        for section in sections:
            sec_content = section.get("content", "")
            if sec_content:
                sec_facts = extract_facts_with_llm(sec_content) or extract_facts_from_text(sec_content)
                extracted_facts.extend(sec_facts)

            requirements = section.get("requirements", [])
            for req in requirements:
                req_text = req.get("text", "") or req.get("description", "")
                if req_text:
                    r_facts = extract_facts_from_text(req_text)
                    extracted_facts.extend(r_facts)

    # 3. Tertiary: Neo4j stored documents
    if not extracted_facts:
        try:
            stored_docs = get_stored_documents()
            for doc in stored_docs:
                doc_text = doc.get("text") or doc.get("content") or ""
                if doc_text:
                    extracted_facts.extend(extract_facts_from_text(doc_text))
                    if len(extracted_facts) >= 50:
                        break
        except Exception as e:
            logger.warning("Could not fetch stored documents from Neo4j: %s", e)

    # 4. Fallback: Graph nodes
    if not extracted_facts and graph_nodes:
        extracted_facts = extract_facts_from_nodes(graph_nodes)

    logger.info("Generated %d evaluation facts", len(extracted_facts))
    return extracted_facts


# =========================================================
# 2. INFORMATION RETENTION & FACT EVALUATOR
# =========================================================

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


# =========================================================
# 3. RELATIONSHIP EVALUATOR
# =========================================================

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

    if total_expected_rels > 0:
        rel_recall = round(matched_fact_rels / total_expected_rels, 4)
        rel_precision = round(matched_fact_rels / total_graph_rels, 4)
        rel_precision = min(1.0, rel_precision)
        if (rel_precision + rel_recall) > 0:
            rel_f1 = round((2 * rel_precision * rel_recall) / (rel_precision + rel_recall), 4)
        else:
            rel_f1 = round(0.5 * connectivity_rate + 0.5 * valid_type_rate, 4)
    else:
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


# =========================================================
# 4. WORKFLOW & SEQUENCE EVALUATOR
# =========================================================

WORKFLOW_NODE_TYPES = {
    "WORKFLOWSTEP", "FEATURE", "STEP", "ACTION", "ACTIVITY",
    "PROCESS", "EVENT", "DECISION", "USECASE", "USERSTORY", "REQUIREMENT"
}

SEQUENCE_REL_TYPES = {
    "NEXT", "HAS_STEP", "LEADS_TO", "TRIGGERS", "PRECEDES",
    "FLOWS_TO", "DEPENDS_ON", "FOLLOWS", "THEN"
}


def evaluate_workflow(workflow_model: Optional[Any], graph_nodes: List[Dict[str, Any]], graph_rels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates business workflow quality and procedural step ordering."""
    total_nodes = len(graph_nodes)
    if total_nodes == 0:
        return {
            "sequence_accuracy": 0.0,
            "step_coverage": 0.0,
            "transition_accuracy": 0.0,
            "total_steps": 0,
            "total_flow_rels": 0,
            "feature_count": 0,
            "total_feature_steps": 0,
            "valid_transitions": 0,
            "orphan_step_count": 0
        }

    step_nodes = [
        n for n in graph_nodes
        if str(n.get("type", "")).upper() in WORKFLOW_NODE_TYPES or n.get("sequence") is not None
    ]
    
    if not step_nodes:
        step_nodes = graph_nodes

    flow_rels = [
        r for r in graph_rels
        if str(r.get("type") or r.get("relationship") or "").upper() in SEQUENCE_REL_TYPES
    ]

    total_steps = len(step_nodes)
    total_flow_rels = len(flow_rels)

    connected_step_ids = set()
    for rel in flow_rels:
        src = str(rel.get("source") or rel.get("start") or rel.get("from") or "")
        tgt = str(rel.get("target") or rel.get("end") or rel.get("to") or "")
        if src: connected_step_ids.add(src)
        if tgt: connected_step_ids.add(tgt)

    step_coverage_ratio = round(len(connected_step_ids) / max(total_steps, 1), 4)

    sequenced_nodes = [n for n in graph_nodes if n.get("sequence") is not None]
    sequence_prop_ratio = round(len(sequenced_nodes) / max(total_nodes, 1), 4)

    feature_count = 0
    total_feature_steps = 0
    valid_transitions = 0

    if workflow_model and hasattr(workflow_model, "features"):
        features = getattr(workflow_model, "features", [])
        feature_count = len(features)
        
        for feat in features:
            steps = getattr(feat, "steps", [])
            total_feature_steps += len(steps)

            for i in range(len(steps) - 1):
                curr_id = str(getattr(steps[i], "id", ""))
                next_id = str(getattr(steps[i + 1], "id", ""))

                has_edge = any(
                    (str(r.get("source") or r.get("start") or r.get("from")) == curr_id and
                     str(r.get("target") or r.get("end") or r.get("to")) == next_id)
                    for r in graph_rels
                )
                if has_edge:
                    valid_transitions += 1

    if total_feature_steps > 1:
        transition_accuracy = round(valid_transitions / max(total_feature_steps - feature_count, 1), 4)
    elif total_flow_rels > 0:
        transition_accuracy = round(min(1.0, total_flow_rels / max(total_steps, 1)), 4)
    else:
        transition_accuracy = sequence_prop_ratio

    sequence_accuracy = round(0.5 * step_coverage_ratio + 0.5 * transition_accuracy, 4)

    return {
        "sequence_accuracy": sequence_accuracy,
        "step_coverage": step_coverage_ratio,
        "transition_accuracy": transition_accuracy,
        "total_steps": total_steps,
        "total_flow_rels": total_flow_rels,
        "feature_count": feature_count,
        "total_feature_steps": total_feature_steps,
        "valid_transitions": valid_transitions,
        "orphan_step_count": max(0, total_steps - len(connected_step_ids))
    }


# =========================================================
# 5. NODE & ENTITY COVERAGE EVALUATOR
# =========================================================

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

    type_dist = {}
    node_ids = set()
    for n in graph_nodes:
        n_type = n.get("type", "Unknown")
        type_dist[n_type] = type_dist.get(n_type, 0) + 1
        n_id = str(n.get("id") or n.get("name"))
        if n_id:
            node_ids.add(n_id)

    connected_nodes = set()
    for rel in graph_rels:
        src = str(rel.get("source") or rel.get("start") or rel.get("from") or "")
        tgt = str(rel.get("target") or rel.get("end") or rel.get("to") or "")
        if src: connected_nodes.add(src)
        if tgt: connected_nodes.add(tgt)

    orphan_nodes = [nid for nid in node_ids if nid not in connected_nodes]
    orphan_count = len(orphan_nodes)
    orphan_ratio = round(orphan_count / total_nodes, 4) if total_nodes > 0 else 0.0

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


# =========================================================
# 6. EVALUATION REPORT BUILDER & PERSISTENCE
# =========================================================

def generate_graph_eval_narrative_summary(report: Dict[str, Any]) -> str:
    """Generates a clean, human-readable executive summary text for Graph Quality Evaluation."""
    doc = report.get("document_name", "Requirement Document")
    overall = report.get("overall_score", 0.0)
    summary = report.get("summary", {})
    metrics = report.get("metrics", {})

    info_ret = metrics.get("information_retention", {})
    rel_q = metrics.get("relationship_quality", {})
    wf_q = metrics.get("workflow_sequence", {})
    nc_q = metrics.get("node_coverage", {})

    total_facts = info_ret.get("total_facts", 0)
    verified = info_ret.get("verified_count", 0)
    partial = info_ret.get("partial_count", 0)
    retention_f1 = round(summary.get("fact_f1", 0.0) * 100, 1)

    nodes_count = summary.get("total_nodes_in_graph", 0)
    rels_count = summary.get("total_relationships", 0)
    orphans = nc_q.get("orphan_nodes", 0)
    connectivity = round(rel_q.get("connectivity_rate", 0.0) * 100, 1)

    steps_count = wf_q.get("total_steps", 0)
    flow_rels = wf_q.get("total_flow_rels", 0)
    wf_acc = round(summary.get("workflow_accuracy", 0.0) * 100, 1)

    quality_tier = "High Quality" if overall >= 80 else ("Moderate Quality" if overall >= 60 else "Needs Improvement")

    narrative = (
        f"Executive Summary for '{doc}':\n"
        f"The Knowledge Graph comprises {nodes_count} nodes and {rels_count} relationships in Neo4j. "
        f"Out of {total_facts} requirement facts extracted from text, {verified} facts were fully verified "
        f"and {partial} facts were partially retained, yielding an Information Retention F1 score of {retention_f1}%.\n\n"
        f"Graph Health & Workflow Structure: Relationship connectivity rate is {connectivity}% with {orphans} orphan nodes. "
        f"The graph covers {steps_count} workflow steps connected across {flow_rels} sequential process edges, "
        f"achieving {wf_acc}% Workflow Sequence Accuracy.\n\n"
        f"Overall Quality Index: {overall}% ({quality_tier})."
    )
    return narrative


def build_evaluation_report(
    retention_results: Dict[str, Any],
    rel_results: Dict[str, Any],
    workflow_results: Dict[str, Any],
    coverage_results: Dict[str, Any],
    doc_name: str = "Requirement Document"
) -> Dict[str, Any]:
    """Combines metrics from all evaluation modules into an overall report."""
    fact_f1 = retention_results.get("fact_f1", 0.0)
    rel_f1 = rel_results.get("rel_f1", 0.0)
    workflow_acc = workflow_results.get("sequence_accuracy", 0.0)
    node_cov = coverage_results.get("node_coverage_score", 0.0)
    facts_available = retention_results.get("facts_available", True)

    if facts_available and retention_results.get("total_facts", 0) > 0:
        w_retention = 0.35
        w_rel = 0.25
        w_wf = 0.25
        w_cov = 0.15
    else:
        w_retention = 0.10
        w_rel = 0.35
        w_wf = 0.35
        w_cov = 0.20

    weighted_score = (
        w_retention * fact_f1 +
        w_rel * rel_f1 +
        w_wf * workflow_acc +
        w_cov * node_cov
    )
    overall_score_pct = round(weighted_score * 100, 2)

    report = {
        "run_id": f"eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "document_name": doc_name,
        "overall_score": overall_score_pct,
        "weights": {
            "retention": int(w_retention * 100),
            "relationship": int(w_rel * 100),
            "workflow": int(w_wf * 100),
            "coverage": int(w_cov * 100)
        },
        "metrics": {
            "information_retention": retention_results,
            "relationship_quality": rel_results,
            "workflow_sequence": workflow_results,
            "node_coverage": coverage_results
        },
        "summary": {
            "fact_f1": fact_f1,
            "rel_f1": rel_f1,
            "workflow_accuracy": workflow_acc,
            "node_coverage": node_cov,
            "total_facts_evaluated": retention_results.get("total_facts", 0),
            "total_nodes_in_graph": coverage_results.get("total_nodes", 0),
            "total_relationships": rel_results.get("total_graph_rels", 0)
        }
    }

    report["narrative_summary"] = generate_graph_eval_narrative_summary(report)
    return report


def save_evaluation_run(report: Dict[str, Any]) -> bool:
    """Saves the evaluation report as an `:EvaluationRun` node in Neo4j."""
    run_id = report.get("run_id")
    timestamp = report.get("timestamp")
    doc_name = report.get("document_name", "Unknown Document")
    overall_score = report.get("overall_score", 0.0)
    summary = report.get("summary", {})

    cypher = """
    CREATE (e:EvaluationRun {
        id: $run_id,
        timestamp: $timestamp,
        document_name: $doc_name,
        overall_score: $overall_score,
        fact_f1: $fact_f1,
        rel_f1: $rel_f1,
        workflow_accuracy: $workflow_accuracy,
        node_coverage: $node_coverage,
        details_json: $details_json
    })
    RETURN e.id AS id
    """
    params = {
        "run_id": run_id,
        "timestamp": timestamp,
        "doc_name": doc_name,
        "overall_score": overall_score,
        "fact_f1": summary.get("fact_f1", 0.0),
        "rel_f1": summary.get("rel_f1", 0.0),
        "workflow_accuracy": summary.get("workflow_accuracy", 0.0),
        "node_coverage": summary.get("node_coverage", 0.0),
        "details_json": json.dumps(report)
    }

    try:
        res = run_cypher_query(cypher, params)
        logger.info("Successfully saved EvaluationRun %s to Neo4j.", run_id)
        return True
    except Exception as e:
        logger.error("Failed to save EvaluationRun to Neo4j: %s", e)
        return False


def get_evaluation_history() -> List[Dict[str, Any]]:
    """Retrieves past evaluation runs from Neo4j."""
    cypher = """
    MATCH (e:EvaluationRun)
    RETURN e.id AS run_id,
           e.timestamp AS timestamp,
           e.document_name AS document_name,
           e.overall_score AS overall_score,
           e.fact_f1 AS fact_f1,
           e.rel_f1 AS rel_f1,
           e.workflow_accuracy AS workflow_accuracy,
           e.node_coverage AS node_coverage,
           e.details_json AS details_json
    ORDER BY e.timestamp DESC
    """
    try:
        records = run_cypher_query(cypher)
        history = []
        for rec in records:
            item = dict(rec)
            if item.get("details_json"):
                try:
                    item["details"] = json.loads(item["details_json"])
                except Exception:
                    item["details"] = {}
            history.append(item)
        return history
    except Exception as e:
        logger.error("Error fetching evaluation history from Neo4j: %s", e)
        return []
