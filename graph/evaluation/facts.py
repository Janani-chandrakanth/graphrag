# graph/evaluation/facts.py
"""Fact extraction module for Evaluation.
Extracts atomic facts (triples / assertions) from requirement text using LLM semantic extraction
with regex sentence parsing fallbacks.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
import streamlit as st
from parser.llm_client import call_ollama
from graph.neo4j_manager import get_stored_documents

logger = logging.getLogger(__name__)

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
