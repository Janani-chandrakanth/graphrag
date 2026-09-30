import json
import re
import requests
from collections import defaultdict
from pydantic import ValidationError

from prompts.extraction_prompts import KG_EXTRACTION_PROMPT, SEQUENCE_EXTRACTION_PROMPT
from graph.schemas import Node, Relationship
from parser.llm_client import call_ollama, extract_json_block
from config import (
    EXTRACTION_MODEL,
    EXTRACTION_MODEL_URL,
    OLLAMA_URL,
    WORKFLOW_MODEL,
    WORKFLOW_MODEL_URL,
)

METADATA_HEADER_KEYWORDS = {
    "version", "revision", "document control", "change description", "author",
    "reviewed by", "approved by", "document history", "version history", "page number"
}

def is_document_metadata_text(text: str) -> tuple[bool, str]:
    """Detect if chunk text is Document Control/Version History table or repeating header/footer artifact."""
    if not text:
        return False, ""
    text_lower = text.lower().strip()
    
    # Check 1: Document Control / Version History table headers
    match_count = sum(1 for kw in METADATA_HEADER_KEYWORDS if kw in text_lower)
    if match_count >= 2:
        return True, "document_control_table"
        
    # Check 2: Header/footer page artifact patterns
    if re.search(r'^(?:forvis\s+mazars\s+\d+|page\s+\d+(?:\s+of\s+\d+)?|document\s+control|version\s+history)$', text_lower):
        return True, "page_header_footer"
        
    if re.search(r'(?:page\s+\d+\s+of\s+\d+|version\s+\d+\.\d+.*author|change\s+description.*version)', text_lower):
        return True, "page_artifact"

    return False, ""


# ── Ontology Mapping Logic ───────────────────────────────────────────────────

# Minimum number of sibling leaf nodes before we even consider
# collapsing them into a canonical concept. Below this, treat
# them as legitimate distinct entities rather than enum values.
MIN_GROUP_SIZE = 3

CANONICALIZATION_PROMPT = """
You are naming a business concept for a knowledge graph.

Below is a parent node and a group of child nodes that are all
connected to it using the same relationship type. These children
look like they may be individual VALUES of one underlying concept
rather than independent business entities.

Parent node: {parent_name} (type: {parent_type})
Relationship type connecting them: {rel_type}
Child nodes: {child_names}

Question: Do these child nodes represent individual values of a
single reusable business concept (for example: Euro/USD/GBP are
values of "Currency Preference", or Penicillin/Aspirin are values
of "Medication")?

If YES, respond with ONLY the canonical concept name in Title Case,
nothing else. Example: Currency Preference

If NO — meaning these children are genuinely distinct entities that
should remain separate nodes (for example: Login, Logout, Register
are distinct features, not values of one concept) — respond with
exactly: NONE

Respond with only the concept name or NONE. No explanation.
"""


def _find_sibling_groups(nodes: list, relationships: list) -> list:
    """
    Find groups of sibling leaf nodes sharing the same parent
    and relationship type.

    Returns:
        list of {
            "parent_id": str,
            "rel_type": str,
            "child_ids": list[str]
        }
    """
    node_by_id = {n["id"]: n for n in nodes}

    # Nodes that have at least one outgoing relationship are not leaves
    has_outgoing = {rel["from"] for rel in relationships}

    # Group children by (parent_id, rel_type)
    groups = defaultdict(list)
    for rel in relationships:
        child_id = rel["to"]
        if child_id in has_outgoing:
            continue  # not a leaf, skip
        if child_id not in node_by_id:
            continue
        groups[(rel["from"], rel["type"])].append(child_id)

    sibling_groups = []
    for (parent_id, rel_type), child_ids in groups.items():
        if len(child_ids) >= MIN_GROUP_SIZE and parent_id in node_by_id:
            sibling_groups.append({
                "parent_id": parent_id,
                "rel_type":  rel_type,
                "child_ids": child_ids
            })

    return sibling_groups


def _ask_llm_for_concept_name(parent_node: dict, rel_type: str, child_nodes: list) -> str:
    """
    Single small LLM call to name the canonical concept for a
    detected sibling group, or return "NONE" if they should stay
    as distinct nodes.
    """
    prompt = CANONICALIZATION_PROMPT.format(
        parent_name=parent_node.get("name", parent_node["id"]),
        parent_type=parent_node.get("type", "Unknown"),
        rel_type=rel_type,
        child_names=", ".join(c.get("name", c["id"]) for c in child_nodes)
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  "llama3.1:latest",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "seed": 42}
            },
            proxies={"http": None, "https": None},
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()
        return answer
    except Exception:
        # If the LLM call fails, fail safe — don't collapse anything
        return "NONE"


def _normalize_id(value: str) -> str:
    return (
        value.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .strip()
    )


def apply_ontology_mapping(nodes: list, relationships: list) -> dict:
    """
    Detect structurally-isolated sibling literal groups and
    collapse each into one canonical concept node, generically,
    using a structural test plus a single LLM naming call —
    no hardcoded domain vocabulary.

    Args:
        nodes:         list of node dicts (already Pydantic-validated)
        relationships: list of relationship dicts

    Returns:
        {
            "nodes": list,
            "relationships": list,
            "mappings_applied": list   # log for debugging/UI display
        }
    """
    node_by_id = {n["id"]: n for n in nodes}

    sibling_groups = _find_sibling_groups(nodes, relationships)

    if not sibling_groups:
        return {
            "nodes":            nodes,
            "relationships":    relationships,
            "mappings_applied": []
        }

    id_redirect_map   = {}
    canonical_nodes    = {}
    mappings_applied  = []

    for group in sibling_groups:
        parent_node = node_by_id.get(group["parent_id"])
        if not parent_node:
            continue

        child_nodes = [
            node_by_id[cid] for cid in group["child_ids"]
            if cid in node_by_id
        ]
        if len(child_nodes) < MIN_GROUP_SIZE:
            continue

        concept_name = _ask_llm_for_concept_name(
            parent_node, group["rel_type"], child_nodes
        )

        if concept_name.upper() == "NONE" or not concept_name:
            continue  # LLM says these are genuinely distinct — leave as is

        canonical_id = _normalize_id(concept_name)

        if canonical_id not in canonical_nodes:
            canonical_nodes[canonical_id] = {
                "id":          canonical_id,
                "type":        "Feature",
                "name":        concept_name,
                "description": (
                    f"Canonical concept collapsing "
                    f"{len(child_nodes)} individual values"
                ),
                "attributes":  {},
                "aliases":     []
            }

        for child in child_nodes:
            id_redirect_map[child["id"]] = canonical_id
            alias_name = child.get("name", child["id"])
            if alias_name not in canonical_nodes[canonical_id]["aliases"]:
                canonical_nodes[canonical_id]["aliases"].append(alias_name)

            mappings_applied.append({
                "original_id":    child["id"],
                "original_name":  alias_name,
                "mapped_to_id":   canonical_id,
                "mapped_to_name": concept_name,
                "parent_id":      group["parent_id"],
                "rel_type":       group["rel_type"]
            })

    if not id_redirect_map:
        return {
            "nodes":            nodes,
            "relationships":    relationships,
            "mappings_applied": []
        }

    # Remove the now-redundant literal child nodes, keep everything else
    final_nodes = [
        n for n in nodes if n["id"] not in id_redirect_map
    ] + list(canonical_nodes.values())

    # Rewrite relationships: redirect any from/to pointing at a
    # collapsed literal node to point at its canonical node instead
    final_relationships = []
    seen_rel_keys = set()

    for rel in relationships:
        new_from = id_redirect_map.get(rel["from"], rel["from"])
        new_to   = id_redirect_map.get(rel["to"], rel["to"])

        if new_from == new_to:
            continue  # drop self-loop created by collapsing

        key = (new_from, new_to, rel["type"])
        if key in seen_rel_keys:
            continue
        seen_rel_keys.add(key)

        new_rel = dict(rel)
        new_rel["from"] = new_from
        new_rel["to"]   = new_to
        final_relationships.append(new_rel)

    return {
        "nodes":            final_nodes,
        "relationships":    final_relationships,
        "mappings_applied": mappings_applied
    }


# ── Procedural-content guard & Sequence Extraction ───────────────────────────
# Pattern signals that a chunk describes a sequential procedure. Only chunks
# that match at least one signal go through the expensive LLM sequence call.
_PROCEDURAL_SIGNALS = re.compile(
    r"""
    (?:^|\n)\s*\d+[\.\)]\s+\w          # numbered list  "1. Login"
    | (?:^|\n)\s*[-*•]\s+\w            # bullet list
    | \b(?:then|next|after\s+that|following(?:ly)?|subsequently|
           finally|first(?:ly)?|second(?:ly)?|third(?:ly)?|
           step\s+\d|proceed\s+to|upon\s+success)\b
    | \b(?:the\s+(?:user|system|admin|customer|actor)\s+
           (?:shall|must|will|should|can|clicks?|enters?|
            submits?|selects?|navigates?|uploads?|triggers?))\b
    | (?:→|->|==>)                      # explicit flow arrows
    | \bAC[-_]?\d+\b                    # Acceptance Criteria IDs
    | \b(?:given|when|then)\b           # Gherkin keywords
    | (?:^|\n)\s*(?:Step|Phase|Stage)\s+\d+
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Patterns that strongly indicate NON-procedural text.
_NON_PROCEDURAL_SIGNALS = re.compile(
    r"""
    \b(?:definition|glossary|abbreviation|terminology|legend)\b
    | \b(?:non[-\s]?functional|NFR|performance\s+requirement|
           scalability|availability|reliability)\b
    | (?:^|\n)\s*(?:Table\s+of\s+Contents|Revision\s+History|
                    Document\s+Control|Appendix\s+[A-Z]|
                    References?|Bibliography)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_procedural_chunk(text: str) -> bool:
    """
    Returns True if *text* is likely to contain a sequential procedure worth
    sending to the sequence LLM. Uses fast regex heuristics — no LLM call.
    """
    if not text or len(text.strip()) < 40:
        return False
    # Hard reject obvious non-procedural sections
    if _NON_PROCEDURAL_SIGNALS.search(text):
        return False
    # Accept if any procedural signal present
    return bool(_PROCEDURAL_SIGNALS.search(text))


def extract_workflow_sequence(requirements_text: str, extracted_nodes: list) -> dict:
    """
    Analyzes the text and the extracted nodes to detect one or more sequential
    workflows. Supports multiple per-actor sequences (e.g., separate chains for
    User, Admin, and Restaurant Owner in the same document).

    Returns:
    {
        "sequences": [
            {
                "actor_id": str | None,
                "flow_entry_id": str,
                "sequence_edges": [{"from": ..., "to": ..., "type": "LEADS_TO"}, ...]
            },
            ...
        ],
        "error": str | None
    }
    """
    if not extracted_nodes:
        return {"sequences": [], "error": None}

    # Fast path: skip LLM for clearly non-procedural chunks
    if not is_procedural_chunk(requirements_text):
        return {"sequences": [], "error": None}

    RELEVANT_TYPES = {
        "Actor", "Feature", "Action", "Requirement", "BusinessProcess",
        "Workflow", "Event", "State", "SystemComponent"
    }
    filtered_nodes = [
        n for n in extracted_nodes
        if n.get("type") in RELEVANT_TYPES
    ]
    # Fall back to all nodes if filtering leaves nothing useful
    nodes_for_prompt = filtered_nodes if len(filtered_nodes) >= 2 else extracted_nodes

    prompt = SEQUENCE_EXTRACTION_PROMPT.replace(
        "{requirements_text}", requirements_text
    ).replace(
        "{extracted_nodes}", json.dumps(
            [{"id": n["id"], "name": n["name"], "type": n.get("type", "")}
             for n in nodes_for_prompt],
            indent=2
        )
    )

    # Increased timeout and retry logic for robustness against occasional server delays
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        result = call_ollama(
            prompt,
            timeout=300,  # extended timeout to 5 minutes
            num_ctx=8192,
            model=WORKFLOW_MODEL,
            base_url=WORKFLOW_MODEL_URL
        )
        if not result["error"]:
            break
        # If timeout error, retry; otherwise break immediately
        if "timed out" in result["error"].lower():
            if attempt < max_retries:
                print(f"Retry {attempt}/{max_retries} after timeout...")
                continue
        # Non-retryable error or max attempts reached
        break

    if result["error"]:
        print(f"Workflow sequence extraction error: {result['error']}")
        return {"sequences": [], "error": result["error"]}

    try:
        parsed = extract_json_block(result["raw"])
        print("\n--- SEQUENCE EXTRACTOR LLM OUTPUT ---")
        print(json.dumps(parsed, indent=2))
        print("---------------------------------------\n")

        # The prompt asks for an array at the top level
        if isinstance(parsed, list):
            sequences = parsed
        elif isinstance(parsed, dict):
            # Graceful fallback: old single-sequence schema
            if parsed.get("is_sequence") and parsed.get("sequence_edges"):
                sequences = [{
                    "actor_id": None,
                    "flow_entry_id": parsed.get("flow_entry_id"),
                    "sequence_edges": parsed.get("sequence_edges", [])
                }]
            else:
                sequences = []
        else:
            sequences = []

        # Validate: filter out sequences with no edges or missing flow_entry_id
        valid_node_ids = {n["id"] for n in extracted_nodes}
        clean_sequences = []
        for seq in sequences:
            entry = seq.get("flow_entry_id")
            edges = seq.get("sequence_edges", [])
            if not entry or not edges:
                continue
            if entry not in valid_node_ids:
                print(f"  [WARN] flow_entry_id '{entry}' not in extracted nodes — skipping sequence")
                continue
            # Filter out edges referencing unknown node IDs
            valid_edges = [
                e for e in edges
                if e.get("from") in valid_node_ids and e.get("to") in valid_node_ids
            ]
            if not valid_edges:
                continue
            clean_sequences.append({
                "actor_id": seq.get("actor_id"),
                "flow_entry_id": entry,
                "sequence_edges": valid_edges
            })

        return {"sequences": clean_sequences, "error": None}

    except Exception as e:
        print(f"Error parsing sequence extraction JSON: {e}")
        return {"sequences": [], "error": f"Error parsing JSON: {e}"}


# ── Entity Extractor Main ────────────────────────────────────────────────────

def extract_entities(requirement_text: str, prior_context: str = None,
                      continue_same_item: bool = False) -> dict:
    """
    Args:
        requirement_text: the current chunk's text — extraction happens from this only.
        prior_context: optional immediately preceding chunk's text for continuity.
        continue_same_item: True when split across chunk boundary due to size.
    """
    if prior_context and continue_same_item:
        continuity_section = (
            "\n=========================================================\n"
            "CONTINUITY CONTEXT — SAME ITEM, SPLIT ONLY BY LENGTH\n"
            "=========================================================\n\n"
            "The following is the immediately preceding part of the SAME "
            "requirement/User Story/Test Case as the text in INPUT "
            "REQUIREMENTS below — it was split into two separate chunks "
            "purely because the combined text was too long for one call, "
            "not because it describes a different item. Treat INPUT "
            "REQUIREMENTS as a direct continuation of this text.\n\n"
            f"{prior_context}\n\n"
            "Rules for using this section:\n"
            "- Do NOT create any node from this section — every node "
            "must still come from INPUT REQUIREMENTS only.\n"
            "- IF an entity in INPUT REQUIREMENTS refers to the same "
            "real-world thing as something named here, reuse the exact "
            "same lowercase snake_case id it would get from this text, "
            "so it is recognized as the same entity.\n"
            "- You MAY create AT MOST ONE relationship whose \"from\" is "
            "the id of the LAST step/action/node described in this "
            "section and whose \"to\" is the id of the FIRST step/action/"
            "node in INPUT REQUIREMENTS, using the appropriate FLOW "
            "relationship type (LEADS_TO/TRIGGERS/CAUSES/...) — ONLY if "
            "this section clearly described a procedure that continues "
            "into INPUT REQUIREMENTS. This is the one exception to the "
            "\"no relationships from context\" rule; do not create any "
            "OTHER relationship referencing a node from this section.\n"
            "- If you add this one continuation relationship, and the "
            "step it points TO has a \"sequence\" attribute, continue the "
            "SAME numbering forward (i.e. if this section's last visible "
            "step was sequence N, INPUT REQUIREMENTS' first step is "
            "N+1) instead of restarting the count at 1.\n"
        )
        prompt = KG_EXTRACTION_PROMPT.replace(
            "{requirements_text}",
            requirement_text
        ).replace(
            "=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "=========================================================",
            continuity_section +
            "\n=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "=========================================================\n"
        )
    elif prior_context:
        continuity_section = (
            "\n=========================================================\n"
            "CONTINUITY CONTEXT (OPTIONAL — READ, DO NOT EXTRACT FROM)\n"
            "=========================================================\n\n"
            "The following is the immediately preceding requirement in "
            "this document. It is provided ONLY so you can recognize "
            "whether the requirement below (in INPUT REQUIREMENTS) refers "
            "to the same real-world entity as something already named "
            "here — e.g. \"the dropdown\" in both.\n\n"
            f"{prior_context}\n\n"
            "Rules for using this section:\n"
            "- Do NOT create any node or relationship from this section.\n"
            "- Do NOT extract anything that appears only here and not in "
            "INPUT REQUIREMENTS.\n"
            "- IF an entity in INPUT REQUIREMENTS clearly refers to the "
            "same real-world thing as something named here, reuse the "
            "exact same lowercase snake_case id for it that you would "
            "assign based on this text, so it is recognized as the same "
            "entity.\n"
            "- If nothing in INPUT REQUIREMENTS continues anything here, "
            "ignore this section completely.\n"
        )
        prompt = KG_EXTRACTION_PROMPT.replace(
            "{requirements_text}",
            requirement_text
        ).replace(
            "=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "=========================================================",
            continuity_section +
            "\n=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "========================================================="
        )
    else:
        prompt = KG_EXTRACTION_PROMPT.replace(
            "{requirements_text}",
            requirement_text
        )

    result = call_ollama(
        prompt,
        timeout=120,
        num_ctx=8192,
        model=EXTRACTION_MODEL,
        base_url=EXTRACTION_MODEL_URL,
    )
    if result["error"]:
        return {
            "nodes": [], "relationships": [],
            "error": result["error"],
            "raw_output": ""
        }
    output = result["raw"]

    try:
        start = output.find("{")

        if start == -1:
            return {
                "nodes": [], "relationships": [],
                "error": "LLM did not return any JSON block",
                "raw_output": output
            }

        decoder = json.JSONDecoder()
        raw_data, idx = decoder.raw_decode(output, start)
        json_str = output[start:idx]

        # Check if current chunk text is document metadata
        is_meta, meta_reason = is_document_metadata_text(requirement_text)

        # Validate NODES one by one
        valid_nodes   = []
        skipped_nodes = []
        for node in raw_data.get("nodes", []):
            try:
                validated = Node(**node)
                ndict = validated.model_dump(exclude_none=True)
                nname_lower = (ndict.get("name") or "").lower().strip()
                nid_lower = (ndict.get("id") or "").lower().strip()
                
                # Tag metadata nodes if chunk is metadata or node matches metadata patterns
                if is_meta or nname_lower in ("parichita", "forvis mazars 5", "document control", "version history") or re.search(r'^(page\s+\d+|forvis\s+mazars|\d+\.\d+.*author)', nname_lower):
                    if "author" in nname_lower or "parichita" in nname_lower:
                        ndict["type"] = "Author"
                    elif "page" in nname_lower or "mazars" in nname_lower:
                        ndict["type"] = "PageArtifact"
                    else:
                        ndict["type"] = "DocumentMetadata"

                valid_nodes.append(ndict)
            except ValidationError as e:
                skipped_nodes.append({"input": node, "reason": str(e.errors())})

        # Validate RELATIONSHIPS one by one
        valid_relationships   = []
        skipped_relationships = []
        for rel in raw_data.get("relationships", []):
            try:
                validated = Relationship(**rel)
                valid_relationships.append(
                    validated.model_dump(by_alias=True, exclude_none=True)
                )
            except ValidationError:
                skipped_relationships.append(rel)

        # ── Ontology Mapping ──────────────────────────────
        mapping_result = apply_ontology_mapping(
            valid_nodes, valid_relationships
        )
        valid_nodes = mapping_result["nodes"]
        valid_relationships = mapping_result["relationships"]
        # ── Phase 2: Workflow Sequence Extraction ─────────
        sequence_data = extract_workflow_sequence(requirement_text, valid_nodes)
        sequences = sequence_data.get("sequences", [])

        if sequences:
            actor_ids = {n["id"] for n in valid_nodes if n.get("type") == "Actor"}

            all_sequence_node_ids = set()
            actor_to_entry = {}

            for seq in sequences:
                entry = seq.get("flow_entry_id")
                edges = seq.get("sequence_edges", [])
                actor_id = seq.get("actor_id")

                if entry:
                    for node in valid_nodes:
                        if node["id"] == entry:
                            if "attributes" not in node or node["attributes"] is None:
                                node["attributes"] = {}
                            node["attributes"]["is_flow_entry"] = "true"
                            break

                valid_relationships.extend(edges)

                for edge in edges:
                    all_sequence_node_ids.add(edge["from"])
                    all_sequence_node_ids.add(edge["to"])

                if actor_id and entry:
                    actor_to_entry[actor_id] = entry

            filtered_relationships = []
            for rel in valid_relationships:
                frm = rel.get("from")
                to  = rel.get("to")

                if frm in actor_ids and to in all_sequence_node_ids:
                    this_actors_entry = actor_to_entry.get(frm)
                    if this_actors_entry is None:
                        is_any_entry = any(
                            s.get("flow_entry_id") == to for s in sequences
                        )
                        if not is_any_entry:
                            continue
                    else:
                        if to != this_actors_entry:
                            continue

                filtered_relationships.append(rel)

            valid_relationships = filtered_relationships

        warnings = []
        if sequence_data.get("error"):
            warnings.append(f"Workflow sequence extraction failed: {sequence_data['error']}")

        if skipped_nodes:
            warnings.append(
                f"{len(skipped_nodes)} node(s) skipped: "
                f"{[s['input'].get('id','?') for s in skipped_nodes]}"
            )
        if skipped_relationships:
            invalid_types = list({r.get('type','?') for r in skipped_relationships})
            warnings.append(
                f"{len(skipped_relationships)} relationship(s) skipped "
                f"with invalid types: {invalid_types}"
            )
        if mapping_result["mappings_applied"]:
            warnings.append(
                f"{len(mapping_result['mappings_applied'])} node(s) "
                f"mapped to ontology concepts"
            )

        return {
            "nodes":                 valid_nodes,
            "relationships":         valid_relationships,
            "raw_output":            json_str,
            "skipped_nodes":         skipped_nodes,
            "skipped_relationships": skipped_relationships,
            "ontology_mappings":     mapping_result["mappings_applied"],
            "error": "; ".join(warnings) if warnings else None
        }

    except json.JSONDecodeError as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"JSON parse error: {str(e)}",
            "raw_output": output
        }
    except Exception as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"Unexpected error: {str(e)}",
            "raw_output": output
        }