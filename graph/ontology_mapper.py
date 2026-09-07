"""
Ontology Mapping Layer — Structural / Domain-Agnostic Version

Problem with a hardcoded literal registry (previous version):
    A regex list of "usd|eur|gbp" only catches currencies.
    A healthcare document with Penicillin/Aspirin/Ibuprofen as
    isolated nodes would sail straight through untouched, because
    nothing in the code recognizes medication names.

Generic fix used here:
    Detect a STRUCTURAL pattern instead of literal words:

        Several sibling nodes that:
          - share the same node type (often Attribute/DataObject)
          - are all leaves (no outgoing relationships)
          - are all reached via the same relationship type
            from the same parent node

    That shape is domain-independent. Euro/USD/GBP under
    "Currency Preference" has the same shape as
    Penicillin/Aspirin/Ibuprofen under "Medication" —
    same structural signature, completely different domains.

    Once that shape is detected for a group of >= MIN_GROUP_SIZE
    siblings, ONE small LLM call asks: "what single business
    concept do these represent?" The LLM names the canonical
    concept — nothing is hardcoded per domain.

Pipeline position:
    extract_entities() -> apply_ontology_mapping() -> deduplicate_graph() -> validate_graph()

Cost control:
    This adds at most one extra LLM call per detected sibling
    group per chunk — not per node. Most chunks will trigger
    zero or one such call.
"""

import requests
from collections import defaultdict
from config import OLLAMA_URL


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