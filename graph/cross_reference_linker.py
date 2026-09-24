"""
graph/cross_reference_linker.py

Discovers semantic cross-requirement relationships by analyzing requirement text
against the deduplicated catalog of entities extracted from other requirements.
"""

import json

from parser.llm_client import call_ollama, extract_json_block
from graph.validator import ALLOWED_RELATIONS
from config import CROSS_REF_MODEL, CROSS_REF_MODEL_URL

_CATALOG_MAX_CHARS = 4000

_PROMPT_TEMPLATE = """You are cross-referencing ONE requirement against entities already extracted from OTHER requirements in the same document.

REQUIREMENT ID: {item_id}
REQUIREMENT TEXT:
{item_text}

ENTITIES ALREADY IN THE GRAPH, extracted from OTHER requirements (id | type | name):
{catalog}

ONLY these relationship types are allowed: {allowed_types}

Before proposing a relationship, check it against these rules:
- A shared Actor is NOT evidence of a relationship. Two requirements both mentioning "Warehouse Manager" does not mean the requirements are related to each other — only propose a relationship when the TEXT itself connects the two things, not because they happen to share an actor.
- Actors are never the "to" of CREATES, GENERATES, PRODUCES, or UPDATES — actors create/generate/produce/update things, never the reverse. If you're about to write "X CREATES Warehouse Manager" or similar, the direction is backwards; either flip it or don't propose it.
- AUTHENTICATES only makes sense from an Actor or a system-like entity (System, SystemComponent, Service) toward whoever is being authenticated — never from a data record, requirement, or artifact.

Task: identify NEW relationships between concepts in THIS requirement's text and entities in the catalog above. Do NOT restate relationships that only involve concepts within this requirement alone — only relationships that connect this requirement to something extracted from a DIFFERENT requirement.

Return ONLY a JSON array, no other text, no markdown fences. Each entry:
{{"from": "<exact catalog id, or a short new snake_case id naming a concept in THIS requirement's text>", "to": "<exact catalog id>", "type": "<one of the allowed types above>"}}
If there are no such relationships, return [].
"""


def build_entity_catalog(nodes: list) -> list:
    """
    One global catalog entry per already-extracted, already-tagged node
    (graph/structural_linker.py's tag_extraction_source() populates
    "source" with the originating item id — nodes without it were never
    tagged and can't be attributed to an item, so they're excluded).
    Excludes backbone "requirement_*" nodes themselves — those are
    already structurally connected via structural_linker; the LLM
    doesn't need to reason about them here.
    """
    return [
        {
            "id": n["id"],
            "type": n.get("type", "?"),
            "name": n.get("name", n["id"]),
            "source_item_id": n.get("source"),
        }
        for n in nodes
        if not n["id"].startswith("requirement_") and n.get("source")
    ]


def _catalog_text_for_item(catalog: list, exclude_item_id: str, max_chars: int) -> str:
    """Catalog text for one item's prompt — every entry EXCEPT ones
    sourced from this same item (no point cross-referencing a
    requirement against its own already-known entities)."""
    text = ""
    for entry in catalog:
        if entry["source_item_id"] == exclude_item_id:
            continue
        line = f"{entry['id']} | {entry['type']} | {entry['name']} (from {entry['source_item_id']})\n"
        if text and len(text) + len(line) > max_chars:
            break
        text += line
    return text


def find_cross_references(
    items_with_links: list,
    nodes: list,
    model: str = None,
    base_url: str = None,
) -> dict:
    """
    Returns {"relationships": [...], "warnings": [...]}. Never raises —
    a failed/unparseable call for one item just means zero new
    relationships from that item, reported in warnings, never blocking
    the rest of the document (same never-drop discipline as the rest
    of the pipeline, applied to failures rather than data here).

    Validates every returned entry before accepting it:
      - "type" must be in ALLOWED_RELATIONS (graph/validator.py) —
        anything else is dropped and reported, same reasoning as the
        prompt/schema mismatch fixed earlier this project for the main
        extractor.
      - "to" must be an id that actually exists in the catalog — a
        hallucinated id is dropped and reported, not silently accepted
        into the graph.
      - "from" may be a new id naming a concept unique to this item's
        own text (not required to already exist in the catalog).
    """
    model = model or CROSS_REF_MODEL
    base_url = base_url or CROSS_REF_MODEL_URL

    catalog = build_entity_catalog(nodes)
    catalog_ids = {c["id"] for c in catalog}

    relationships = []
    warnings = []

    for item in items_with_links:
        item_id = item.get("id", "")
        catalog_text = _catalog_text_for_item(catalog, item_id, _CATALOG_MAX_CHARS)
        if not catalog_text.strip():
            continue  # nothing extracted from any other item yet to cross-reference against

        prompt = _PROMPT_TEMPLATE.format(
            item_id=item_id,
            item_text=item.get("content", ""),
            catalog=catalog_text,
            allowed_types=", ".join(sorted(ALLOWED_RELATIONS)),
        )

        result = call_ollama(prompt, model=model, base_url=base_url)
        if result["error"]:
            warnings.append(f"{item_id}: cross-reference call failed — {result['error']}")
            continue

        try:
            parsed = extract_json_block(result["raw"])
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array")
        except (ValueError, json.JSONDecodeError) as e:
            warnings.append(f"{item_id}: cross-reference response could not be parsed — {e}")
            continue

        for entry in parsed:
            rel_type = entry.get("type")
            from_id = entry.get("from")
            to_id = entry.get("to")

            if rel_type not in ALLOWED_RELATIONS:
                warnings.append(
                    f"{item_id}: dropped relationship with disallowed type '{rel_type}'"
                )
                continue
            if to_id not in catalog_ids:
                warnings.append(
                    f"{item_id}: dropped relationship — 'to' id '{to_id}' isn't in the "
                    f"catalog (hallucinated reference)"
                )
                continue
            if not from_id:
                warnings.append(f"{item_id}: dropped relationship with empty 'from'")
                continue

            relationships.append({
                "from": from_id,
                "to": to_id,
                "type": rel_type,
                "source": item_id,
            })

    return {"relationships": relationships, "warnings": warnings}