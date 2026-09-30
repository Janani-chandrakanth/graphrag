"""
parser/requirement_linker.py

Links validated document items by scanning content for exact requirement ID references
and mapping family pairs to relationship types.
"""

import re

from parser.models import find_id_mentions

# ── Family-pair -> relation type ────────────────────────────────────────
# Keyed by (source_family, target_family) — i.e. the family of the item
# whose content contains the mention, and the family of the item it
# mentions. Unlisted pairs (including same-family pairs, e.g. FR -> FR)
# fall back to the generic RELATES_TO.
RELATION_RULES = {
    ("US", "TC"): "VERIFIED_BY",   ("TC", "US"): "VERIFIES",
    ("FR", "TC"): "VERIFIED_BY",   ("TC", "FR"): "VERIFIES",
    ("NFR", "TC"): "VERIFIED_BY",  ("TC", "NFR"): "VERIFIES",
    ("UC", "TC"): "VERIFIED_BY",   ("TC", "UC"): "VERIFIES",
    ("BR", "TC"): "VERIFIED_BY",   ("TC", "BR"): "VERIFIES",

    ("FR", "BR"): "REALIZES",      ("BR", "FR"): "REALIZED_BY",
    ("NFR", "BR"): "REALIZES",     ("BR", "NFR"): "REALIZED_BY",
    ("US", "BR"): "REALIZES",      ("BR", "US"): "REALIZED_BY",
    ("FR", "UC"): "REALIZES",      ("UC", "FR"): "REALIZED_BY",

    ("US", "FR"): "RELATED_TO",    ("FR", "US"): "RELATED_TO",
    ("US", "NFR"): "RELATED_TO",   ("NFR", "US"): "RELATED_TO",

    ("REQ", "TC"): "VERIFIED_BY",  ("TC", "REQ"): "VERIFIES",
    ("REQ", "BR"): "REALIZES",     ("BR", "REQ"): "REALIZED_BY",
}
DEFAULT_RELATION = "RELATES_TO"


def _canonical(id_str: str) -> str:
    """Normalize an ID for comparison purposes only (e.g. 'FR-001',
    'fr001', 'FR 001' all compare equal). Output ordering/format for
    links always uses the *stored* item ID, never this canonical form."""
    return re.sub(r'[^A-Za-z0-9]', '', id_str or '').upper()


def link_requirements(valid_items: list) -> dict:
    """
    Cross-link valid_items by exact ID reference.

    Args:
        valid_items: the "valid_items" list from
                      normalization_validator.validate_normalized_document()
                      (each item must have "id", "family", "content")

    Returns:
        {
            "links": [
                {"source_id", "source_family", "target_id", "target_family",
                 "link_type", "evidence"},
                ...
            ],
            "broken_references": [
                {"source_id", "referenced_text", "reason"}, ...
            ],
            "items_with_links": [ item + "linked_ids": [id, ...] ],
            "summary": {
                "total_items", "total_links", "broken_references",
                "items_with_at_least_one_link",
            },
        }
    """
    id_lookup = {}          # canonical(id) -> (stored_id, family)
    for item in valid_items:
        id_lookup[_canonical(item.get("id", ""))] = (
            item.get("id"), item.get("family")
        )

    links = []
    broken_references = []
    linked_ids_by_item = {item.get("id"): set() for item in valid_items}

    for item in valid_items:
        source_id = item.get("id")
        source_family = item.get("family")
        content = item.get("content") or ""
        self_canonical = _canonical(source_id)

        seen_targets_this_item = set()  # dedupe repeated mentions of same target

        for mention_family, raw_text, _pos in find_id_mentions(content):
            mention_canonical = _canonical(raw_text)

            if mention_canonical == self_canonical:
                continue  # the item's own ID appearing in its own content

            resolved = id_lookup.get(mention_canonical)

            if resolved is None:
                key = (source_id, mention_canonical)
                if key in seen_targets_this_item:
                    continue
                seen_targets_this_item.add(key)
                broken_references.append({
                    "source_id": source_id,
                    "referenced_text": raw_text,
                    "reason": (
                        f"'{raw_text}' looks like an ID reference but no "
                        f"item with that ID exists in this document's "
                        f"validated items."
                    ),
                })
                continue

            target_id, target_family = resolved
            if (source_id, target_id) in seen_targets_this_item:
                continue
            seen_targets_this_item.add((source_id, target_id))

            if source_family == target_family:
                # A same-family exact-ID mention (e.g. a BR item's text
                # saying "as mentioned in BR_STAY_6") is a cross-reference
                # between two instances of the same kind of item, not the
                # cross-family "realizes/verifies" relationships
                # RELATION_RULES covers below — those are keyed by
                # (source_family, target_family) pairs, so same-family
                # pairs would otherwise silently fall through to the
                # generic RELATES_TO default and lose that distinction.
                link_type = "REFERENCES"
            else:
                link_type = RELATION_RULES.get(
                    (source_family, target_family), DEFAULT_RELATION
                )
            links.append({
                "source_id": source_id,
                "source_family": source_family,
                "target_id": target_id,
                "target_family": target_family,
                "link_type": link_type,
                "evidence": raw_text,
            })
            linked_ids_by_item[source_id].add(target_id)

    items_with_links = []
    for item in valid_items:
        enriched = dict(item)
        enriched["linked_ids"] = sorted(linked_ids_by_item.get(item.get("id"), set()))
        items_with_links.append(enriched)

    items_with_at_least_one_link = sum(
        1 for ids in linked_ids_by_item.values() if ids
    )

    return {
        "links": links,
        "broken_references": broken_references,
        "items_with_links": items_with_links,
        "summary": {
            "total_items": len(valid_items),
            "total_links": len(links),
            "broken_references": len(broken_references),
            "items_with_at_least_one_link": items_with_at_least_one_link,
        },
    }