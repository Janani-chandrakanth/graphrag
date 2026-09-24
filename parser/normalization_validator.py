"""
parser/normalization_validator.py

Validates normalized document items against deterministic schema rules,
categorizing items into valid_items and review_queue buckets.
"""

from collections import Counter

from parser.item_patterns import match_item_id

MIN_CONTENT_LENGTH = 10

# Document-level: if more than this fraction of blocks couldn't be
# normalized into an item at all, that's a signal the doc type detection
# or template choice may be off, not just noise in one or two blocks.
UNMATCHED_RATIO_WARNING_THRESHOLD = 0.3


def _check_item(item: dict, template: dict, id_counts: Counter) -> list:
    """Run all rules against one item. Returns a list of failure reason
    strings — empty list means the item is valid."""
    reasons = []

    if item.get("matched_by") == "llm_gap_fill":
        reasons.append(
            "LLM gap-filled — not grounded in an explicit ID in the source "
            "document, requires human confirmation."
        )

    item_id = item.get("id", "")
    if id_counts.get(item_id, 0) > 1:
        reasons.append(
            f"Duplicate ID '{item_id}' — used by {id_counts[item_id]} items "
            f"in this document."
        )

    family = item.get("family")
    if family and family not in template.get("id_families", []):
        reasons.append(
            f"ID family '{family}' is not native to the detected doc type "
            f"'{template.get('description', '')}' (expected one of "
            f"{template.get('id_families', [])})."
        )

    content = (item.get("content") or "").strip()
    if len(content) < MIN_CONTENT_LENGTH:
        reasons.append(
            f"Content is only {len(content)} character(s) — too short to "
            f"plausibly be a real item."
        )

    if not item_id.startswith("GEN-"):
        matched_family, _ = match_item_id(item_id)
        if matched_family is None:
            reasons.append(
                f"ID '{item_id}' doesn't match any known family pattern and "
                f"isn't an LLM-generated placeholder."
            )

    return reasons


def validate_normalized_document(normalized: dict) -> dict:
    """
    Validate a normalize_document() result.

    Args:
        normalized: the dict returned by template_normalizer.normalize_document()
                    (must have "items" and "template" keys at minimum)

    Returns:
        {
            "doc_type":         str,
            "source_filename":  str,
            "valid_items":      [item, ...],   # passed every rule
            "review_queue":     [{"item": item, "reasons": [str, ...]}, ...],
            "document_warnings":[str, ...],    # document-level signals
            "summary":          {"total_items": int, "valid": int, "needs_review": int},
        }
    """
    template = normalized.get("template", {})
    items = normalized.get("items", [])
    id_counts = Counter(item.get("id", "") for item in items)

    valid_items = []
    review_queue = []

    for item in items:
        reasons = _check_item(item, template, id_counts)
        if reasons:
            review_queue.append({"item": item, "reasons": reasons})
        else:
            valid_items.append(item)

    document_warnings = []
    unmatched_count = len(normalized.get("unmatched_blocks", []))
    total_blocks = len(items) + unmatched_count
    if total_blocks and (unmatched_count / total_blocks) > UNMATCHED_RATIO_WARNING_THRESHOLD:
        document_warnings.append(
            f"{unmatched_count}/{total_blocks} blocks could not be normalized "
            f"into an item at all — the detected doc type or template may not "
            f"fit this document well."
        )

    return {
        "doc_type": normalized.get("doc_type"),
        "source_filename": normalized.get("source_filename"),
        "valid_items": valid_items,
        "review_queue": review_queue,
        "document_warnings": document_warnings,
        "summary": {
            "total_items": len(items),
            "valid": len(valid_items),
            "needs_review": len(review_queue),
        },
    }
