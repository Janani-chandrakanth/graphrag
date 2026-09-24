"""
chunking/chunker.py

Auto-detecting structural and semantic text chunker for requirement items,
user stories, test cases, and generic document sections.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Chunk size limits ──────────────────────────────────────
# MAX_CHUNK_SIZE was 1000 chars. That's small enough that almost any
# User Story with a real acceptance-criteria list (10-15 bullets is
# routine — see the SharkNinja login example, ~1900 chars) blows past
# it and gets cut into 2+ chunks by the LangChain splitter below,
# purely on a character count that has nothing to do with where one
# step ends and the next begins. Every extraction is one independent
# LLM call (see graph/entity_extractor.py) with its own "sequence"
# numbering that restarts at 1 — so a mid-story character split is
# exactly what turns "step 8 leads to step 9" into two disconnected
# little chains that never mention each other, which is the single
# biggest cause of the hub-and-spoke / lost-sequence pattern for User
# Stories specifically (see graph/entity_extractor.py's
# continue_same_item parameter for the other half of this fix — what
# happens on the rare split that's still unavoidable at this size).
#
# 3500 chars is comfortably inside the extraction model's num_ctx=8192
# token budget (prompts/kg_extraction_prompt.py itself is large, but
# even so this leaves several thousand tokens of headroom for input +
# JSON output) and covers the large majority of real single-item
# requirement/User-Story/Test-Case text without splitting at all.
MAX_CHUNK_SIZE  = 3500   # chars — LangChain splits anything larger
CHUNK_OVERLAP   = 100    # chars overlap between sub-chunks


# ── Pattern Registry ───────────────────────────────────────
# Each entry: (pattern_name, regex_to_split_on, boundary_lookahead)
# Order matters — more specific patterns first

PATTERN_REGISTRY = [

    # Functional Requirements
    # FR-001, FR001, F-001, F001
    (
        "FR",
        r'((?:FR|F)-?\d{3,}:.*?)(?=(?:FR|F)-?\d{3,}:|(?:NFR|NR|NF)-?\d{3,}:|(?:BR|BRD|BU)-?\d{3,}:|$)',
        "Functional Requirement"
    ),

    # Non-Functional Requirements
    # NFR-001, NFR001, NR-001, NF-001
    (
        "NFR",
        r'((?:NFR|NR|NF)-?\d{3,}:.*?)(?=(?:NFR|NR|NF)-?\d{3,}:|(?:BR|BRD)-?\d{3,}:|$)',
        "Non-Functional Requirement"
    ),

    # Business Rules
    # BR-001, BR001, BRD-001, BU-001
    (
        "BR",
        r'((?:BR|BRD|BU)-?\d{3,}.*?)(?=(?:BR|BRD|BU)-?\d{3,}|$)',
        "Business Rule"
    ),

    # Test Cases
    # TC-001, TC001, TEST-001, TS-001
    (
        "TC",
        r'((?:TC|TEST|TS)-?\d{3,}:.*?)(?=(?:TC|TEST|TS)-?\d{3,}:|$)',
        "Test Case"
    ),

    # Use Cases
    # UC-001, UC001, USE-001
    (
        "UC",
        r'((?:UC|USE)-?\d{3,}:.*?)(?=(?:UC|USE)-?\d{3,}:|$)',
        "Use Case"
    ),

    # User Stories — numbered format
    # US-001, US001, UST-001
    (
        "US_NUMBERED",
        r'((?:US|UST)-?\d{3,}:.*?)(?=(?:US|UST)-?\d{3,}:|$)',
        "User Story"
    ),

    # User Stories — prose format
    # "User Story 1:", "Story 1:", "User Story:"
    (
        "US_PROSE",
        r'((?:User Story|Story)\s+\d+:.*?)(?=(?:User Story|Story)\s+\d+:|$)',
        "User Story"
    ),

    # Requirements without prefix
    # REQ-001, RQ-001, R-001
    (
        "REQ",
        r'((?:REQ|RQ|R)-?\d{3,}:.*?)(?=(?:REQ|RQ|R)-?\d{3,}:|$)',
        "Requirement"
    ),

    # Numbered sections
    # 1.1 Title, 2.3.1 Title
    (
        "NUMBERED_SECTION",
        r'(\d+\.\d+(?:\.\d+)?\s+[A-Z][^\n]*\n.*?)(?=\d+\.\d+(?:\.\d+)?\s+[A-Z]|$)',
        "Section"
    ),
]


# ── Splitter for oversized chunks ──────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


def detect_document_type(text: str) -> list:
    """
    Scan text and detect which document types are present.

    Returns list of detected pattern names e.g. ["FR", "NFR", "BR"]
    """
    detected = []

    for pattern_name, pattern, label in PATTERN_REGISTRY:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            detected.append({
                "pattern_name": pattern_name,
                "label":        label,
                "count":        len(matches)
            })

    return detected


def split_by_pattern(text: str, pattern: str) -> list:
    """
    Split text using a regex pattern.
    Returns list of non-empty stripped chunks.
    """
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def split_large_chunks(chunks: list) -> list:
    """
    For any chunk over MAX_CHUNK_SIZE, use LangChain to split further.
    Smaller chunks stay as-is.
    """
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > MAX_CHUNK_SIZE:
            sub_chunks = splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    return final_chunks


def create_chunks(text: str) -> list:
    """
    Main chunking function.

    Auto-detects document type and splits accordingly.
    Falls back to LangChain semantic chunking if no patterns found.

    Args:
        text: full document text

    Returns:
        list of chunk strings + detection metadata in session info
    """
    all_chunks   = []
    detected     = detect_document_type(text)

    if detected:
        # Pattern-based chunking — split by each detected type
        for item in detected:
            pattern_name = item["pattern_name"]

            # Find the matching pattern from registry
            for reg_name, pattern, label in PATTERN_REGISTRY:
                if reg_name == pattern_name:
                    chunks = split_by_pattern(text, pattern)
                    all_chunks.extend(chunks)
                    break

        # Remove duplicates while preserving order
        seen       = set()
        unique_chunks = []
        for chunk in all_chunks:
            key = chunk[:100]  # use first 100 chars as key
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
        all_chunks = unique_chunks

    else:
        # No patterns detected — LangChain semantic fallback
        # Works for any free-form document
        all_chunks = splitter.split_text(text)

    # Final pass — split anything still too large
    all_chunks = split_large_chunks(all_chunks)

    return all_chunks


def create_chunks_from_items(items_with_links: list) -> list:
    """
    Structural Chunker, item-aware entry point.

    Pipeline position: Rule-based Requirement Linker (items_with_links)
    -> HERE -> Embedding Generator & LLM Entity/Relation Extractor.

    Unlike create_chunks(text), which re-parses flat/markdown text with
    regex from scratch, this consumes the already-structured item list
    coming out of the linker — one chunk per item, by construction, with
    no re-detection needed since Document Type Detector / Template
    Normalizer / Normalization Validator already did that work upstream.
    This is what lets id/family/linked_ids metadata survive into the
    chunks instead of being thrown away and semantically re-discovered
    later by the LLM extractor.

    Each item becomes exactly one chunk unless its content exceeds
    MAX_CHUNK_SIZE, in which case it's split into "<id>#1", "<id>#2", ...
    sub-chunks via the same LangChain splitter create_chunks() uses —
    every sub-chunk carries the same metadata (id/family/linked_ids),
    since the link is a property of the requirement, not of which
    character range a given sub-chunk happens to cover.

    Args:
        items_with_links: the "items_with_links" list from
                           requirement_linker.link_requirements()
                           (each item needs "id", "family", "content";
                           "linked_ids" defaults to [] if absent so this
                           also works directly on valid_items pre-linking)

    Returns:
        [
            {
                "chunk_id": str,       # item id, or "<id>#N" if split
                "text": str,           # the chunk's actual content
                "item_id": str,        # always the parent item's id,
                                       # even for split sub-chunks
                "family": str | None,
                "linked_ids": [str, ...],
                "is_split": bool,      # True if this is a sub-chunk
            },
            ...
        ]
    """
    result = []

    for item in items_with_links:
        item_id = item.get("id")
        family = item.get("family")
        linked_ids = item.get("linked_ids", [])
        content = (item.get("content") or "").strip()

        if not content:
            # An item with genuinely empty content produces an empty
            # chunk, which produces an empty/degenerate embedding
            # request, which is what crashes ChromaDB's upsert with
            # "list index out of range" further down the pipeline —
            # not a weak signal to flag, just nothing to chunk at all.
            # Skip rather than emit a hollow chunk. In the normal case
            # this should already be prevented upstream (Pass B now
            # rejects empty descriptions before creating a GEN item),
            # so hitting this is itself worth investigating if it
            # happens often.
            continue

        if len(content) > MAX_CHUNK_SIZE:
            sub_chunks = splitter.split_text(content)
            for i, sub_text in enumerate(sub_chunks, start=1):
                result.append({
                    "chunk_id": f"{item_id}#{i}",
                    "text": sub_text,
                    "item_id": item_id,
                    "family": family,
                    "linked_ids": linked_ids,
                    "is_split": True,
                })
        else:
            result.append({
                "chunk_id": item_id,
                "text": content,
                "item_id": item_id,
                "family": family,
                "linked_ids": linked_ids,
                "is_split": False,
            })

    return result


def get_document_info(text: str) -> dict:
    """
    Analyze document and return detection info.
    Used by app.py to show what type of document was uploaded.

    Returns:
        {
            "detected_types": [...],
            "total_patterns_found": int,
            "chunking_strategy": "pattern-based" or "semantic-fallback"
        }
    """
    detected = detect_document_type(text)

    return {
        "detected_types":       detected,
        "total_patterns_found": sum(d["count"] for d in detected),
        "chunking_strategy":    "pattern-based" if detected else "semantic-fallback"
    }