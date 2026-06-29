"""
Smart Auto-Detecting Chunker

Supports multiple document types:
  - Functional Requirements   (FR-001, REQ-001, F-001)
  - Non-Functional Requirements (NFR-001, NR-001)
  - Business Rules            (BR-001, BRD-001)
  - Test Cases                (TC-001, TEST-001, TS-001)
  - User Stories              (US-001, Story 1, User Story 1)
  - Use Cases                 (UC-001, USE-001)
  - Generic numbered sections (1.1, 2.3 etc.)

Flow:
  1. Detect document type from content
  2. Try pattern-based chunking for that type
  3. If no patterns found → LangChain semantic fallback
  4. If chunks too long → LangChain splits further
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Chunk size limits ──────────────────────────────────────
MAX_CHUNK_SIZE  = 1000   # chars — LangChain splits anything larger
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