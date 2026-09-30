# --- Merged from item_patterns.py, templates.py, text_input.py ---
"""
parser/models.py

Consolidated models, pattern registry, canonical templates, and text input wrappers:
- Item ID pattern matching and family detection (formerly item_patterns.py)
- Canonical templates per document type (formerly templates.py)
- Pasted-text input file wrapper (formerly text_input.py)
"""

import io
import re
from datetime import datetime, timezone

# ── Per-family ID token patterns (anchored at start of string) ─────────
# Each compiled pattern matches just the ID token itself, e.g. "FR-001",
# not the rest of the line. Order matters for FAMILY_PRIORITY below —
# more specific prefixes must be tried before looser ones.

def _build_family_pattern(numeric_prefixes, word_prefixes=None):
    """
    Build a family pattern with two alternatives:

    1. Classic "PREFIX-NNN" / "PREFIXNNN" style (>= 3 digits), using the
       full prefix list including any risky bare single-letter aliases
       (e.g. "F", "R") — safe here because the 3+ digit requirement
       keeps false positives low.

    2. "PREFIX_WORD_N" style (e.g. "BR_HEADER_CURRENCY_1"): a real,
       common convention where an area/topic word sits between the
       prefix and a short numeric suffix (often just 1-2 digits).
       Only built from `word_prefixes` (multi-letter, unambiguous
       prefixes) and REQUIRES an underscore/hyphen immediately after
       the prefix, so it can't drift into matching ordinary sentences
       that happen to start with a short word like "F" or "R".
    """
    numeric_alt = '|'.join(numeric_prefixes)
    alt = rf'(?:{numeric_alt})-?\d{{3,}}'
    if word_prefixes:
        word_alt = '|'.join(word_prefixes)
        alt += rf'|(?:{word_alt})[_-](?:[A-Za-z]+[_-])+\d+'
    return re.compile(rf'^\s*(?:{alt})\b', re.IGNORECASE)


ITEM_ID_PATTERNS = {
    "FR":  _build_family_pattern(["FR", "F"], word_prefixes=["FR"]),
    "NFR": _build_family_pattern(["NFR", "NR", "NF"], word_prefixes=["NFR", "NR", "NF"]),
    "BR":  _build_family_pattern(["BR", "BRD", "BU"], word_prefixes=["BR", "BRD"]),
    "TC":  _build_family_pattern(["TC", "TEST", "TS"], word_prefixes=["TC", "TEST"]),
    "UC":  _build_family_pattern(["UC", "USE"], word_prefixes=["UC"]),
    "US":  _build_family_pattern(["US", "UST"], word_prefixes=["US"]),
    "US_PROSE": re.compile(r'^\s*(?:User Story|Story)\s+\d+\b', re.IGNORECASE),
    "REQ": _build_family_pattern(["REQ", "RQ", "R"], word_prefixes=["REQ", "RQ"]),
}

# Families checked in this order when classifying a block — most specific
# prefix families first so e.g. "NFR-001" doesn't accidentally get eaten
# by a looser "FR" or "REQ" pattern.
FAMILY_PRIORITY = ["NFR", "BR", "TC", "UC", "US", "US_PROSE", "FR", "REQ"]

# Keyword/heading signals per doc type, used by the Document Type Detector
# for its regex/keyword scoring pass. Case-insensitive substring match
# against heading text.
DOC_TYPE_KEYWORDS = {
    "BRD": [
        "business requirement", "business objective", "business rule",
        "stakeholder", "scope", "business case", "problem statement",
    ],
    "SRS": [
        "functional requirement", "non-functional requirement",
        "system requirement", "software requirement", "use case",
        "system architecture", "interface requirement",
    ],
    "USER_STORY_DOC": [
        "user story", "user stories", "acceptance criteria",
        "as a user", "epic", "sprint backlog",
    ],
    "TEST_PLAN": [
        "test case", "test plan", "test scenario", "expected result",
        "preconditions", "test steps",
    ],
}

# Which ID families "belong" to which doc type, for the frequency half of
# the detector's scoring.
DOC_TYPE_ID_FAMILIES = {
    "BRD":            ["BR"],
    "SRS":            ["FR", "NFR", "REQ", "UC"],
    "USER_STORY_DOC": ["US", "US_PROSE"],
    "TEST_PLAN":      ["TC"],
}


def match_item_id(text: str):
    """
    Check if `text` starts with a known item ID token.

    Returns (family, matched_id_string) if found, else (None, None).
    Checked in FAMILY_PRIORITY order so more specific prefixes win.
    """
    for family in FAMILY_PRIORITY:
        m = ITEM_ID_PATTERNS[family].match(text)
        if m:
            return family, m.group(0).strip()
    return None, None


def find_id_mentions(text: str) -> list:
    """
    Find every ID *mention* anywhere in `text`, returning the actual
    matched substrings (not just a per-family count like
    count_families_in_text) — used by the Rule-based Requirement Linker
    to resolve exact-ID cross-references (e.g. "see TC-010" inside an
    FR-001 item's content).

    Returns a list of (family, raw_matched_text, start_index) tuples,
    in the order they appear in `text`. Uses the same loose
    (not-preceded-by-alnum) boundary as count_families_in_text so it
    can match mid-string, but resolves overlapping matches across
    families by FAMILY_PRIORITY (same precedence match_item_id uses)
    so e.g. an "R-001" match inside "REQ-001" doesn't also get counted
    separately once a higher-priority family already claimed that span.
    """
    claimed = []  # list of (start, end) spans already claimed
    found = []

    for family in FAMILY_PRIORITY:
        pattern = ITEM_ID_PATTERNS[family]
        loose = re.compile(
            pattern.pattern.replace(r'^\s*', r'(?<![A-Za-z0-9])'),
            re.IGNORECASE
        )
        for m in loose.finditer(text):
            start, end = m.span()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue  # overlaps a higher-priority family's match
            claimed.append((start, end))
            found.append((family, m.group(0).strip(), start))

    found.sort(key=lambda t: t[2])
    return found


def count_families_in_text(text: str) -> dict:
    """
    Count how many times each ID family appears anywhere in `text`
    (not anchored — used for whole-document frequency scoring, unlike
    match_item_id which is anchored for per-block classification).

    Returns {family: count} for families with count > 0.
    """
    counts = {}
    for family, pattern in ITEM_ID_PATTERNS.items():
        # Replace the "start of string" anchor with "not preceded by a
        # letter/digit" so this can match anywhere in the text, but
        # WITHOUT losing the left boundary entirely — dropping it outright
        # let e.g. "R-001" (REQ family) spuriously match inside "BR-001"
        # (BR family), since REQ's pattern is a substring of BR's.
        loose = re.compile(
            pattern.pattern.replace(r'^\s*', r'(?<![A-Za-z0-9])'),
            re.IGNORECASE
        )
        matches = loose.findall(text)
        if matches:
            counts[family] = len(matches)
    return counts


# ── Canonical Templates (formerly templates.py) ───────────────────────

DOCUMENT_TYPES = ["BRD", "SRS", "USER_STORY_DOC", "TEST_PLAN", "MIXED", "UNKNOWN"]

CANONICAL_TEMPLATES = {
    "BRD": {
        "item_type": "business_requirement",
        "id_families": DOC_TYPE_ID_FAMILIES["BRD"],
        "fields": ["description", "stakeholders", "rationale"],
        "description": "Business Requirements Document — business rules and objectives.",
        "extra_categories": [
            "business_context", "assumption", "constraint",
            "dependency", "risk", "glossary_term",
        ],
    },
    "SRS": {
        "item_type": "system_requirement",
        "id_families": DOC_TYPE_ID_FAMILIES["SRS"],
        "fields": ["description", "priority", "actor"],
        "description": "Software/System Requirements Spec — functional & non-functional requirements.",
        "extra_categories": [
            "assumption", "constraint", "dependency", "risk", "glossary_term",
        ],
    },
    "USER_STORY_DOC": {
        "item_type": "user_story",
        "id_families": DOC_TYPE_ID_FAMILIES["USER_STORY_DOC"],
        "fields": ["actor", "action", "benefit", "acceptance_criteria"],
        "description": "User Story collection — 'As a ... I want ... so that ...' format.",
        "extra_categories": ["assumption", "constraint", "glossary_term"],
    },
    "TEST_PLAN": {
        "item_type": "test_case",
        "id_families": DOC_TYPE_ID_FAMILIES["TEST_PLAN"],
        "fields": ["preconditions", "steps", "expected_result"],
        "description": "Test Plan / Test Case document.",
        "extra_categories": ["assumption", "constraint"],
    },
    "MIXED": {
        "item_type": "requirement_item",
        "id_families": ["FR", "NFR", "BR", "TC", "UC", "US", "US_PROSE", "REQ"],
        "fields": ["description"],
        "description": "Multiple document types detected with comparable confidence.",
        "extra_categories": [
            "business_context", "assumption", "constraint",
            "dependency", "risk", "glossary_term",
        ],
    },
    "UNKNOWN": {
        "item_type": "item",
        "id_families": ["FR", "NFR", "BR", "TC", "UC", "US", "US_PROSE", "REQ"],
        "fields": ["description"],
        "description": "Document type could not be determined.",
        "extra_categories": [
            "business_context", "assumption", "constraint",
            "dependency", "risk", "glossary_term",
        ],
    },
}


def get_template(doc_type: str) -> dict:
    """Look up a template, falling back to UNKNOWN for unrecognized types
    rather than raising — normalization should always be able to run."""
    template = CANONICAL_TEMPLATES.get(doc_type, CANONICAL_TEMPLATES["UNKNOWN"])
    template.setdefault("extra_categories", [])
    if "other" not in template["extra_categories"]:
        template = {**template, "extra_categories": template["extra_categories"] + ["other"]}
    return template


# ── Pasted-text input adapter (formerly text_input.py) ─────────────────

class PastedTextFile(io.BytesIO):
    """BytesIO with a `.name` attribute."""

    def __init__(self, text: str, filename: str = None):
        super().__init__((text or "").encode("utf-8"))
        self.name = filename or f"pasted_text_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.txt"


def wrap_pasted_text(text: str, filename: str = None) -> PastedTextFile:
    """Wrap raw text so it can be passed straight into
    parser.parser.parse_document() / parse_document's callers."""
    return PastedTextFile(text, filename)
