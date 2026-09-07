"""
Canonical Templates — per-doc-type shape the Template Normalizer produces.

Each template says: which ID families are "native" to this doc type (so
Pass A regex extraction knows what it's looking for and what to prioritize
in ambiguous blocks), what an extracted item should be called, and what
descriptive fields the LLM gap-filler (Pass B) should try to populate for
un-ID'd prose it recovers.

Kept deliberately small and data-only — no behavior lives here, just the
shape. document_type_detector.py and template_normalizer.py both import
from this module so detector output ("BRD") and normalizer input line up
without either one hardcoding the doc-type list twice.
"""

from parser.item_patterns import DOC_TYPE_ID_FAMILIES

DOCUMENT_TYPES = ["BRD", "SRS", "USER_STORY_DOC", "TEST_PLAN", "MIXED", "UNKNOWN"]

CANONICAL_TEMPLATES = {
    "BRD": {
        "item_type": "business_requirement",
        "id_families": DOC_TYPE_ID_FAMILIES["BRD"],
        "fields": ["description", "stakeholders", "rationale"],
        "description": "Business Requirements Document — business rules and objectives.",
        # Real BRD content that ISN'T phrased as a requirement — a BRD
        # always has these sections, and prose describing them (workstream
        # context, assumptions, constraints, dependencies, risks, glossary
        # terms) is genuine document substance, not boilerplate. Without
        # these, Pass B's "is this a business_requirement? if not, skip
        # it" instruction silently discards all of it — it's not vague,
        # it's just a different kind of item than a requirement.
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
    # MIXED: document has strong signal for 2+ types (e.g. an SRS with an
    # embedded test-case appendix). Normalizer falls back to a permissive
    # template accepting any known family instead of picking one.
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
    # UNKNOWN: no reliable signal at all. Normalizer still runs — Pass A
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
    # Universal catch-all: whatever doc-type-specific categories exist
    # above are for BETTER LABELING (so a risk reads as "risk" not just
    # "other requirement"), not for gatekeeping what gets kept. "other"
    # is always available so Pass B never has to choose between "force
    # this into a category it doesn't fit" and "drop it" — genuine
    # content with no better label still gets extracted, just tagged
    # generically instead of silently discarded.
    if "other" not in template["extra_categories"]:
        template = {**template, "extra_categories": template["extra_categories"] + ["other"]}
    return template