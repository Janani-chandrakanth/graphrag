"""
Document Type Detector

Classifies a whole DocumentStructure as BRD / SRS / USER_STORY_DOC /
TEST_PLAN / MIXED / UNKNOWN.

This is a different job from chunker.py's detect_document_type(), which
answers "which item patterns appear in this text" (used for chunking
strategy, per-chunk). This answers "what kind of document is this, overall"
(used to pick a normalization template).

Per the decision made for this pipeline: cheap regex/keyword scoring first,
LLM fallback only when confidence is low or signals conflict — same
pattern the chunker already uses for pattern-detection-vs-LangChain-fallback,
applied one level up.
"""

from parser.normalization import DocumentStructure
from parser.models import DOC_TYPE_KEYWORDS, DOC_TYPE_ID_FAMILIES, count_families_in_text
from parser.llm_client import call_ollama, extract_json_block
from parser.models import DOCUMENT_TYPES

# Below this top-score confidence, or when the top two scores are within
# CONFLICT_MARGIN of each other, fall back to the LLM.
CONFIDENCE_FLOOR = 0.35
CONFLICT_MARGIN = 0.10

# How much weight ID-family frequency vs heading keywords contribute.
# ID families are a stronger signal (an explicit "TC-001" is unambiguous;
# a heading containing "requirement" is weaker, appears in multiple types).
WEIGHT_ID_FAMILY = 0.65
WEIGHT_KEYWORD = 0.35

LLM_FALLBACK_PROMPT = """You are classifying a software requirements document.

Read the excerpt below and decide which single category it belongs to:

- BRD: Business Requirements Document (business rules, objectives, stakeholders)
- SRS: Software/System Requirements Spec (functional/non-functional requirements, use cases)
- USER_STORY_DOC: User story collection ("As a ... I want ... so that ...", acceptance criteria)
- TEST_PLAN: Test cases / test plan (preconditions, steps, expected results)
- MIXED: Clearly contains two or more of the above in comparable amounts
- UNKNOWN: None of the above fit

Respond with ONLY a JSON object, no other text:
{"doc_type": "<one of BRD|SRS|USER_STORY_DOC|TEST_PLAN|MIXED|UNKNOWN>", "reasoning": "<one sentence>"}

DOCUMENT EXCERPT:
---
{excerpt}
---
"""

# Cap what we send the LLM — headings + first N chars of body is plenty
# signal for a type classification and keeps latency/cost down.
LLM_EXCERPT_CHAR_LIMIT = 3000


def _score_by_id_families(full_text: str) -> dict:
    """Score each doc type by how many of its native ID-family hits
    appear in the document, normalized against total ID hits found."""
    family_counts = count_families_in_text(full_text)
    total_hits = sum(family_counts.values())
    if total_hits == 0:
        return {dt: 0.0 for dt in DOC_TYPE_ID_FAMILIES}

    scores = {}
    for doc_type, families in DOC_TYPE_ID_FAMILIES.items():
        hits = sum(family_counts.get(f, 0) for f in families)
        scores[doc_type] = hits / total_hits
    return scores


def _score_by_keywords(structure: DocumentStructure) -> dict:
    """Score each doc type by keyword hits in heading text (headings are
    a much cleaner signal than scanning body prose for these phrases)."""
    heading_text = " ".join(b.content for b in structure.headings()).lower()
    if not heading_text:
        return {dt: 0.0 for dt in DOC_TYPE_KEYWORDS}

    scores = {}
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in heading_text)
        scores[doc_type] = hits / len(keywords)
    return scores


def _combine_scores(id_scores: dict, keyword_scores: dict) -> dict:
    combined = {}
    for doc_type in DOC_TYPE_ID_FAMILIES:  # BRD, SRS, USER_STORY_DOC, TEST_PLAN
        combined[doc_type] = (
            WEIGHT_ID_FAMILY * id_scores.get(doc_type, 0.0)
            + WEIGHT_KEYWORD * keyword_scores.get(doc_type, 0.0)
        )
    return combined


def _build_excerpt(structure: DocumentStructure) -> str:
    headings = " / ".join(b.content for b in structure.headings()[:15])
    body = structure.to_markdown()
    excerpt = f"Headings: {headings}\n\n{body}"
    return excerpt[:LLM_EXCERPT_CHAR_LIMIT]


def _llm_classify(structure: DocumentStructure) -> dict:
    """LLM fallback classification. Returns {"doc_type", "confidence",
    "method", "reasoning"} — confidence is fixed at 0.5 for LLM results
    since Ollama doesn't give us a real probability, just enough to rank
    below a solid regex win but count as a decision, not a guess."""
    excerpt = _build_excerpt(structure)
    prompt = LLM_FALLBACK_PROMPT.replace("{excerpt}", excerpt)

    result = call_ollama(prompt)
    if result["error"]:
        return {
            "doc_type": "UNKNOWN", "confidence": 0.0,
            "method": "llm_fallback_failed", "reasoning": result["error"],
        }

    try:
        parsed = extract_json_block(result["raw"])
        doc_type = parsed.get("doc_type", "UNKNOWN")
        if doc_type not in DOCUMENT_TYPES:
            doc_type = "UNKNOWN"
        return {
            "doc_type": doc_type, "confidence": 0.5,
            "method": "llm_fallback", "reasoning": parsed.get("reasoning", ""),
        }
    except (ValueError, KeyError) as e:
        return {
            "doc_type": "UNKNOWN", "confidence": 0.0,
            "method": "llm_fallback_failed", "reasoning": f"Could not parse LLM response: {e}",
        }


def detect_document_type(structure: DocumentStructure) -> dict:
    """
    Classify a DocumentStructure's overall document type.

    Returns:
        {
            "doc_type":   one of DOCUMENT_TYPES,
            "confidence": float 0-1,
            "method":     "regex" | "llm_fallback" | "llm_fallback_failed",
            "scores":     {doc_type: combined_score, ...},   # regex pass only
            "reasoning":  str (only present for llm_fallback)
        }
    """
    id_scores = _score_by_id_families(structure.full_text())
    keyword_scores = _score_by_keywords(structure)
    combined = _combine_scores(id_scores, keyword_scores)

    if not any(combined.values()):
        # No signal at all from either pass — go straight to LLM rather
        # than reporting a meaningless "top score" over all-zero scores.
        result = _llm_classify(structure)
        result["scores"] = combined
        return result

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    top_type, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    is_conflict = (top_score - runner_up_score) < CONFLICT_MARGIN and runner_up_score > 0
    is_low_confidence = top_score < CONFIDENCE_FLOOR

    if is_conflict:
        # Two doc types scoring near-equally is itself a signal — a
        # genuinely mixed document — before even trying the LLM.
        # Only escalate to LLM if it's *also* low confidence overall;
        # otherwise a confident tie is legitimately MIXED.
        if top_score >= CONFIDENCE_FLOOR:
            return {
                "doc_type": "MIXED", "confidence": round(top_score, 3),
                "method": "regex", "scores": combined,
            }
        result = _llm_classify(structure)
        result["scores"] = combined
        return result

    if is_low_confidence:
        result = _llm_classify(structure)
        result["scores"] = combined
        return result

    return {
        "doc_type": top_type, "confidence": round(top_score, 3),
        "method": "regex", "scores": combined,
    }
