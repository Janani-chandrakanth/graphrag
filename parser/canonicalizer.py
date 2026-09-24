"""
parser/canonicalizer.py

Preprocesses parsed document structures into a clean, canonical markdown format
while preserving requirement IDs, section hierarchy, and content completeness.
"""

import io
import re

from parser.llm_client import call_ollama
from parser.txt_extractor import extract_txt_structure
from config import OLLAMA_CODER_URL

# Coder-tuned models are trained to transform text literally and
# structurally rather than creatively paraphrase it — closer to what
# canonicalization needs than a chat-tuned generalist inclined to
# summarize. gpt-oss:20b and gemma4:26b are included as generalist
# controls: if either does as well as the coder models on your real
# documents, the "coder" bias isn't actually what's helping and you can
# save the compute. Pick empirically per your own completeness-check
# results, not off this list alone — see the handoff doc's evaluation
# notes.
AVAILABLE_CANONICALIZER_MODELS = [
    "qwen2.5-coder:32b",
    "qwen3-coder:latest",
    "gpt-oss:20b",
    "gemma4:26b",
]
DEFAULT_CANONICALIZER_MODEL = AVAILABLE_CANONICALIZER_MODELS[0]

# Whole-document canonicalization runs once per document, not once per
# chunk (unlike per-chunk entity extraction) — a bigger context budget
# here is cheap relative to the current per-chunk cost across dozens of
# chunks. Input batches are capped well below this so there's headroom
# left for the model's own output.
_CANON_NUM_CTX = 16384
_BATCH_MAX_CHARS = 6000

CANONICAL_SECTIONS = [
    "Overview",
    "Actors",
    "Functional Requirements",
    "Non-Functional Requirements",
    "Business Rules",
    "Assumptions and Dependencies",
    "Risks",
    "Glossary",
]

_CANONICALIZER_PROMPT = """You are reformatting a fragment of a requirements document into a fixed template. This is a REFORMATTING task, not a summarization task.

RULES — follow these exactly:
1. Do NOT drop, compress, or summarize away any information. Every requirement, rule, constraint, and detail sentence in the input must appear somewhere in your output, in your own words is fine, but nothing may be omitted.
2. Use ONLY these section headers, exactly as written, each on its own line as "## <Section Name>": {sections}
3. Only emit a section header if this fragment actually has content for it. Do not emit an empty section.
4. If the input text already contains an ID token (examples: FR-001, NFR-002, BR_STAY_6, TC-014 — any short alphanumeric code with a hyphen or underscore before a number), you MUST preserve that exact token verbatim as the start of that item's heading: "### <ID>: <short title>". Do not reword, renumber, or drop the ID.
5. If a sentence or requirement has NO existing ID token, write it under the most appropriate section as "### <short title>" (no fabricated ID — never invent an ID that wasn't in the source).
6. If a sentence doesn't clearly fit any section above, put it under "## Overview" rather than omitting it.
7. Output ONLY the reformatted markdown. No preamble, no explanation, no commentary, no code fences.

INPUT FRAGMENT:
{fragment}
"""


def _strip_code_fence(text: str) -> str:
    """Models sometimes wrap output in ```markdown ... ``` despite being
    told not to — strip that if present rather than let it leak into the
    canonical document as literal fence characters."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _make_batches(markdown_text: str, max_chars: int = _BATCH_MAX_CHARS):
    """Split on paragraph boundaries (blank lines), accumulating into
    batches capped at max_chars — same batching discipline as
    template_normalizer._make_batches, applied to raw markdown text
    instead of parsed Block objects since this step runs before
    document structure has been reduced to items."""
    paragraphs = [p for p in markdown_text.split("\n\n") if p.strip()]
    batches = []
    current = []
    current_len = 0
    for p in paragraphs:
        p_len = len(p) + 2
        if current and current_len + p_len > max_chars:
            batches.append("\n\n".join(current))
            current, current_len = [], 0
        if p_len > max_chars:
            # a single paragraph alone exceeds the cap (e.g. a huge
            # table) — send it on its own rather than silently split
            # mid-paragraph, which would risk cutting an ID away from
            # its body text.
            if current:
                batches.append("\n\n".join(current))
                current, current_len = [], 0
            batches.append(p)
            continue
        current.append(p)
        current_len += p_len
    if current:
        batches.append("\n\n".join(current))
    return batches


def _merge_batches(batch_outputs: list) -> str:
    """Structural (non-LLM) merge: parse each batch's '## Section' /
    '### Item' output and concatenate same-named sections together, in
    CANONICAL_SECTIONS order, preserving each batch's internal item
    order. This is the "light merge pass" from the design notes — a
    second LLM reconciliation pass is a reasonable future upgrade if
    duplicate/near-duplicate items across batches turn out to be a real
    problem in practice, but isn't needed for a first version and adds
    another place content could be silently altered."""
    section_content = {name: [] for name in CANONICAL_SECTIONS}
    unknown_section_content = []

    header_re = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

    for output in batch_outputs:
        matches = list(header_re.finditer(output))
        if not matches:
            if output.strip():
                unknown_section_content.append(output.strip())
            continue
        for i, m in enumerate(matches):
            section_name = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
            body = output[start:end].strip()
            if not body:
                continue
            if section_name in section_content:
                section_content[section_name].append(body)
            else:
                # model used a header not in CANONICAL_SECTIONS despite
                # instructions — keep the content, don't drop it, just
                # can't guarantee its position in the final document.
                unknown_section_content.append(f"## {section_name}\n\n{body}")

    lines = []
    for name in CANONICAL_SECTIONS:
        if section_content[name]:
            lines.append(f"## {name}")
            lines.append("")
            lines.append("\n\n".join(section_content[name]))
            lines.append("")

    if unknown_section_content:
        lines.append("## Additional Detail")
        lines.append("")
        lines.append(
            "*(content the canonicalizer produced under a section name "
            "outside the fixed template — kept, not dropped)*"
        )
        lines.append("")
        lines.append("\n\n".join(unknown_section_content))
        lines.append("")

    return "\n".join(lines).strip()


def _completeness_check(original_text: str, canonical_text: str) -> dict:
    """
    Deterministic (no LLM) completeness signal. Two checks:

    1. ID-token coverage — every ID mention (FR-001, BR_STAY_6, ...)
       found anywhere in the original text must still appear somewhere
       in the canonical text. This is the strong signal: IDs are
       unambiguous, exact strings, so "missing" here means real,
       checkable evidence of dropped content — not a fuzzy guess.
    2. Word-count ratio — a coarse fallback signal for content that had
       no ID token to track (e.g. free-form prose a document opens
       with). A canonical output dramatically shorter than the
       original is worth a human glancing at even if every ID is
       accounted for, since ID coverage alone can't catch "the actor
       list next to FR-001 got summarized away."

    Neither check is a proof of correctness — they're cheap, honest
    signals for "does this look complete," not a guarantee.
    """
    from parser.item_patterns import find_id_mentions

    original_ids = {m[1].upper() for m in find_id_mentions(original_text)}
    canonical_upper = canonical_text.upper()
    missing_ids = sorted(i for i in original_ids if i not in canonical_upper)

    original_words = len(original_text.split())
    canonical_words = len(canonical_text.split())
    ratio = round(canonical_words / original_words, 3) if original_words else 1.0

    flagged = bool(missing_ids) or ratio < 0.6

    return {
        "total_ids_in_source": len(original_ids),
        "missing_ids": missing_ids,
        "original_word_count": original_words,
        "canonical_word_count": canonical_words,
        "word_count_ratio": ratio,
        "flagged": flagged,
    }


def canonicalize_document(
    markdown_text: str,
    model: str = DEFAULT_CANONICALIZER_MODEL,
) -> dict:
    """
    Rewrite `markdown_text` (typically DocumentStructure.to_markdown())
    into one fixed canonical template.

    Returns:
        {
            "canonical_markdown": str,
            "model_used": str,
            "batch_count": int,
            "warnings": [str, ...],       # per-batch call/parse failures
            "completeness": {...},         # see _completeness_check
            "error": Optional[str],        # set only if EVERY batch failed
        }

    On partial failure (some batches succeed, some don't), this still
    returns whatever canonical content was produced, with the failure
    recorded in warnings AND reflected in the completeness check (the
    failed batch's IDs won't be found in the output, so they'll show up
    in missing_ids) — never a silent partial result.
    """
    warnings = []
    batches = _make_batches(markdown_text)
    batch_outputs = []

    for i, batch_text in enumerate(batches):
        prompt = _CANONICALIZER_PROMPT.format(
            sections=", ".join(CANONICAL_SECTIONS),
            fragment=batch_text,
        )
        result = call_ollama(prompt, num_ctx=_CANON_NUM_CTX, model=model, base_url=OLLAMA_CODER_URL)
        if result["error"]:
            warnings.append(f"Batch {i + 1}/{len(batches)} failed: {result['error']}")
            continue
        cleaned = _strip_code_fence(result["raw"])
        if not cleaned:
            warnings.append(f"Batch {i + 1}/{len(batches)} returned empty output.")
            continue
        batch_outputs.append(cleaned)

    if not batch_outputs:
        return {
            "canonical_markdown": "",
            "model_used": model,
            "batch_count": len(batches),
            "warnings": warnings,
            "completeness": _completeness_check(markdown_text, ""),
            "error": "All batches failed — see warnings.",
        }

    canonical_markdown = _merge_batches(batch_outputs)
    completeness = _completeness_check(markdown_text, canonical_markdown)

    if completeness["flagged"]:
        if completeness["missing_ids"]:
            warnings.append(
                f"{len(completeness['missing_ids'])} ID token(s) from the "
                f"source no longer appear in the canonical output: "
                f"{completeness['missing_ids']}"
            )
        if completeness["word_count_ratio"] < 0.6:
            warnings.append(
                f"Canonical output is {completeness['word_count_ratio'] * 100:.0f}% "
                f"of the original word count — review before trusting this "
                f"over the original."
            )

    return {
        "canonical_markdown": canonical_markdown,
        "model_used": model,
        "batch_count": len(batches),
        "warnings": warnings,
        "completeness": completeness,
        "error": None,
    }


def canonical_to_structure(canonical_markdown: str, source_filename: str):
    """
    Wrap canonicalize_document()'s output back into a DocumentStructure so
    it can be handed to document_type_detector / normalize_document
    exactly like structure.to_markdown() output normally is — this is
    what makes the canonicalizer a drop-in swap rather than a parallel
    pipeline. Reuses txt_extractor.extract_txt_structure rather than a
    second markdown parser, since our "## Section" / "### ID: title"
    output is exactly the heading convention it already understands.
    """
    return extract_txt_structure(io.StringIO(canonical_markdown), filename=source_filename)