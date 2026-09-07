import json
import re
from config import WORKFLOW_MODEL, WORKFLOW_MODEL_URL
from parser.llm_client import call_ollama, extract_json_block
from prompts.sequence_extraction_prompt import SEQUENCE_EXTRACTION_PROMPT


# ── Procedural-content guard ────────────────────────────────────────────────
# Pattern signals that a chunk describes a sequential procedure.  Only chunks
# that match at least one signal go through the expensive LLM sequence call.
# Chunks that clearly don't (definitions, NFR-only, glossaries, headings) are
# skipped entirely — saving the 300 s / 3-retry qwen3 round-trip.
_PROCEDURAL_SIGNALS = re.compile(
    r"""
    (?:^|\n)\s*\d+[\.\)]\s+\w          # numbered list  "1. Login"
    | (?:^|\n)\s*[-*•]\s+\w            # bullet list
    | \b(?:then|next|after\s+that|following(?:ly)?|subsequently|
           finally|first(?:ly)?|second(?:ly)?|third(?:ly)?|
           step\s+\d|proceed\s+to|upon\s+success)\b
    | \b(?:the\s+(?:user|system|admin|customer|actor)\s+
           (?:shall|must|will|should|can|clicks?|enters?|
            submits?|selects?|navigates?|uploads?|triggers?))\b
    | (?:→|->|==>)                      # explicit flow arrows
    | \bAC[-_]?\d+\b                    # Acceptance Criteria IDs
    | \b(?:given|when|then)\b           # Gherkin keywords
    | (?:^|\n)\s*(?:Step|Phase|Stage)\s+\d+
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Patterns that strongly indicate NON-procedural text.
_NON_PROCEDURAL_SIGNALS = re.compile(
    r"""
    \b(?:definition|glossary|abbreviation|terminology|legend)\b
    | \b(?:non[-\s]?functional|NFR|performance\s+requirement|
           scalability|availability|reliability)\b
    | (?:^|\n)\s*(?:Table\s+of\s+Contents|Revision\s+History|
                    Document\s+Control|Appendix\s+[A-Z]|
                    References?|Bibliography)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_procedural_chunk(text: str) -> bool:
    """
    Returns True if *text* is likely to contain a sequential procedure worth
    sending to the sequence LLM.  Uses fast regex heuristics — no LLM call.

    False positives (non-procedural text that passes) are harmless: the LLM
    will simply return an empty sequence.  False negatives (procedural text
    that fails) are also rare and only mean a sequence is missed for that
    chunk, which is acceptable given the 300 s cost saved per skipped call.
    """
    if not text or len(text.strip()) < 40:
        return False
    # Hard reject obvious non-procedural sections
    if _NON_PROCEDURAL_SIGNALS.search(text):
        return False
    # Accept if any procedural signal present
    return bool(_PROCEDURAL_SIGNALS.search(text))


def extract_workflow_sequence(requirements_text: str, extracted_nodes: list) -> dict:
    """
    Analyzes the text and the extracted nodes to detect one or more sequential
    workflows. Supports multiple per-actor sequences (e.g., separate chains for
    User, Admin, and Restaurant Owner in the same document).

    Returns:
    {
        "sequences": [
            {
                "actor_id": str | None,
                "flow_entry_id": str,
                "sequence_edges": [{"from": ..., "to": ..., "type": "LEADS_TO"}, ...]
            },
            ...
        ],
        "error": str | None
    }
    """
    if not extracted_nodes:
        return {"sequences": [], "error": None}

    # ── Fast path: skip LLM for clearly non-procedural chunks ───────────────
    if not is_procedural_chunk(requirements_text):
        return {"sequences": [], "error": None}


    # Only send Action/Feature/Requirement/BusinessProcess nodes to keep the
    # prompt focused — Actor nodes are included so the LLM can set actor_id
    # correctly, but we don't need Attribute/Constraint/etc. cluttering it.
    RELEVANT_TYPES = {
        "Actor", "Feature", "Action", "Requirement", "BusinessProcess",
        "Workflow", "Event", "State", "SystemComponent"
    }
    filtered_nodes = [
        n for n in extracted_nodes
        if n.get("type") in RELEVANT_TYPES
    ]
    # Fall back to all nodes if filtering leaves nothing useful
    nodes_for_prompt = filtered_nodes if len(filtered_nodes) >= 2 else extracted_nodes

    prompt = SEQUENCE_EXTRACTION_PROMPT.replace(
        "{requirements_text}", requirements_text
    ).replace(
        "{extracted_nodes}", json.dumps(
            [{"id": n["id"], "name": n["name"], "type": n.get("type", "")}
             for n in nodes_for_prompt],
            indent=2
        )
    )

    # Increased timeout and retry logic for robustness against occasional server delays
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        result = call_ollama(
            prompt,
            timeout=300,  # extended timeout to 5 minutes
            num_ctx=8192,
            model=WORKFLOW_MODEL,
            base_url=WORKFLOW_MODEL_URL
        )
        if not result["error"]:
            break
        # If timeout error, retry; otherwise break immediately
        if "timed out" in result["error"].lower():
            if attempt < max_retries:
                print(f"Retry {attempt}/{max_retries} after timeout...")
                continue
        # Non-retryable error or max attempts reached
        break

    if result["error"]:
        print(f"Workflow sequence extraction error: {result['error']}")
        return {"sequences": [], "error": result["error"]}

    try:
        parsed = extract_json_block(result["raw"])
        print("\n--- SEQUENCE EXTRACTOR LLM OUTPUT ---")
        print(json.dumps(parsed, indent=2))
        print("---------------------------------------\n")

        # The prompt asks for an array at the top level
        if isinstance(parsed, list):
            sequences = parsed
        elif isinstance(parsed, dict):
            # Graceful fallback: old single-sequence schema
            if parsed.get("is_sequence") and parsed.get("sequence_edges"):
                sequences = [{
                    "actor_id": None,
                    "flow_entry_id": parsed.get("flow_entry_id"),
                    "sequence_edges": parsed.get("sequence_edges", [])
                }]
            else:
                sequences = []
        else:
            sequences = []

        # Validate: filter out sequences with no edges or missing flow_entry_id
        valid_node_ids = {n["id"] for n in extracted_nodes}
        clean_sequences = []
        for seq in sequences:
            entry = seq.get("flow_entry_id")
            edges = seq.get("sequence_edges", [])
            if not entry or not edges:
                continue
            if entry not in valid_node_ids:
                print(f"  [WARN] flow_entry_id '{entry}' not in extracted nodes — skipping sequence")
                continue
            # Filter out edges referencing unknown node IDs
            valid_edges = [
                e for e in edges
                if e.get("from") in valid_node_ids and e.get("to") in valid_node_ids
            ]
            if not valid_edges:
                continue
            clean_sequences.append({
                "actor_id": seq.get("actor_id"),
                "flow_entry_id": entry,
                "sequence_edges": valid_edges
            })

        return {"sequences": clean_sequences, "error": None}

    except Exception as e:
        print(f"Error parsing sequence extraction JSON: {e}")
        return {"sequences": [], "error": f"Error parsing JSON: {e}"}
