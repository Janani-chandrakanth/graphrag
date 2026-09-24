"""
graph/story_extractor.py — turns narrative requirement text into the
structured entities the User Story -> Gherkin flow's graph schema
needs: Requirement, UserStory, Persona, AcceptanceCriteria.

Ported from graphrag2/ingestion/entity_extractor.py, adapted to this
project's Ollama-only call convention (parser/llm_client.call_ollama +
extract_json_block) instead of graphrag2's multi-provider
llm/llm_client.complete — per the scoping decision for this flow
(Ollama only, matching the rest of f-8).
"""

import json
from parser.llm_client import call_ollama, extract_json_block

_SYSTEM_PROMPT = (
    "You are a requirements analyst extracting structured entities from software "
    "specification text. You always respond with ONLY a valid JSON object -- no "
    "preamble, no markdown fences, no commentary."
)

_USER_PROMPT_TEMPLATE = """{system}

Extract structured entities from the text below. Invent short stable ids (e.g. REQ-1, US-1, PER-1, AC-1) that are consistent if the same real-world entity is mentioned more than once in this text. Only include entities that are actually present -- use empty lists where nothing of that type appears.

Every user story's "summary" MUST be phrased in the literal "As a <role>, I want <goal>, so that <benefit>" form, even if the source text states the same requirement as a plain declarative sentence -- that conversion IS the point of this extraction step.

Return exactly this JSON schema, nothing else:
{{
  "requirements": [
    {{"req_id": "REQ-1", "title": "...", "description": "..."}}
  ],
  "user_stories": [
    {{"story_id": "US-1", "title": "...", "summary": "As a ... I want ... so that ...",
      "persona_role": "...", "related_req_ids": ["REQ-1"]}}
  ],
  "personas": [
    {{"persona_id": "PER-1", "role": "...", "context": "..."}}
  ],
  "acceptance_criteria": [
    {{"ac_id": "AC-1", "text": "...", "type": "Functional", "related_story_id": "US-1"}}
  ]
}}

Source text:
---
{text}
---"""


def extract_story_entities(text: str, source: str = "", model: str = None) -> dict:
    """
    Args:
        text: freeform requirement text (ingestion/simple_doc_parser.blocks_to_text
            output, or any chunk of prose).
        source: label (filename/doc id) stamped onto each requirement's
            "source" field for traceability — doesn't affect extraction.
        model: None uses parser/llm_client.py's default model.

    Returns:
        {"requirements": [...], "user_stories": [...], "personas": [...],
         "acceptance_criteria": [...], "error": Optional[str]}
        Never raises: a call/parse failure returns empty lists plus an
        "error" string, so one bad chunk can't take down a whole-document
        extraction loop — same never-let-one-failure-cascade principle
        the rest of this project already follows (e.g. app.py's
        failed_chunks handling around extract_entities()).
    """
    prompt = _USER_PROMPT_TEMPLATE.format(system=_SYSTEM_PROMPT, text=text[:12000])
    result = call_ollama(prompt, model=model, num_ctx=8192)

    empty = {"requirements": [], "user_stories": [], "personas": [], "acceptance_criteria": []}

    if result["error"]:
        return {**empty, "error": result["error"]}

    try:
        parsed = extract_json_block(result["raw"])
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
    except (ValueError, Exception) as e:  # json.JSONDecodeError subclasses ValueError
        return {**empty, "error": f"could not parse extraction response: {e}"}

    for key in ("requirements", "user_stories", "personas", "acceptance_criteria"):
        parsed.setdefault(key, [])

    for req in parsed["requirements"]:
        req["source"] = source

    parsed["error"] = None
    return parsed
