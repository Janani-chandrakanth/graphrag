"""
graph/negative_scenario_generator.py

Generates grounded negative and edge-case test scenarios for requirement
series using an LLM pass, strictly constrained to existing graph entities.
"""

import json
from collections import defaultdict

from parser.llm_client import call_ollama, extract_json_block
from prompts.prompt_library import get_prompt as _get_library_prompt

_CATALOG_MAX_CHARS = 3000

# Kept as a literal fallback (byte-for-byte the original prompt this
# module always used) in case the prompt library file is unreadable —
# generation must never hard-fail just because prompts/prompt_library_store.json
# is missing/corrupt; it degrades to the old fixed behavior instead.
_NEGATIVE_PROMPT = """You are a QA engineer identifying NEGATIVE and EDGE-CASE test scenarios for one part of a system, based ONLY on entities already extracted from its requirements.

REQUIREMENT SERIES: {req_ids}

POSITIVE TEST CASE ALREADY BUILT FOR THIS SERIES:
  Actor: {actor}
  Feature: {feature}
  Steps: {steps}
  Expected Result: {expected_result}

ENTITIES YOU MAY REFERENCE (id | type | name) — you may ONLY use entities from this list. Do NOT invent, assume, or reference anything not in this list. No external systems, no generic infrastructure ("the database", "the network", "the API") unless it is literally in the list below:
{catalog}

Propose 1 to 3 realistic NEGATIVE or EDGE-CASE scenarios that stress, negate, or find the boundary of the entities above (e.g. an entity's value is missing, invalid, unmatched, empty, out of range, or in an unexpected state) — grounded ENTIRELY in the entities listed. Every entity name in "entities_used" must be copied EXACTLY as it appears in the catalog above. If you cannot construct a grounded negative scenario from this list, return an empty array — do not fabricate one just to have something to say.

Return ONLY a JSON array, no other text, no markdown fences. Each entry:
{{"title": "<short scenario title>", "steps": ["<step 1>", "<step 2>", "..."], "expected_result": "<expected outcome>", "entities_used": ["<exact name from catalog>", "..."]}}
"""


def _resolve_prompt_template(prompt_name: str) -> str:
    """
    Looks up `prompt_name` in the prompt library (prompts/prompt_library.py).
    Falls back to the original fixed prompt if the library entry is
    missing or unreadable — never blocks generation over a library
    problem, matches this project's degrade-visibly-not-crash pattern.
    """
    try:
        return _get_library_prompt(prompt_name)["template"]
    except KeyError:
        return _NEGATIVE_PROMPT


def _catalog_for_series(source_items: list, nodes: list) -> tuple:
    """
    Entities already extracted for this series (node["source"] in the
    series' item ids) — the exact subgraph
    graph_test_case_generator.py already walked. Returns
    (catalog_text, valid_tokens) where valid_tokens is used to
    validate the LLM's entities_used after the fact.

    NOTE: the LLM is asked to copy the entity name exactly, but in
    practice it very often returns the snake_case id instead (e.g.
    "country_of_user" rather than "Country of User") even when it is
    referring to a real, in-catalog entity. Matching against name
    alone caused most real (non-hallucinated) references to be
    dropped as false positives. valid_tokens therefore includes each
    entity's id, its raw name, and a normalized underscore form of the
    name, all lowercased, so any of those spellings is accepted.
    """
    entries = [
        {"id": n["id"], "type": n.get("type", "?"), "name": n.get("name", n["id"])}
        for n in nodes
        if n.get("source") in source_items
    ]
    text = ""
    for e in entries:
        line = f"{e['id']} | {e['type']} | {e['name']}\n"
        if text and len(text) + len(line) > _CATALOG_MAX_CHARS:
            break
        text += line
    valid_tokens = set()
    for e in entries:
        valid_tokens.add(e["id"].strip().lower())
        name_lower = e["name"].strip().lower()
        valid_tokens.add(name_lower)
        valid_tokens.add(name_lower.replace(" ", "_"))
        valid_tokens.add(name_lower.replace("_", " "))
    return text, valid_tokens


def _escape(text: str) -> str:
    """Escape curly braces so str.format() treats them as literals."""
    return text.replace("{", "{{").replace("}", "}}") if text else text


def generate_negative_scenarios(
    positive_test_cases: list,
    nodes: list,
    model: str = None,
    prompt_name: str = "edge_case_focused",
) -> dict:
    """
    Args:
        positive_test_cases: graph_test_case_generator.py's
            generate_test_cases_from_graph()["test_cases"] — used for
            req_ids/actor/feature/steps/expected_result context and
            source_items (which entities belong to this series).
        nodes: the same deduplicated/validated node list already in
            session_state["nodes"] — used to build each series' entity
            catalog.
        model: None uses the pipeline's default model (same server as
            per-chunk extraction) — deliberately NOT the canonicalizer's
            coder model; this is a reasoning task on already-clean
            structured data, not whole-document reformatting.
        prompt_name: which prompts/prompt_library.py style to use
            ("edge_case_focused" default, "gherkin_style",
            "regression_suite", or any custom saved template). Falls
            back to the original fixed prompt if the name isn't found.

    Returns:
        {
            "scenarios": [ {...same shape as a graph_test_case_generator.py
                             test case, plus "llm_derived": True and
                             "based_on_tc_id": str}, ... ],
            "warnings": [str, ...],   # dropped/hallucinated entries,
                                       # call/parse failures — never
                                       # silently swallowed
            "prompt_used": str,       # which library name actually ran
        }
    """
    scenarios = []
    warnings = []
    seq_by_req = defaultdict(int)
    active_template = _resolve_prompt_template(prompt_name)

    for tc in positive_test_cases:
        req_ids = tc.get("source_items", [tc.get("req_id", "")])
        catalog_text, valid_tokens = _catalog_for_series(req_ids, nodes)

        if not catalog_text.strip():
            warnings.append(
                f"{tc.get('req_id')}: no entity catalog available for this "
                f"series — skipped (nothing grounded to reason from)."
            )
            continue

        # Escape curly braces in user-provided text so they are treated
        # as literal characters by str.format(), not format placeholders.
        # Entity names, step descriptions, and actor/feature values can
        # contain curly braces (e.g. "{system}" or "{x} happens")
        # that would otherwise cause a KeyError from .format().
        prompt = active_template.format(
            req_ids=", ".join(req_ids),
            actor=_escape(tc.get("actor", "?")),
            feature=_escape(tc.get("feature", "?")),
            steps=_escape("; ".join(tc.get("steps", []))),
            expected_result=_escape(tc.get("expected_result", "?")),
            catalog=_escape(catalog_text),
        )

        result = call_ollama(prompt, model=model)
        if result["error"]:
            warnings.append(f"{tc.get('req_id')}: LLM call failed — {result['error']}")
            continue

        try:
            parsed = extract_json_block(result["raw"])
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array")
        except (ValueError, json.JSONDecodeError) as e:
            warnings.append(f"{tc.get('req_id')}: response could not be parsed — {e}")
            continue

        for entry in parsed:
            title = (entry.get("title") or "").strip()
            steps = entry.get("steps") or []
            expected = (entry.get("expected_result") or "").strip()
            entities_used = entry.get("entities_used") or []

            if not title or not steps or not expected:
                warnings.append(
                    f"{tc.get('req_id')}: dropped a scenario missing "
                    f"title/steps/expected_result."
                )
                continue

            # STRICT grounding check — every referenced entity must
            # exist in this series' actual catalog. Not a soft filter:
            # any single ungrounded entity drops the WHOLE scenario,
            # since a partially-hallucinated scenario is still an
            # ungrounded claim about the system.
            ungrounded = [
                e for e in entities_used
                if e.strip().lower() not in valid_tokens
            ]
            if ungrounded:
                warnings.append(
                    f"{tc.get('req_id')}: dropped scenario '{title}' — "
                    f"referenced entities not in the graph (hallucinated): "
                    f"{ungrounded}"
                )
                continue

            seq_by_req[tc.get("req_id", "")] += 1
            seq = seq_by_req[tc.get("req_id", "")]

            scenarios.append({
                "tc_id": f"TC-NEG-{tc.get('req_id', 'X')}-{seq:03d}",
                "based_on_tc_id": tc.get("tc_id"),
                "req_id": tc.get("req_id"),
                "title": title,
                "type": "Negative",
                "priority": tc.get("priority", "Medium"),
                "actor": tc.get("actor", "User"),
                "feature": tc.get("feature", "?"),
                "precondition": tc.get("precondition", ""),
                "steps": steps,
                "expected_result": expected,
                "graph_nodes": entities_used,
                "source_items": req_ids,
                "fallbacks": [],
                "llm_derived": True,
            })

    return {"scenarios": scenarios, "warnings": warnings, "prompt_used": prompt_name}
