"""
graph/gherkin_test_case_generator.py — Prompt Enrichment + Generation
for the User Story -> Gherkin flow.

Takes a new "use case" (a user story typed in against an already-
ingested document) plus the context object from
graph/story_hybrid_retriever.retrieve_story_context(), and produces
high-coverage Gherkin BDD scenarios with traceability ids in each
scenario title, so every generated test links back to the graph nodes
(Feature / AcceptanceCriteria) it was grounded in.

Ported from graphrag2/generation/test_case_generator.py, swapped to
this project's Ollama-only call convention.
"""
import datetime
import json
import re

from parser.llm_client import call_ollama

_SYSTEM_PROMPT = (
    "You are an expert QA Automation Architect. Generate complete, deterministic "
    "Behavior-Driven Development (BDD) Gherkin test cases using precise system context. "
    "Include traceability IDs in every Scenario title, referencing the feature and any "
    "acceptance criteria ids provided. Cover edge cases implied by the business rules, "
    "not just the happy path."
)

_USER_PROMPT_TEMPLATE = """{system}

New User Story (use case): {story}

Retrieved Ground-Truth Context From Knowledge Graph (mode: {mode}):
- Feature Scope: {features}
- Flowchart / UI Sequence: {flow_steps}
- Cross-Modality Business Rules: {rules}
- Known Acceptance Criteria: {acceptance_criteria}

Generate high-coverage Gherkin Scenarios, including edge cases based on the business \
rules above. Include a traceability id in each Scenario title (e.g. "Scenario: \
Successful login [Feature: Biometric Authentication]"). Output valid Gherkin only, \
no extra commentary before or after it."""


def generate_bdd_test_suite(story_text: str, context: dict, model: str = None) -> dict:
    """
    Returns {"gherkin": str, "error": Optional[str]} — never raises;
    a failed call surfaces as an empty gherkin + error string, same
    convention parser/llm_client.call_ollama already uses, so the
    caller can decide how to degrade (e.g. keep the previous suite on
    screen rather than blanking it out).
    """
    prompt = _USER_PROMPT_TEMPLATE.format(
        system=_SYSTEM_PROMPT,
        story=story_text,
        mode=context.get("mode", "none"),
        features=json.dumps(context.get("features", [])),
        flow_steps=json.dumps(context.get("flow_steps", [])),
        rules=json.dumps(context.get("upstream_rules", [])),
        acceptance_criteria=json.dumps(context.get("acceptance_criteria", [])),
    )
    result = call_ollama(prompt, model=model, num_ctx=8192)
    if result["error"]:
        return {"gherkin": "", "error": result["error"]}
    return {"gherkin": result["raw"].strip(), "error": None}


# ─────────────────────────────────────────────────────────────────
# Gherkin -> structured rows, for export and for writing
# TestCase/TestStep nodes back into the graph with traceability.
# ─────────────────────────────────────────────────────────────────
_SCENARIO_RE = re.compile(r"^\s*Scenario(?: Outline)?:\s*(.+)$", re.IGNORECASE)
_STEP_RE = re.compile(r"^\s*(Given|When|Then|And|But)\s+(.+)$", re.IGNORECASE)


def parse_gherkin_to_cases(gherkin_text: str) -> list:
    """
    Returns: [{"title": str, "steps": [{"stepIndex": int, "action": str,
                                          "expectedResult": str}]}]
    "expectedResult" is populated for Then/And-after-Then steps; Given/When
    steps carry the action with expectedResult left blank.
    """
    cases = []
    current = None
    in_then_block = False

    for line in gherkin_text.splitlines():
        scenario_match = _SCENARIO_RE.match(line)
        if scenario_match:
            if current:
                cases.append(current)
            current = {"title": scenario_match.group(1).strip(), "steps": []}
            in_then_block = False
            continue

        step_match = _STEP_RE.match(line)
        if step_match and current is not None:
            keyword, text = step_match.groups()
            if keyword.lower() == "then":
                in_then_block = True
            elif keyword.lower() in ("given", "when"):
                in_then_block = False
            step_index = len(current["steps"]) + 1
            current["steps"].append({
                "stepIndex": step_index,
                "action": text.strip() if not in_then_block else "",
                "expectedResult": text.strip() if in_then_block else "",
            })

    if current:
        cases.append(current)

    return cases


def extract_case_id(title: str, fallback_index: int) -> str:
    match = re.search(r"\[([^\]]+)\]", title)
    if match:
        return re.sub(r"[^A-Za-z0-9]+", "-", match.group(1)).strip("-")
    return f"TC-{fallback_index}"


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"
