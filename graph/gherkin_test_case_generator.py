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


def generate_bdd_test_suite(story_text_or_nodes, context_or_rels=None, model: str = None) -> dict:
    """
    Supports dual signatures:
    1. generate_bdd_test_suite(nodes: list, rels: list)
    2. generate_bdd_test_suite(story_text: str, context: dict)

    Returns {"gherkin_suite": str, "test_cases": list, "error": Optional[str]}
    """
    if isinstance(story_text_or_nodes, list):
        nodes = story_text_or_nodes
        rels = context_or_rels if isinstance(context_or_rels, list) else []
        from graph.graph_test_case_generator import generate_test_cases_from_graph
        graph_res = generate_test_cases_from_graph(nodes, rels)
        test_cases = graph_res.get("test_cases", [])
        
        gherkin_lines = ["Feature: Knowledge Graph Grounded Feature Suite\n"]
        for tc in test_cases:
            title = tc.get("title", "Scenario")
            tc_id = tc.get("tc_id", "TC-001")
            prec = tc.get("precondition") or "System is initialized"
            exp = tc.get("expected_result") or "Operation completes successfully"
            gherkin_lines.append(f"  Scenario: {title} [{tc_id}]")
            gherkin_lines.append(f"    Given {prec}")
            steps = tc.get("steps", [])
            if steps:
                gherkin_lines.append(f"    When {steps[0]}")
                for s in steps[1:]:
                    gherkin_lines.append(f"    And {s}")
            gherkin_lines.append(f"    Then {exp}\n")
        
        suite_text = "\n".join(gherkin_lines)
        return {
            "gherkin_suite": suite_text,
            "gherkin": suite_text,
            "test_cases": test_cases,
            "error": None
        }

    story_text = str(story_text_or_nodes or "")
    context = context_or_rels if isinstance(context_or_rels, dict) else {}
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
    if result.get("error"):
        return {"gherkin_suite": "", "gherkin": "", "test_cases": [], "error": result["error"]}
    
    raw_gherkin = result["raw"].strip()
    parsed_cases = parse_gherkin_to_cases(raw_gherkin)
    return {"gherkin_suite": raw_gherkin, "gherkin": raw_gherkin, "test_cases": parsed_cases, "error": None}



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
