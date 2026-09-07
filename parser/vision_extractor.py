"""
Visual Intelligence Extractor — flowcharts, BPMN diagrams, wireframes.

Produces the SAME {"nodes": [...], "relationships": [...]} shape that
graph/entity_extractor.py produces for narrative text, so it plugs
straight into the existing graph.graph_builder.build_graph() with no
changes needed there. Node/relationship types are drawn from the
project's existing vocabulary (graph/schemas.py's Node.type is
free-form; ALLOWED_RELATIONS is fixed):

    Action     -- a step the user/system performs
    Condition  -- a decision diamond
    LEADS_TO   -- sequential flow (no branch condition)
    TRIGGERS   -- conditional branch out of a Condition node
"""
import re

from parser.llm_client import call_ollama_vision, extract_json_block

_PROMPT = """You are an expert system analyst translating a visual workflow into a \
deterministic execution path. Analyze the attached flowchart/diagram image. Extract \
all structural steps, decision diamonds, and directed edges.

Respond with ONLY a valid JSON object, no other text, no markdown fences, matching \
exactly this schema:
{
  "flow_name": "String identifying the overall process",
  "steps": [
    {"id": "S1", "label": "User enters credentials", "type": "Action"},
    {"id": "D1", "label": "Is 2FA enabled?", "type": "Decision"}
  ],
  "edges": [
    {"source": "S1", "target": "D1", "condition": null},
    {"source": "D1", "target": "S2", "condition": "Yes"},
    {"source": "D1", "target": "S3", "condition": "No"}
  ]
}"""


def _norm_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", value.lower().strip())


def extract_flowchart_entities(image_path: str, source: str = None,
                                model: str = None, base_url: str = None) -> dict:
    """
    Returns {"nodes": [...], "relationships": [...], "error": Optional[str],
             "flow_name": str}
    """
    result = call_ollama_vision(_PROMPT, image_path, model=model, base_url=base_url)
    if result["error"]:
        return {"nodes": [], "relationships": [], "error": result["error"], "flow_name": None}

    parsed = extract_json_block(result["raw"])
    if not parsed or "steps" not in parsed:
        return {
            "nodes": [], "relationships": [],
            "error": f"Vision model did not return the expected JSON shape. Raw output:\n{result['raw'][:500]}",
            "flow_name": None,
        }

    flow_name = parsed.get("flow_name", "Untitled Flow")
    nodes, relationships = [], []

    for step in parsed.get("steps", []):
        step_id = _norm_id(step["id"])
        node_type = "Condition" if step.get("type", "").lower().startswith("decis") else "Action"
        nodes.append({
            "id": step_id,
            "type": node_type,
            "name": step.get("label", step_id),
            "description": None,
            "source": source,
            "attributes": {"flow_name": flow_name},
            "aliases": [],
        })

    for edge in parsed.get("edges", []):
        rel_type = "TRIGGERS" if edge.get("condition") else "LEADS_TO"
        relationships.append({
            "from": _norm_id(edge["source"]),
            "to": _norm_id(edge["target"]),
            "type": rel_type,
            "description": f"condition: {edge['condition']}" if edge.get("condition") else None,
            "source": source,
        })

    return {"nodes": nodes, "relationships": relationships, "error": None, "flow_name": flow_name}
