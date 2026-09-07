# graph/evaluation/llm_bridge.py
"""LLM Bridge for Test Case Semantic Evaluation.

Calls the local Ollama model to produce per-test-case semantic judgments.
Passes the evaluation_contract.py output (filtered to just workflow nodes,
requirements, business rules, inputs) to the LLM prompt instead of dumping the
raw unstructured graph nodes. Reduces prompt size by stripping Neo4j system IDs.
"""

import json
import logging
from typing import List, Dict, Any

from graph.evaluation.evaluation_contract import build_evaluation_contract

logger = logging.getLogger(__name__)

# Maximum number of test cases to send to the LLM in one batch
_BATCH_SIZE = 5


def _build_judgment_prompt(batch: List[Dict[str, Any]], contract: Dict[str, Any]) -> str:
    # Filter and format the contract to just names/descriptions, no Neo4j system IDs
    def format_nodes(nodes: List[Dict[str, Any]]) -> List[str]:
        return [
            f"{n.get('name') or n.get('id')}" + (f": {n.get('description')[:50]}" if n.get("description") else "")
            for n in nodes if n.get('name') or n.get('id')
        ][:15] # Cap each category to avoid blowing up context

    context_data = {
        "workflow_steps": format_nodes(contract.get("workflow_nodes", [])),
        "requirements": format_nodes(contract.get("requirements", [])),
        "business_rules": format_nodes(contract.get("business_rules", [])),
        "inputs": format_nodes(contract.get("inputs", []))
    }

    tc_block = json.dumps(
        [
            {
                "id": tc.get("id", ""),
                "title": tc.get("title", ""),
                "scenario_type": tc.get("scenario_type") or tc.get("type") or "unknown",
                "steps": tc.get("steps", [])[:6],
                "expected_result": tc.get("expected_result") or tc.get("expected_results") or "",
            }
            for tc in batch
        ],
        indent=2,
    )

    return f"""You are a QA expert and knowledge graph analyst.

EVALUATION UNIVERSE (Semantically Extracted):
{json.dumps(context_data, indent=2)}

TEST CASES:
{tc_block}

For each test case, output a JSON array. Each element must have:
  "id": the test case id (string)
  "scenario_type_correct": true if the scenario_type label correctly describes the test (bool)
  "graph_nodes_covered": list of node names from the EVALUATION UNIVERSE that this test case semantically covers (list of strings)
  "is_valid_flow": true if the test steps follow a plausible, sequential workflow path (bool)

Return ONLY the raw JSON array. No markdown, no explanation.
"""


def invoke_llm_for_judgment(
    test_cases: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    graph_rels: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Call Ollama to get semantic judgments for each test case.

    Returns a list of judgment dicts keyed by test case id.
    On any failure, returns an empty list so the aggregator can fall back gracefully.
    """
    if not test_cases or not graph_nodes:
        return []

    try:
        from parser.llm_client import call_ollama, extract_json_block
    except ImportError:
        logger.warning("llm_bridge: cannot import call_ollama - skipping LLM judgments")
        return []

    # Get the semantic contract (deterministic/fast)
    contract = build_evaluation_contract(graph_nodes, graph_rels, use_llm=False)

    all_judgments: List[Dict[str, Any]] = []

    # Process in batches to respect LLM context limits
    for i in range(0, len(test_cases), _BATCH_SIZE):
        batch = test_cases[i: i + _BATCH_SIZE]
        prompt = _build_judgment_prompt(batch, contract)
        try:
            res = call_ollama(prompt, timeout=60)
            parsed = extract_json_block(res["raw"])
            if isinstance(parsed, list):
                all_judgments.extend(parsed)
        except Exception as exc:
            logger.warning("llm_bridge: batch %d failed: %s", i, exc)
            continue

    return all_judgments
