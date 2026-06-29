import json
import requests
from pydantic import ValidationError

from prompts.kg_extraction_prompt import KG_EXTRACTION_PROMPT
from graph.schemas import Node, Relationship
from config import OLLAMA_URL


def extract_entities(requirement_text: str) -> dict:

    prompt = KG_EXTRACTION_PROMPT.replace(
        "{requirements_text}",
        requirement_text
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  "llama3.1:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,     # deterministic output
                    "seed":        42,    # same seed = same output every run
                    "top_p":       1,     # no nucleus sampling
                    "top_k":       1      # always pick highest probability token
                }
            },
            proxies={"http": None, "https": None},
            timeout=120
        )
        response.raise_for_status()
        output = response.json()["response"]

    except requests.exceptions.ConnectionError as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"Cannot reach Ollama server: {str(e)}",
            "raw_output": ""
        }
    except requests.exceptions.Timeout:
        return {
            "nodes": [], "relationships": [],
            "error": "Ollama request timed out (120s)",
            "raw_output": ""
        }
    except Exception as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"Ollama request failed: {str(e)}",
            "raw_output": ""
        }

    try:
        start = output.find("{")
        end   = output.rfind("}")

        if start == -1 or end == -1:
            return {
                "nodes": [], "relationships": [],
                "error": "LLM did not return any JSON block",
                "raw_output": output
            }

        json_str = output[start:end + 1]
        raw_data = json.loads(json_str)

        # Validate NODES one by one
        valid_nodes   = []
        skipped_nodes = []
        for node in raw_data.get("nodes", []):
            try:
                validated = Node(**node)
                valid_nodes.append(validated.model_dump())
            except ValidationError as e:
                skipped_nodes.append({"input": node, "reason": str(e.errors())})

        # Validate RELATIONSHIPS one by one
        valid_relationships   = []
        skipped_relationships = []
        for rel in raw_data.get("relationships", []):
            try:
                validated = Relationship(**rel)
                valid_relationships.append(validated.model_dump(by_alias=True))
            except ValidationError:
                skipped_relationships.append(rel)

        warnings = []
        if skipped_nodes:
            warnings.append(
                f"{len(skipped_nodes)} node(s) skipped: "
                f"{[s['input'].get('id','?') for s in skipped_nodes]}"
            )
        if skipped_relationships:
            invalid_types = list({r.get('type','?') for r in skipped_relationships})
            warnings.append(
                f"{len(skipped_relationships)} relationship(s) skipped "
                f"with invalid types: {invalid_types}"
            )

        return {
            "nodes":                 valid_nodes,
            "relationships":         valid_relationships,
            "raw_output":            json_str,
            "skipped_nodes":         skipped_nodes,
            "skipped_relationships": skipped_relationships,
            "error": "; ".join(warnings) if warnings else None
        }

    except json.JSONDecodeError as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"JSON parse error: {str(e)}",
            "raw_output": output
        }
    except Exception as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"Unexpected error: {str(e)}",
            "raw_output": output
        }