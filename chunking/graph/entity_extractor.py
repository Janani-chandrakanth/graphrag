from ollama import Client
import json
from pydantic import ValidationError

from prompts.kg_extraction_prompt import KG_EXTRACTION_PROMPT
from graph.schemas import GraphData

# =========================================================
# Ollama client — connects to your remote LLM server
# =========================================================

client = Client(
    host="http://52.206.209.141:8002"
)


def extract_entities(requirement_text: str) -> dict:
    """
    Send requirement text to LLM and extract
    nodes + relationships as a validated graph.

    Returns a dict with:
        nodes         → list of valid node dicts
        relationships → list of valid relationship dicts
        raw_output    → raw string from LLM (for debugging)
        error         → error message if something failed
    """

    # -------------------------------------------------------
    # Step 1: Build prompt by injecting requirement text
    # -------------------------------------------------------

    prompt = KG_EXTRACTION_PROMPT.replace(
        "{requirements_text}",
        requirement_text
    )

    # -------------------------------------------------------
    # Step 2: Call LLM and get raw text response
    # -------------------------------------------------------

    response = client.generate(
        model="llama3.1:latest",
        prompt=prompt
    )

    output = response["response"]

    # -------------------------------------------------------
    # Step 3: Extract JSON block from LLM response
    # LLMs sometimes add text before/after the JSON
    # We find the outermost { } and slice it out
    # -------------------------------------------------------

    try:

        start = output.find("{")
        end = output.rfind("}")

        if start == -1 or end == -1:
            # No JSON braces found at all
            return {
                "nodes": [],
                "relationships": [],
                "error": "LLM did not return any JSON block",
                "raw_output": output
            }

        json_str = output[start:end + 1]

        # -------------------------------------------------------
        # Step 4: Parse JSON string → Python dict
        # -------------------------------------------------------

        raw_data = json.loads(json_str)

        # -------------------------------------------------------
        # Step 5: Validate with Pydantic
        # This is the NEW step your manager expected
        #
        # GraphData(**raw_data) will:
        #   - Check every node has id, type, name
        #   - Normalize all ids to snake_case automatically
        #   - Check every relationship has from, to, type
        #   - Reject any relationship type not in ALLOWED_RELATIONS
        #   - Raise ValidationError with clear message if anything wrong
        # -------------------------------------------------------

        graph = GraphData(**raw_data)

        # -------------------------------------------------------
        # Step 6: Convert validated Pydantic objects back to dicts
        # model_dump() = Pydantic v2 method (replaces .dict())
        # by_alias=True → uses "from" key instead of "from_node"
        # so downstream code (neo4j_manager etc.) still works
        # -------------------------------------------------------

        validated_nodes = [
            node.model_dump()
            for node in graph.nodes
        ]

        validated_relationships = [
            rel.model_dump(by_alias=True)
            for rel in graph.relationships
        ]

        return {
            "nodes": validated_nodes,
            "relationships": validated_relationships,
            "raw_output": json_str
        }

    except json.JSONDecodeError as e:

        # JSON parsing failed — LLM returned invalid JSON
        return {
            "nodes": [],
            "relationships": [],
            "error": f"JSON parse error: {str(e)}",
            "raw_output": output
        }

    except ValidationError as e:

        # Pydantic validation failed
        # e.errors() gives a detailed list of exactly what was wrong
        # Example:
        #   [{"loc": ("relationships", 2, "type"),
        #     "msg": "value is not a valid enum member",
        #     "input": "RELATES_TO"}]

        error_details = e.errors()

        return {
            "nodes": [],
            "relationships": [],
            "error": f"Schema validation failed: {error_details}",
            "raw_output": output
        }

    except Exception as e:

        # Catch-all for anything unexpected
        return {
            "nodes": [],
            "relationships": [],
            "error": f"Unexpected error: {str(e)}",
            "raw_output": output
        }