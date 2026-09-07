from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer

from config import OLLAMA_URL


llm = ChatOllama(
    model="qwen3:latest",       
    base_url=OLLAMA_URL,
    temperature=0
)

# --------------------------------------------
# Generic Graph Transformer
# (No ontology restrictions)
# --------------------------------------------

graph_transformer = LLMGraphTransformer(
    llm=llm,
    strict_mode=False
)

# --------------------------------------------
# Load BRD
# --------------------------------------------

text = Path("requirement.txt").read_text(
    encoding="utf-8"
)

document = Document(
    page_content=text
)

print("=" * 80)
print("Generating Knowledge Graph...")
print("=" * 80)

graph_documents = graph_transformer.convert_to_graph_documents(
    [document]
)

graph = graph_documents[0]

# --------------------------------------------
# Print Statistics
# --------------------------------------------

print("\n")
print("=" * 80)
print("GRAPH STATISTICS")
print("=" * 80)

print("Nodes :", len(graph.nodes))
print("Relationships :", len(graph.relationships))

print("\n")
print("=" * 80)
print("NODES")
print("=" * 80)

for node in graph.nodes:
    print(
        f"{node.id:35}"
        f"{node.type}"
    )

print("\n")
print("=" * 80)
print("RELATIONSHIPS")
print("=" * 80)

for rel in graph.relationships:
    print(
        f"{rel.source.id}"
        f" --{rel.type}--> "
        f"{rel.target.id}"
    )

# --------------------------------------------
# Save JSON
# --------------------------------------------

graph_json = {
    "nodes": [
        {
            "id": node.id,
            "type": node.type,
            "properties": node.properties
        }
        for node in graph.nodes
    ],
    "relationships": [
        {
            "source": rel.source.id,
            "target": rel.target.id,
            "type": rel.type,
            "properties": rel.properties
        }
        for rel in graph.relationships
    ]
}

with open(
    "graph_builder_output.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(graph_json, f, indent=4)

print("\n")
print("=" * 80)
print("Graph saved to graph_builder_output.json")
print("=" * 80)