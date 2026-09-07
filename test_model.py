from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer

llm = ChatOllama(
    model="qwen3:latest",
    base_url="http://52.206.209.141:8002",
    temperature=0
)

ALLOWED_NODES = [
    "Actor",
    "Feature",
    "Requirement",
    "BusinessRule",
    "Condition",
    "Action",
    "Output",
    "State",
    "SystemComponent",
    "DataObject",
    "Constraint",
    "Event",
    "NonFunctionalRequirement",
    "Attribute",
    "BusinessProcess",
    "Module",
    "Workflow",
    "Service",
    "API",
    "UIElement",
    "Screen",
    "Page",
    "Role",
    "Location",
    "Country",
    "Currency",
    "Language"
]

ALLOWED_RELATIONSHIPS = [
    "USES",
    "REQUIRED_FOR",
    "TRIGGERS",
    "SHOWS",
    "LEADS_TO",
    "CAUSES",
    "CONTRIBUTES_TO",
    "PART_OF",
    "CREATES",
    "UPDATES",
    "GENERATES",
    "TRACKS",
    "CONTAINS",
    "OWNS",
    "AUTHENTICATES",
    "VALIDATES",
    "MONITORS",
    "NOTIFIES",
    "DEPENDS_ON",
    "ASSOCIATED_WITH",
    "PROCESSES",
    "STORES",
    "RETRIEVES",
    "EXECUTES"
]

graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=ALLOWED_NODES,
    allowed_relationships=ALLOWED_RELATIONSHIPS,
    strict_mode=True
)

text = """
The user searches hotels.
The system displays available hotels.
The user books a hotel.
"""

docs = [Document(page_content=text)]

graph_documents = graph_transformer.convert_to_graph_documents(docs)

for graph in graph_documents:

    print("\n========== NODES ==========\n")

    for node in graph.nodes:
        print(node)

    print("\n========== RELATIONSHIPS ==========\n")

    for rel in graph.relationships:
        print(rel)