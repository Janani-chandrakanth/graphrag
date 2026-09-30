from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict


# =========================================================
# WORKFLOW & FEATURE MODELS (Dataclasses)
# =========================================================

@dataclass
class Step:
    """Represents a single workflow step."""
    id: str
    action: str
    actors: List[str] = field(default_factory=list)
    systems: List[str] = field(default_factory=list)
    data_objects: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Feature:
    """A feature containing an ordered list of steps."""
    name: str
    steps: List[Step] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowModel:
    """Top‑level model holding all extracted features."""
    features: List[Feature] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


# =========================================================
# STRUCTURAL LINKER HELPER
# =========================================================

def tag_extraction_source(graph_data: dict, item_id: str, doc_id: str = None, *args, **kwargs) -> dict:
    """
    Tag every node/relationship extract_entities() returned for one
    chunk with the requirement item it came from (populates the
    "source" field graph/schemas.py's Node/Relationship models already
    define). Optionally tags doc_id for document-level filtering.
    """
    effective_doc_id = doc_id or kwargs.get("doc_id")
    for node in graph_data.get("nodes", []):
        node["source"] = item_id
        if effective_doc_id:
            node["doc_id"] = effective_doc_id

    for rel in graph_data.get("relationships", []):
        rel["source"] = item_id
        if effective_doc_id:
            rel["doc_id"] = effective_doc_id

    return graph_data


# =========================================================
# GRAPH BUILDER HELPER
# =========================================================

def build_graph(extracted_result):
    from graph.neo4j_manager import insert_graph

    nodes = extracted_result.get("nodes", [])
    relationships = extracted_result.get("relationships", [])
    print(f"Nodes: {len(nodes)}")
    print(f"Relationships: {len(relationships)}")
    insert_graph(nodes, relationships)


# =========================================================
# ALLOWED RELATIONSHIP TYPES
# Pydantic enforces this — replaces the manual check
# in validator.py for type validation
# =========================================================

ALLOWED_RELATIONS = Literal[
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
    "EXECUTES",
    # Deterministic types from graph/structural_linker.py — see
    # graph/validator.py's ALLOWED_RELATIONS for the matching note.
    "VERIFIED_BY",
    "VERIFIES",
    "REALIZES",
    "REALIZED_BY",
    "RELATED_TO",
    "RELATES_TO",
    # ── Added to cover richer BR/NFR/Risk/Glossary documents —
    # keep in sync with graph/validator.py's ALLOWED_RELATIONS ──
    "ACTOR_OF",
    "PRODUCES",
    "VIEWS",
    "PERFORMS",
    "REFERENCES",
    "SUPPORTS",
    "CONSTRAINS",
    "THREATENS",
    "DEFINES",
    "NEXT_IN_DOCUMENT",
    "NAVIGATES_TO",
]


# =========================================================
# NODE SCHEMA
# Every entity extracted by the LLM must match this shape
# =========================================================

class Node(BaseModel):

    id: str = Field(
        ...,                        # ... means REQUIRED — cannot be missing
        description="Unique snake_case identifier"
    )

    type: str = Field(
        ...,
        description="Entity type e.g. Actor, Feature, Requirement"
    )

    name: str = Field(
        ...,
        description="Human readable name"
    )


    description: Optional[str] = Field(
        default=None,
        description="Short description of the entity"
    )

    source: Optional[str] = Field(
        default=None,
        description="Requirement ID where this entity originated"
    )

    attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional entity attributes"
    )

    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names for the entity"
    )

    # -------------------------------------------------------
    # Auto-normalize the id when it comes in
    # So even if LLM returns "User-Login" we fix it here
    # -------------------------------------------------------

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return (
            value.lower()
            .replace("-", "_")
            .replace(" ", "_")
            .strip()
        )


# =========================================================
# RELATIONSHIP SCHEMA
# Every relationship extracted by LLM must match this shape
# =========================================================

class Relationship(BaseModel):

    # "from" is a reserved Python keyword
    # so we store it as from_node internally
    # but accept "from" from the JSON using alias

    from_node: str = Field(
        ...,
        alias="from",               # LLM returns "from" key → maps to from_node
        description="Source node id"
    )

    to: str = Field(
        ...,
        description="Target node id"
    )

    type: str = Field(
        ...,
        description="Relationship type — auto-normalized to UPPER_SNAKE_CASE"
    )

    description: Optional[str] = Field(
        default=None,
        description="Description of why these nodes are connected"
    )

    source: Optional[str] = Field(
        default=None,
        description="Requirement ID where this relationship originated"
    )

    @field_validator("type")
    @classmethod
    def normalize_rel_type(cls, value: str) -> str:
        if not value or not str(value).strip():
            return "RELATED_TO"
        return str(value).strip().upper().replace(" ", "_").replace("-", "_")

    # Normalize from_node and to ids
    @field_validator("from_node", "to")
    @classmethod
    def normalize_node_id(cls, value: str) -> str:
        return (
            value.lower()
            .replace("-", "_")
            .replace(" ", "_")
            .strip()
        )

    # Allow "from" alias when creating from dict
    model_config = {"populate_by_name": True}


# =========================================================
# GRAPH DATA SCHEMA
# The full LLM output must match this shape
# =========================================================

class GraphData(BaseModel):

    nodes: List[Node] = Field(
        default_factory=list,       # if "nodes" missing → use empty list
        description="List of extracted nodes"
    )

    relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of extracted relationships"
    )