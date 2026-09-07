from typing import Tuple, List
from .models import Feature, Step

def build_feature_subgraph(feature: Feature) -> Tuple[List[dict], List[dict]]:
    """Build node and relationship lists for a single feature.
    Returns a tuple (nodes, relationships) suitable for Neo4j insertion.
    """
    nodes: List[dict] = []
    relationships: List[dict] = []
    # Feature node
    nodes.append({
        "id": f"feature_{feature.name}",
        "name": feature.name,
        "type": "Feature",
    })
    # Steps
    for step in feature.steps:
        nodes.append({
            "id": step.id,
            "name": step.action,
            "type": "WorkflowStep",
            "actors": step.actors,
        })
        # NEXT relationships from this step to its successors
        for nxt in step.next_steps:
            relationships.append({
                "from": step.id,
                "to": nxt,
                "type": "NEXT",
            })
        # PERFORMS relationships from actors to this step
        for actor in step.actors:
            relationships.append({
                "from": actor,
                "to": step.id,
                "type": "PERFORMS",
            })
    return nodes, relationships
