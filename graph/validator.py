# graph/validator.py

ALLOWED_RELATIONS = {
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
}

# =========================================================
# VALID NODE TYPES
# Only these types are allowed as nodes
# Fixes Problem 1: LLM creating "Relationship" typed nodes
# e.g. {"id": "fulfills", "type": "Relationship"} → removed
# =========================================================

VALID_NODE_TYPES = {
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
    "Attribute"
}


def validate_graph(nodes, relationships):
    """
    Clean the graph by removing:
      - Nodes with invalid types        (Problem 1)
      - Relationships with invalid type (Problem 2)
      - Relationships with missing nodes(Problem 3)

    Returns:
        valid_nodes, valid_relationships
    """

    # =========================================================
    # Step 1: Validate Nodes
    # Remove any node whose type is not in VALID_NODE_TYPES
    # This kills nodes like {"type": "Relationship"} that the
    # LLM hallucinates when it confuses edges with entities
    # =========================================================

    valid_nodes = []
    removed_nodes = []

    for node in nodes:

        # Normalize type: strip spaces so "Data Object" → "DataObject"
        node_type = (
            node.get("type", "")
            .replace(" ", "")
            .strip()
        )

        if node_type not in VALID_NODE_TYPES:
            # Log what we're removing so you can debug in Streamlit
            removed_nodes.append({
                "id":   node.get("id"),
                "type": node.get("type"),
                "reason": f"Invalid node type '{node.get('type')}'"
            })
            continue

        # Store the normalized type back onto the node
        node["type"] = node_type
        valid_nodes.append(node)

    # =========================================================
    # Step 2: Build set of valid node IDs
    # Used to check relationship endpoints exist
    # =========================================================

    node_ids = {node["id"] for node in valid_nodes}

    # =========================================================
    # Step 3: Validate Relationships
    # Remove relationships that:
    #   - have a type not in ALLOWED_RELATIONS  (Problem 2)
    #   - reference a node that doesn't exist   (Problem 3)
    # =========================================================

    valid_relationships = []
    removed_relationships = []

    for rel in relationships:

        rel_type = rel.get("type", "").strip()

        # Check 1: relationship type must be in whitelist
        if rel_type not in ALLOWED_RELATIONS:
            removed_relationships.append({
                **rel,
                "reason": f"Invalid relationship type '{rel_type}'"
            })
            continue

        # Check 2: source node must exist
        if rel["from"] not in node_ids:
            removed_relationships.append({
                **rel,
                "reason": f"Source node '{rel['from']}' not found"
            })
            continue

        # Check 3: target node must exist
        if rel["to"] not in node_ids:
            removed_relationships.append({
                **rel,
                "reason": f"Target node '{rel['to']}' not found"
            })
            continue

        valid_relationships.append(rel)

    # Return removed items too so app.py can show them for debugging
    return valid_nodes, valid_relationships, removed_nodes, removed_relationships