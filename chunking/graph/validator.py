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


def validate_graph(nodes, relationships):

    node_ids = {
        node["id"]
        for node in nodes
    }

    valid_relationships = []

    for rel in relationships:

        # Invalid relation type
        if rel["type"] not in ALLOWED_RELATIONS:
            continue

        # Missing source node
        if rel["from"] not in node_ids:
            continue

        # Missing target node
        if rel["to"] not in node_ids:
            continue

        valid_relationships.append(rel)

    return nodes, valid_relationships