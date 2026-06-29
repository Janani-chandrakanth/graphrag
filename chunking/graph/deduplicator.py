def normalize_id(node_id):

    return (
        node_id.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .strip()
    )


def deduplicate_graph(
    nodes,
    relationships
):

    unique_nodes = {}

    for node in nodes:

        node["id"] = normalize_id(
            node["id"]
        )

        if node["id"] not in unique_nodes:

            unique_nodes[
                node["id"]
            ] = node

    unique_relationships = []

    seen_relationships = set()

    for rel in relationships:

        rel["from"] = normalize_id(
            rel["from"]
        )

        rel["to"] = normalize_id(
            rel["to"]
        )

        key = (
            rel["from"],
            rel["to"],
            rel["type"]
        )

        if key not in seen_relationships:

            seen_relationships.add(key)

            unique_relationships.append(
                rel
            )

    return (
        list(unique_nodes.values()),
        unique_relationships
    )