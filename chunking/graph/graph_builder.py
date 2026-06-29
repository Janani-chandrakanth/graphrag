from graph.neo4j_manager import insert_graph


def build_graph(extracted_result):

    nodes = extracted_result.get(
        "nodes",
        []
    )

    relationships = extracted_result.get(
        "relationships",
        []
    )
    print(
        f"Nodes: {len(nodes)}"
    )
    print(
        f"Relationships: {len(relationships)}"
    )

    insert_graph(
        nodes,
        relationships
    )