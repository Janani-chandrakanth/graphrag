from neo4j import GraphDatabase

URI = "neo4j+s://704ad9bd.databases.neo4j.io"
USERNAME = "704ad9bd"
PASSWORD = "ojr6x4zagt--vRGFYOdz47ml9yr9zoc_eZAs8rpLoPQ"

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)


def create_node(tx, node):

    tx.run(
        """
        MERGE (n:Entity {
            id:$id
        })

        SET
            n.name=$name,
            n.type=$type
        """,
        id=node["id"],
        name=node["name"],
        type=node["type"]
    )


def create_relationship(tx, rel):

    query = f"""
    MATCH (a:Entity {{id:$from_id}})
    MATCH (b:Entity {{id:$to_id}})
    MERGE (a)-[r:{rel['type']}]->(b)
    """

    tx.run(
        query,
        from_id=rel["from"],
        to_id=rel["to"]
    )


def insert_graph(nodes, relationships):

    with driver.session() as session:

        for node in nodes:

            session.execute_write(
                create_node,
                node
            )

        for rel in relationships:

            session.execute_write(
                create_relationship,
                rel
            )