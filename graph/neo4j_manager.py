"""
Neo4j Manager
Handles all Neo4j operations with automatic reconnection.
Neo4j AuraDB free tier goes to sleep after inactivity —
SessionExpired errors are caught and the driver is recreated.
"""

import json
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


# ── Driver with reconnection support ──────────────────────
def _create_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        max_connection_lifetime=200,   # recreate connections after 200s
        max_connection_pool_size=10,
        connection_timeout=30
    )

driver = _create_driver()


def _get_driver():
    """
    Returns the driver, recreating it if the connection is dead.
    Called before every session to ensure connection is alive.
    """
    global driver
    try:
        # Quick connectivity check
        driver.verify_connectivity()
        return driver
    except Exception:
        # Connection dead — recreate driver
        driver = _create_driver()
        return driver


def _run_with_retry(fn, retries=2):
    """
    Run a Neo4j function with automatic retry on SessionExpired.

    Args:
        fn: function that takes a session and returns a result
        retries: how many times to retry

    Returns:
        result of fn
    """
    global driver
    for attempt in range(retries):
        try:
            d = _get_driver()
            with d.session() as session:
                return fn(session)
        except (SessionExpired, ServiceUnavailable):
            # Recreate driver and retry
            driver = _create_driver()
            if attempt == retries - 1:
                raise  # re-raise on last attempt
    return None


# ── Write functions ────────────────────────────────────────

def _write_node(tx, node, batch_id=None):
    attributes = node.get("attributes") or {}
    # `sequence` and `is_flow_entry` are the two attribute keys the
    # extraction prompt (prompts/kg_extraction_prompt.py's SEQUENCE /
    # FLOW EXTRACTION RULES section) is now asked to populate directly
    # from the source text's own explicit ordering, at the one point
    # in the pipeline where "what order did the text describe this
    # in" is cheapest and most reliable to determine — a single LLM
    # pass reading the paragraph top-to-bottom, vs. graph/
    # flow_graph_analysis.py inferring it after the fact purely from
    # edge topology across a multi-chunk, multi-document merged graph.
    # Promoted to real typed properties (not buried in a JSON blob)
    # so graph/flow_graph_analysis.py and graph/langchain_qa.py's
    # generated Cypher can both filter/order on them directly.
    # Everything else in `attributes` is preserved too, as a JSON
    # string — previously the entire dict was silently dropped here.
    sequence_raw = attributes.get("sequence")
    try:
        sequence_val = int(sequence_raw) if sequence_raw not in (None, "") else None
    except (TypeError, ValueError):
        sequence_val = None
    is_flow_entry_val = str(attributes.get("is_flow_entry", "")).strip().lower() in ("true", "1", "yes")

    # batch_id: optional, additive only. Existing callers (graph_builder.py's
    # main pipeline) never pass this, so `n.batch_id` is simply never SET for
    # them -- zero behavior change. Only graph/batch_upload.py's comparison
    # flow passes it, tagging its (id-namespaced, see that module's
    # docstring) nodes so graph/version_compare.py can filter by batch
    # without touching the main pipeline's write path at all.
    query = """
        MERGE (n:Entity {id: $id})
        SET
            n.name          = $name,
            n.type          = $type,
            n.description   = $description,
            n.source        = $source,
            n.aliases       = $aliases,
            n.sequence      = $sequence,
            n.is_flow_entry = $is_flow_entry,
            n.attributes_json = $attributes_json
    """
    params = dict(
        id=node["id"],
        name=node["name"],
        type=node["type"],
        description=node.get("description"),
        source=node.get("source"),
        aliases=node.get("aliases", []),
        sequence=sequence_val,
        is_flow_entry=is_flow_entry_val,
        attributes_json=json.dumps(attributes) if attributes else None,
    )
    doc_id = node.get("doc_id") or node.get("source_document")
    if doc_id:
        query += ", n.doc_id = $doc_id"
        params["doc_id"] = doc_id

    if batch_id is not None:
        query += ", n.batch_id = $batch_id"
        params["batch_id"] = batch_id

    tx.run(query, **params)

def _write_relationship(tx, rel):
    doc_id = rel.get("doc_id") or rel.get("source_document")
    source = rel.get("source")
    sets = []
    if doc_id:
        sets.append("r.doc_id = $doc_id")
    if source:
        sets.append("r.source = $source")
    set_clause = ("SET " + ", ".join(sets)) if sets else ""
    query = f"""
    MATCH (a:Entity {{id: $from_id}})
    MATCH (b:Entity {{id: $to_id}})
    MERGE (a)-[r:{rel['type']}]->(b)
    {set_clause}
    """
    params = dict(from_id=rel["from"], to_id=rel["to"])
    if doc_id:
        params["doc_id"] = doc_id
    if source:
        params["source"] = source
    tx.run(query, **params)


def _write_community(tx, node_id, community_id):
    tx.run(
        """
        MATCH (n:Entity {id: $id})
        SET n.community = $community_id
        """,
        id=node_id,
        community_id=int(community_id)
    )


# ── Public functions ───────────────────────────────────────

def create_node(tx, node):
    _write_node(tx, node)


def create_relationship(tx, rel):
    _write_relationship(tx, rel)


def insert_graph(nodes, relationships):
    def fn(session):
        for node in nodes:
            session.execute_write(_write_node, node)
        for rel in relationships:
            session.execute_write(_write_relationship, rel)

    _run_with_retry(fn)


def clear_graph():
    """
    Delete ALL nodes and relationships from the graph.
    Used before re-ingesting a document so stale hub-and-spoke
    edges from a previous extraction don't pollute the new result.
    """
    def fn(session):
        result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
        record = result.single()
        return record["deleted"] if record else 0

    deleted = _run_with_retry(fn) or 0
    return {"deleted": deleted}


def insert_graph_with_batch(nodes, relationships, batch_id: str):
    """
    Same as insert_graph() but tags every node with n.batch_id = batch_id.
    Used ONLY by graph/batch_upload.py's comparison flow -- the main
    pipeline (graph/graph_builder.py) keeps calling plain insert_graph()
    unchanged, so nothing about the existing upload flow is touched.
    """
    def fn(session):
        for node in nodes:
            session.execute_write(_write_node, node, batch_id)
        for rel in relationships:
            session.execute_write(_write_relationship, rel)

    _run_with_retry(fn)


def delete_relationships(relationships: list):
    """
    Delete a specific set of (from, to, type) relationships. Used ONLY by
    graph/incremental_merge.py's apply_merge_plan() when a conflict was
    resolved as "replace" (the old edge for that from/type is superseded
    by the new upload, see graph/incremental_merge.preserve_workflow's
    docstring) -- the main pipeline never calls this, so plain uploads
    are completely unaffected.
    """
    def fn(session):
        for rel in relationships:
            session.run(
                f"""
                MATCH (a:Entity {{id: $from_id}})-[r:{rel['type']}]->(b:Entity {{id: $to_id}})
                DELETE r
                """,
                from_id=rel["from"], to_id=rel["to"],
            )

    _run_with_retry(fn)


def run_cypher_query(query: str, parameters: dict = None) -> list:
    """Run any read/write Cypher query with optional parameters, returns list of dicts."""
    def fn(session):
        params = parameters or {}
        result = session.run(query, params)
        return [dict(record) for record in result]

    return _run_with_retry(fn) or []


def get_stored_documents() -> list:
    """Return a list of distinct document names/IDs stored in Neo4j."""
    # 1. Query nodes with explicit doc_id set
    query_docs = """
    MATCH (n:Entity)
    WHERE n.doc_id IS NOT NULL AND n.doc_id <> ''
    RETURN DISTINCT n.doc_id AS doc_id
    ORDER BY doc_id
    """
    records = run_cypher_query(query_docs)
    doc_ids = [r["doc_id"] for r in records if r.get("doc_id")]

    # 2. Query RequirementDoc nodes
    query_req_docs = """
    MATCH (d:RequirementDoc)
    RETURN DISTINCT coalesce(d.name, d.docId) AS doc_id
    ORDER BY doc_id
    """
    req_records = run_cypher_query(query_req_docs)
    for r in req_records:
        d = r.get("doc_id")
        if d and d not in doc_ids:
            doc_ids.append(d)

    # 3. Fallback for legacy database records where doc_id was not explicitly set on nodes
    if not doc_ids:
        fallback_records = run_cypher_query("""
            MATCH (n:Entity)
            WHERE n.source IS NOT NULL AND n.source <> ''
            RETURN DISTINCT n.source AS source
            ORDER BY source
        """)
        sources = [r["source"] for r in fallback_records if r.get("source")]
        gen_sources = [s for s in sources if s.startswith("GEN-")]
        non_gen_sources = [s for s in sources if not s.startswith("GEN-")]

        if gen_sources:
            doc_ids.append(f"Stored Requirement Graph ({gen_sources[0]}..{gen_sources[-1]})")
        doc_ids.extend(non_gen_sources)

    return doc_ids


def get_graph_by_document(doc_id: str) -> tuple:
    """Retrieve nodes and relationships specifically belonging to a given document ID."""
    if not doc_id or doc_id.startswith("Stored Requirement Graph") or doc_id.startswith("All"):
        return get_all_nodes(), get_all_relationships()

    nodes_query = """
    MATCH (n:Entity)
    WHERE n.doc_id = $doc_id OR n.source = $doc_id
    RETURN n.id AS id, n.name AS name, n.type AS type, n.description AS description,
           n.source AS source, n.aliases AS aliases, n.sequence AS sequence,
           n.is_flow_entry AS is_flow_entry, n.attributes_json AS attributes_json,
           n.doc_id AS doc_id
    """
    node_records = run_read_query(nodes_query, doc_id=doc_id)
    nodes = []
    seen_ids = set()
    for rec in node_records:
        if rec["id"] in seen_ids:
            continue
        seen_ids.add(rec["id"])
        attrs = {}
        if rec.get("attributes_json"):
            try:
                attrs = json.loads(rec["attributes_json"])
            except Exception:
                pass
        if rec.get("sequence") is not None:
            attrs["sequence"] = str(rec["sequence"])
        if rec.get("is_flow_entry"):
            attrs["is_flow_entry"] = "true"

        nodes.append({
            "id": rec["id"],
            "name": rec.get("name") or rec["id"],
            "type": rec.get("type", "Requirement"),
            "description": rec.get("description"),
            "source": rec.get("source"),
            "aliases": rec.get("aliases") or [],
            "attributes": attrs,
            "doc_id": rec.get("doc_id") or doc_id,
        })

    rels_query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE (a.doc_id = $doc_id OR a.source = $doc_id)
      AND (b.doc_id = $doc_id OR b.source = $doc_id)
    RETURN a.id AS from, b.id AS to, type(r) AS type, r.doc_id AS doc_id
    """
    rel_records = run_read_query(rels_query, doc_id=doc_id)
    rels = []
    seen_rels = set()
    for rec in rel_records:
        key = (rec["from"], rec["to"], rec["type"])
        if key in seen_rels:
            continue
        seen_rels.add(key)
        rels.append({
            "from": rec["from"],
            "to": rec["to"],
            "type": rec["type"],
            "doc_id": rec.get("doc_id") or doc_id,
        })

    return nodes, rels


def run_write_query(query: str, **params) -> list:
    """
    Run any parametrized WRITE Cypher query, returns list of dicts.

    Generic counterpart to insert_graph()/_write_node() above, which
    are hardcoded to this project's original :Entity shape. Added for
    graph/story_graph_writer.py (User Story -> Gherkin flow), whose
    schema (RequirementDoc/Requirement/UserStory/Persona/Feature/
    FlowStep/AcceptanceCriteria/TestCase/TestStep) is a different node
    shape entirely — rather than hand-roll a bespoke _write_x/create_x
    function per label the way the original :Entity writer did, this
    lets that module send its own MERGE...SET Cypher directly, same
    retry/reconnection semantics (_run_with_retry) as every other
    write in this module.
    """
    def fn(session):
        result = session.run(query, **params)
        return [dict(record) for record in result]

    return _run_with_retry(fn) or []


def run_read_query(query: str, **params) -> list:
    """
    Parametrized READ counterpart to run_write_query() above — same
    body, kept as its own name purely so call sites (e.g.
    graph/story_hybrid_retriever.py) read clearly as read vs write,
    same as this module's existing run_cypher_query (no params)
    already does for the main pipeline's ad-hoc queries.
    """
    def fn(session):
        result = session.run(query, **params)
        return [dict(record) for record in result]

    return _run_with_retry(fn) or []


def get_all_nodes() -> list:
    """Get all nodes from Neo4j."""
    def fn(session):
        result = session.run(
            """
            MATCH (n:Entity)
            RETURN
                n.id            AS id,
                n.name          AS name,
                n.type          AS type,
                n.description   AS description,
                n.source        AS source,
                n.aliases       AS aliases,
                n.sequence      AS sequence,
                n.is_flow_entry AS is_flow_entry,
                n.attributes_json AS attributes_json,
                n.batch_id      AS batch_id
            """
        )
        rows = []
        for r in result:
            try:
                attributes = json.loads(r["attributes_json"]) if r["attributes_json"] else {}
            except (TypeError, ValueError):
                attributes = {}
            # sequence/is_flow_entry are stored as their own typed
            # properties (see _write_node) but also folded back into
            # `attributes` here so a node dict read from Neo4j has the
            # SAME shape as one straight from graph/entity_extractor.py
            # — callers (graph/flow_graph_analysis.py, graph_visualizer.py)
            # only need to look in one place either way.
            if r["sequence"] is not None:
                attributes["sequence"] = str(r["sequence"])
            if r["is_flow_entry"]:
                attributes["is_flow_entry"] = "true"
            rows.append({
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "source": r["source"],
                "aliases": r["aliases"],
                "attributes": attributes,
                "batch_id": r["batch_id"],   # None for every node the
                                              # existing main pipeline
                                              # ever wrote -- only set
                                              # for graph/batch_upload.py's
                                              # comparison-flow nodes.
            })
        return rows
    return _run_with_retry(fn) or []


def get_all_relationships() -> list:
    """Get all relationships from Neo4j."""
    def fn(session):
        result = session.run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "RETURN a.id AS from, b.id AS to, type(r) AS rel_type, r.source AS source, r.doc_id AS doc_id"
        )
        return [
            {"from": r["from"], "to": r["to"], "type": r["rel_type"], "source": r["source"], "doc_id": r["doc_id"]}
            for r in result
        ]
    return _run_with_retry(fn) or []


def delete_all_nodes() -> dict:
    """Delete all nodes from the graph."""
    def fn(session):
        result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
        record = result.single()
        return record["deleted"] if record else 0

    deleted = _run_with_retry(fn) or 0
    return {"deleted": deleted}


def get_node_count() -> int:
    """Return the total number of nodes in the graph."""
    def fn(session):
        result = session.run("MATCH (n) RETURN count(n) AS count")
        record = result.single()
        return record["count"] if record else 0

    return _run_with_retry(fn) or 0


def get_community_nodes_db(community_id: int) -> list:
    """Get all nodes in a specific community."""
    def fn(session):
        result = session.run(
            """
            MATCH (n:Entity {community: $community_id})
            RETURN
            n.id          AS id,
            n.name        AS name,
            n.type        AS type,
            n.description AS description,
            n.source      AS source,
           
            n.aliases     AS aliases,
            n.community   AS community
            """,
            community_id=community_id
        )
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "source": r["source"],
                
                "aliases": r["aliases"],
                "community": r["community"],
            }
            for r in result
        ]
    return _run_with_retry(fn) or []


def get_community_relationships_db(community_id: int) -> list:
    """Get all relationships within a community."""
    def fn(session):
        result = session.run(
            """
            MATCH (a:Entity {community: $community_id})
                  -[r]->
                  (b:Entity {community: $community_id})
            RETURN a.id AS from, b.id AS to, type(r) AS rel_type
            """,
            community_id=community_id
        )
        return [
            {"from": r["from"], "to": r["to"], "type": r["rel_type"]}
            for r in result
        ]
    return _run_with_retry(fn) or []


def get_neighbor_nodes_db(node_ids: list, max_neighbors: int = 40) -> dict:
    """
    Live 1-hop Cypher traversal OUT from a given set of node ids —
    the "graph" half of hybrid retrieval for test-case generation
    (graph/hybrid_test_case_generator.py).

    Unlike get_community_nodes_db (which is scoped to one Louvain
    community), this walks from an EXPLICIT node-id set regardless of
    community membership, in either direction. That matters here
    because a requirement series' own subgraph
    (node["source"] in series items) is frequently connected to real
    entities that were extracted under a DIFFERENT series — e.g. a
    shared DataObject, a cross-referenced Business Rule
    (graph/cross_reference_linker.py's edges), a common Screen. Those
    neighbors are genuine graph ground truth, just not originally
    "owned" by this series — so they're returned separately, tagged
    by which input node they're reachable from, rather than merged
    silently into the series' own catalog.

    Args:
        node_ids: the series' own node ids (e.g. from
            negative_scenario_generator._catalog_for_series).
        max_neighbors: cap on distinct neighbor nodes returned, so a
            hub node (see graph/graph_eval.py's hub-dominance warning)
            in the input set can't blow out the prompt token budget —
            same reasoning as hybrid_retriever.py's
            max_nodes_per_community.

    Returns:
        {"nodes": [{"id","name","type","description","reached_from"}, ...],
         "relationships": [{"from","to","type"}, ...]}
        nodes/relationships exclude the input node_ids themselves —
        this is NEW context beyond what the series already has.
    """
    if not node_ids:
        return {"nodes": [], "relationships": []}

    def fn(session):
        result = session.run(
            """
            MATCH (seed:Entity)
            WHERE seed.id IN $node_ids
            MATCH (seed)-[r]-(neighbor:Entity)
            WHERE NOT neighbor.id IN $node_ids
            RETURN DISTINCT
                neighbor.id          AS id,
                neighbor.name        AS name,
                neighbor.type        AS type,
                neighbor.description AS description,
                seed.id               AS reached_from,
                type(r)               AS rel_type,
                startNode(r).id       AS rel_from,
                endNode(r).id         AS rel_to
            LIMIT $limit
            """,
            node_ids=node_ids,
            limit=max_neighbors * 4,  # over-fetch before local dedup/cap below
        )
        rows = [dict(record) for record in result]
        nodes_by_id = {}
        rels = []
        seen_rels = set()
        for r in rows:
            if r["id"] not in nodes_by_id and len(nodes_by_id) < max_neighbors:
                nodes_by_id[r["id"]] = {
                    "id": r["id"],
                    "name": r["name"],
                    "type": r["type"],
                    "description": r.get("description"),
                    "reached_from": r["reached_from"],
                }
            rel_key = (r["rel_from"], r["rel_to"], r["rel_type"])
            allowed_endpoints = set(nodes_by_id.keys()) | set(node_ids)
            if r["rel_from"] in allowed_endpoints and r["rel_to"] in allowed_endpoints and rel_key not in seen_rels:
                seen_rels.add(rel_key)
                rels.append({"from": r["rel_from"], "to": r["rel_to"], "type": r["rel_type"]})
        return {"nodes": list(nodes_by_id.values()), "relationships": rels}

    return _run_with_retry(fn) or {"nodes": [], "relationships": []}


def get_all_communities_db() -> dict:
    """Get all communities and their info."""
    def fn(session):
        result = session.run(
            """
            MATCH (n:Entity)
            WHERE n.community IS NOT NULL
            RETURN n.community AS community_id,
                   collect(n.id)   AS node_ids,
                   collect(n.type) AS types
            """
        )
        communities = {}
        for record in result:
            community_id = int(record["community_id"])
            types        = record["types"]
            type_counts  = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            primary_types = sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            communities[community_id] = {
                "node_count":    len(record["node_ids"]),
                "nodes":         record["node_ids"],
                "primary_types": [t[0] for t in primary_types]
            }
        return communities
    return _run_with_retry(fn) or {}


def write_community_ids(partition: dict) -> dict:
    """Write community IDs to Neo4j nodes."""
    def fn(session):
        for node_id, community_id in partition.items():
            session.execute_write(_write_community, node_id, community_id)

    _run_with_retry(fn)

    community_sizes = {}
    for community_id in partition.values():
        community_sizes[int(community_id)] = \
            community_sizes.get(int(community_id), 0) + 1

    return {
        "total_nodes":       len(partition),
        "total_communities": len(set(partition.values())),
        "community_sizes":   community_sizes,
        "success":           True
    }

# ----- Workflow storage utilities -----
from .models import WorkflowModel, Feature, Step

def store_workflow(workflow: WorkflowModel, doc_id: str = None) -> None:
    """Store extracted workflow into Neo4j.

    Creates Feature nodes (type='Feature') and WorkflowStep nodes (type='WorkflowStep').
    Adds NEXT relationships between sequential steps. Nodes include a 'feature_name'
    attribute to allow later filtering.
    """
    nodes = []
    relationships = []
    for feature in workflow.features:
        feature_node = {
            "id": f"feature_{feature.name.replace(' ', '_')}",
            "name": feature.name,
            "type": "Feature",
            "description": f"Feature representing {feature.name}",
            "source": "workflow_extractor",
            "attributes": {"feature_name": feature.name},
        }
        if doc_id:
            feature_node["doc_id"] = doc_id
        nodes.append(feature_node)
        for step in feature.steps:
            step_node = {
                "id": step.id,
                "name": step.action,
                "type": "WorkflowStep",
                "description": f"Step {step.id} of feature {feature.name}",
                "source": "workflow_extractor",
                "attributes": {
                    "feature_name": feature.name,
                    "actors": step.actors,
                    "systems": step.systems,
                    "data_objects": step.data_objects,
                    "conditions": step.conditions,
                },
            }
            if doc_id:
                step_node["doc_id"] = doc_id
            nodes.append(step_node)
            # Link feature to its first step
            if feature.steps and step.id == feature.steps[0].id:
                rel = {"from": feature_node["id"], "to": step.id, "type": "HAS_STEP"}
                if doc_id:
                    rel["doc_id"] = doc_id
                relationships.append(rel)
            # NEXT relationships
            for nxt in step.next_steps:
                rel = {"from": step.id, "to": nxt, "type": "NEXT"}
                if doc_id:
                    rel["doc_id"] = doc_id
                relationships.append(rel)
    # Insert into Neo4j using existing insert_graph helper
    insert_graph(nodes, relationships)

def store_feature_graph(feature: Feature, doc_id: str = None) -> None:
    """Store a single feature's workflow and supporting entities.
    Thin wrapper around ``store_workflow`` for a single feature.
    """
    store_workflow(WorkflowModel(features=[feature]), doc_id=doc_id)