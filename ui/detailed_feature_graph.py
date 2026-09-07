import streamlit as st
from graph.models import Feature
from graph.neo4j_manager import run_read_query
from graph.graph_visualizer import render_graph


def _load_feature_subgraph(feature: Feature) -> tuple:
    """
    Load nodes and relationships belonging to a specific feature.
    Ensures the WorkflowStep spine, sequential NEXT flow edges, and 1-hop
    supporting entities (Actors, Systems, Screens, DataObjects, Rules)
    are always present and correctly structured.
    """
    feature_name = feature.name
    feature_node_id = f"feature_{feature_name.replace(' ', '_')}"

    # 1. Build base spine nodes and edges directly from feature.steps model
    spine_nodes = {}
    spine_rels = []

    # Feature root node
    spine_nodes[feature_node_id] = {
        "id": feature_node_id,
        "name": feature_name,
        "type": "Feature",
        "description": f"Feature representing {feature_name}",
        "source": "workflow_extractor",
        "attributes": {"feature_name": feature_name},
    }

    for idx, s in enumerate(feature.steps):
        step_id = s.id
        step_name = s.action or f"Step {idx + 1}"
        
        # Avoid exact duplicate name with Feature root node
        if step_name.strip().lower() == feature_name.strip().lower():
            step_name = f"{step_name} (Step {idx + 1})"

        spine_nodes[step_id] = {
            "id": step_id,
            "name": step_name,
            "type": "WorkflowStep",
            "description": f"Step {idx + 1}: {step_name}",
            "source": "workflow_extractor",
            "attributes": {
                "feature_name": feature_name,
                "actors": s.actors,
                "systems": s.systems,
                "data_objects": s.data_objects,
                "conditions": s.conditions,
            },
        }

        # Link Feature to first step
        if idx == 0:
            spine_rels.append({
                "from": feature_node_id,
                "to": step_id,
                "type": "HAS_STEP"
            })

        # Sequential NEXT relationships
        for nxt in s.next_steps:
            if nxt != step_id:  # prevent self-loop
                spine_rels.append({
                    "from": step_id,
                    "to": nxt,
                    "type": "NEXT"
                })

        # Actors
        for actor in s.actors:
            actor_id = f"actor_{actor.lower().replace(' ', '_')}"
            if actor_id not in spine_nodes:
                spine_nodes[actor_id] = {
                    "id": actor_id,
                    "name": actor,
                    "type": "Actor",
                    "source": "workflow_extractor",
                }
            spine_rels.append({
                "from": actor_id,
                "to": step_id,
                "type": "PERFORMS"
            })

        # Systems / SoftwareSystems
        for sys in s.systems:
            sys_id = f"sys_{sys.lower().replace(' ', '_')}"
            if sys_id not in spine_nodes:
                spine_nodes[sys_id] = {
                    "id": sys_id,
                    "name": sys,
                    "type": "SoftwareSystem",
                    "source": "workflow_extractor",
                }
            spine_rels.append({
                "from": step_id,
                "to": sys_id,
                "type": "USES"
            })

        # DataObjects
        for dobj in s.data_objects:
            dobj_id = f"data_{dobj.lower().replace(' ', '_')}"
            if dobj_id not in spine_nodes:
                spine_nodes[dobj_id] = {
                    "id": dobj_id,
                    "name": dobj,
                    "type": "DataObject",
                    "source": "workflow_extractor",
                }
            spine_rels.append({
                "from": step_id,
                "to": dobj_id,
                "type": "USES"
            })

    # 2. Enrich with 1-hop supporting entities from Neo4j / session state if available
    core_ids = set(spine_nodes.keys())
    extra_nodes = {}
    extra_rels = []

    try:
        halo_records = run_read_query(
            """
            MATCH (core:Entity)-[r]-(neighbor:Entity)
            WHERE core.id IN $core_ids
              AND NOT neighbor.id IN $core_ids
            RETURN DISTINCT
                neighbor.id AS id, neighbor.name AS name,
                neighbor.type AS type, neighbor.description AS description,
                neighbor.source AS source, neighbor.attributes_json AS attributes_json
            LIMIT 40
            """,
            core_ids=list(core_ids),
        )
        for rec in halo_records:
            nid = rec.get("id")
            if nid and nid not in spine_nodes and nid not in extra_nodes:
                import json
                try:
                    attrs = json.loads(rec.get("attributes_json") or "{}")
                except Exception:
                    attrs = {}
                extra_nodes[nid] = {
                    "id": nid,
                    "name": rec.get("name") or nid,
                    "type": rec.get("type", "Requirement"),
                    "description": rec.get("description"),
                    "source": rec.get("source"),
                    "attributes": attrs,
                }

        all_ids = list(core_ids | set(extra_nodes.keys()))
        rel_records = run_read_query(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.id IN $all_ids AND b.id IN $all_ids
            RETURN a.id AS from, b.id AS to, type(r) AS type
            """,
            all_ids=all_ids,
        )
        for r in rel_records:
            if r["from"] != r["to"]:
                extra_rels.append({"from": r["from"], "to": r["to"], "type": r["type"]})
    except Exception:
        pass

    # Also check session state fallback for connected nodes
    session_nodes = st.session_state.get("nodes", [])
    session_rels = st.session_state.get("relationships", [])
    if session_nodes:
        session_dict = {n["id"]: n for n in session_nodes}
        for r in session_rels:
            frm, to = r.get("from"), r.get("to")
            if not frm or not to or frm == to:
                continue
            if frm in core_ids and to in session_dict and to not in spine_nodes and to not in extra_nodes:
                extra_nodes[to] = session_dict[to]
                extra_rels.append({"from": frm, "to": to, "type": r.get("type", "RELATED_TO")})
            elif to in core_ids and frm in session_dict and frm not in spine_nodes and frm not in extra_nodes:
                extra_nodes[frm] = session_dict[frm]
                extra_rels.append({"from": frm, "to": to, "type": r.get("type", "RELATED_TO")})

    # 3. Merge nodes and relationships without duplicates or self-loops
    all_nodes_dict = {**spine_nodes, **extra_nodes}
    final_nodes = list(all_nodes_dict.values())

    seen_rels = set()
    final_rels = []
    for r in (spine_rels + extra_rels):
        frm, to, rtype = r["from"], r["to"], r.get("type", "RELATED_TO")
        if frm == to:
            continue  # drop self-loops
        if frm not in all_nodes_dict or to not in all_nodes_dict:
            continue
        rel_key = (frm, to, rtype)
        if rel_key not in seen_rels:
            seen_rels.add(rel_key)
            final_rels.append({"from": frm, "to": to, "type": rtype})

    return final_nodes, final_rels


def render_detailed_feature_graph(feature: Feature):
    """
    Render the sub-graph for a feature using only that feature's nodes
    (WorkflowStep spine + 1-hop supporting entities).
    Falls back to a warning if no nodes are found in the database for this feature.
    """
    st.subheader(f"Detailed Graph: {feature.name}")

    with st.spinner("Loading feature subgraph..."):
        nodes, rels = _load_feature_subgraph(feature)

    if not nodes:
        st.warning(
            f"No graph data found in Neo4j for feature **{feature.name}**. "
            "Make sure you have indexed a document and the workflow layer was stored."
        )
        return

    spine_count = sum(1 for n in nodes if n.get("type") == "WorkflowStep")
    st.caption(
        f"{len(nodes)} node(s) · {len(rels)} relationship(s) · "
        f"{spine_count} workflow step(s) in spine"
    )

    show_graph = st.checkbox(
        "Show Interactive Graph",
        value=len(nodes) <= 80,
        key=f"show_detail_graph_{feature.name}",
    )
    if show_graph:
        import re
        safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', feature.name)
        render_graph(nodes, rels, height=700, key=f"detail_{safe_key}", flow_only=False)
    else:
        st.info("Graph hidden (large). Check the checkbox above to render.")
