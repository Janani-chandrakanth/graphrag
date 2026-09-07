import streamlit as st
from graph.models import Feature
from graph.neo4j_manager import run_read_query
from graph.graph_visualizer import render_graph


def _load_screen_subgraph(feature_name: str) -> tuple:
    """
    Load the Screen and UIElement nodes associated with a feature,
    along with their navigation edges, and the sequence steps connecting them.
    """
    # Find all WorkflowSteps that belong to this feature
    # and any Screen or UIElement they touch.
    records = run_read_query(
        """
        MATCH (f:Entity {type: 'Feature', name: $feature_name})
        OPTIONAL MATCH (f)-[:HAS_STEP]->(s:Entity {type: 'WorkflowStep'})
        WITH collect(s.id) as step_ids, f.id as f_id
        
        MATCH (n:Entity)
        WHERE n.id IN step_ids OR n.attributes_json CONTAINS $feature_name
        WITH collect(DISTINCT n.id) as core_ids
        
        // Find screens and UI elements connected to the core steps
        MATCH (core:Entity)-[r]-(neighbor:Entity)
        WHERE core.id IN core_ids
          AND neighbor.type IN ['Screen', 'UIElement', 'SystemComponent', 'Interface', 'Dashboard']
        WITH core_ids, collect(DISTINCT neighbor.id) as neighbor_ids
        WITH core_ids + neighbor_ids as all_ids
        
        // Return only the UI/Screen/Action nodes
        MATCH (a:Entity)
        WHERE a.id IN all_ids 
          AND a.type IN ['Screen', 'UIElement', 'SystemComponent', 'Interface', 'Dashboard', 'WorkflowStep', 'Action', 'Actor']
        
        RETURN a.id AS id, a.name AS name, a.type AS type,
               a.description AS description, a.source AS source,
               a.attributes_json AS attributes_json
        """,
        feature_name=feature_name,
    )

    if not records:
        return [], []

    nodes = []
    seen_ids = set()
    for rec in records:
        if not rec.get("id") or rec["id"] in seen_ids:
            continue
        seen_ids.add(rec["id"])
        
        import json
        try:
            attrs = json.loads(rec.get("attributes_json") or "{}")
        except Exception:
            attrs = {}
            
        nodes.append({
            "id": rec["id"],
            "name": rec.get("name") or rec["id"],
            "type": rec.get("type", "Unknown"),
            "description": rec.get("description"),
            "source": rec.get("source"),
            "attributes": attrs,
        })

    # Relationships between these nodes
    rels = []
    if seen_ids:
        rel_records = run_read_query(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.id IN $all_ids AND b.id IN $all_ids
            RETURN a.id AS from, b.id AS to, type(r) AS type
            """,
            all_ids=list(seen_ids),
        )
        rels = [{"from": r["from"], "to": r["to"], "type": r["type"]} for r in rel_records]

    return nodes, rels


def render_screen_flow(feature: Feature):
    """
    Render a graph focused purely on UI elements, Screens, and how the user navigates them.
    """
    st.subheader(f"Screen Flow: {feature.name}")
    st.caption("Visualizing screen navigation, UI elements, and interactions for this feature.")

    with st.spinner("Loading screen subgraph..."):
        nodes, rels = _load_screen_subgraph(feature.name)

    if not nodes:
        st.info("No screen or UI element data found for this feature.")
        return
        
    screen_count = sum(1 for n in nodes if n.get("type") in ['Screen', 'UIElement', 'SystemComponent', 'Interface', 'Dashboard'])
    st.caption(f"Found {screen_count} UI/Screen components and {len(rels)} interactions.")

    if screen_count == 0:
        st.warning("No UI components or screens were detected in the text for this feature. The text might be backend-focused.")

    import re
    safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', feature.name)
    render_graph(nodes, rels, height=650, key=f"screenflow_{safe_key}", flow_only=False)
