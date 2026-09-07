import streamlit as st
from graph.models import Feature
from graph.neo4j_manager import run_read_query

def render_test_coverage(feature: Feature):
    """
    Render test coverage metrics and a list of test cases that cover this feature.
    """
    st.subheader(f"Test Coverage: {feature.name}")
    st.caption("Tracking test case coverage and failure trace analysis for this feature.")

    with st.spinner("Loading test coverage data..."):
        records = run_read_query(
            """
            MATCH (f:Entity {type: 'Feature', name: $feature_name})
            OPTIONAL MATCH (f)-[:HAS_STEP]->(s:Entity {type: 'WorkflowStep'})
            WITH collect(s.id) as step_ids, f.id as f_id
            
            MATCH (n:Entity)
            WHERE n.id IN step_ids OR n.attributes_json CONTAINS $feature_name
            WITH f_id, collect(DISTINCT n.id) as n_ids
            WITH n_ids + [f_id] as core_ids
            
            MATCH (tc:Entity {type: 'TestCase'})-[r:VALIDATES]->(target:Entity)
            WHERE target.id IN core_ids
            RETURN tc.id AS tc_id, tc.name AS tc_name, tc.description AS description, 
                   collect(DISTINCT target.name) as validated_targets
            """,
            feature_name=feature.name,
        )

    if not records:
        st.info(f"No test cases recorded for {feature.name}.")
        return

    st.write(f"**{len(records)}** Test Cases validate components of this feature.")
    
    st.write("### Test Cases")
    from graph.failure_trace import trace_failure
    for rec in records:
        tc_id = rec.get("tc_id") or rec.get("tc_name")
        with st.expander(f"{rec['tc_name']} - {rec.get('description', 'Test Case')}"):
            st.write("**Validates:**")
            targets = rec.get('validated_targets', [])
            if targets:
                st.write(", ".join([f"`{t}`" for t in targets]))

            if st.button(f"Trace Failure Backward", key=f"btn_trace_{tc_id}"):
                trace_res = trace_failure(tc_id)
                if not trace_res.get("found"):
                    st.warning("Test case node not found in Neo4j.")
                elif not trace_res.get("is_negative", True):
                    st.error(trace_res.get("error_message"))
                else:
                    st.success("Backward Failure Trace:")
                    for tr in trace_res.get("trace", []):
                        st.write(f"**Failing Point:** `{tr['failing_node_name']}`")
                        if tr.get("upstream_steps"):
                            st.write("Sequential Predecessors:")
                            for up in tr["upstream_steps"]:
                                st.write(f"- Step {up.get('sequence_position', '?')}: `{up['name']}`")
                        if tr.get("supporting_structural_nodes"):
                            st.write("Supporting Structural Context:")
                            for st_node in tr["supporting_structural_nodes"]:
                                st.write(f"- `{st_node['name']}` ({st_node['type']}) via `{st_node['relation']}`")
