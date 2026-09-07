import streamlit as st
from graph.models import WorkflowModel, Feature


def render_workflow_overview(workflow: WorkflowModel):
    """
    Render the workflow overview: a card per feature showing its name,
    step count, and a preview of the first few steps.
    Navigation buttons set st.session_state so the parent (app.py) can
    switch to the feature_flow or detailed_graph sub-view.
    """
    st.subheader("Workflow Overview")
    if not workflow.features:
        st.info("No features detected in the indexed document.")
        return

    st.caption(
        f"{len(workflow.features)} feature(s) identified. "
        "Select a feature to view its sequential workflow."
    )

    # Draw feature cards with arrows between them
    for idx, feature in enumerate(workflow.features):
        with st.container(border=True):
            col_info, col_btns = st.columns([4, 1])

            with col_info:
                step_count = len(feature.steps)
                st.markdown(f"### {idx + 1}. {feature.name}")
                st.caption(f"**{step_count}** step(s)")

                # Preview first 3 steps inline
                preview_steps = feature.steps[:3]
                if preview_steps:
                    preview_parts = []
                    for s in preview_steps:
                        preview_parts.append(s.action or s.id)
                    dots = " → ".join(preview_parts)
                    if step_count > 3:
                        dots += f" → … (+{step_count - 3} more)"
                    st.markdown(f"`{dots}`")

                # Actor summary
                all_actors = []
                for s in feature.steps:
                    all_actors.extend(s.actors or [])
                if all_actors:
                    unique_actors = list(dict.fromkeys(all_actors))[:5]
                    st.caption("Actors: " + ", ".join(unique_actors))

            with col_btns:
                st.write("")  # vertical spacer
                if st.button("▶ Flow", key=f"flow_{idx}", use_container_width=True):
                    st.session_state["selected_feature"] = feature.name
                    st.session_state["view"] = "feature_flow"
                    st.rerun()
                if step_count > 0:
                    if st.button("🔍 Graph", key=f"graph_{idx}", use_container_width=True):
                        st.session_state["selected_feature"] = feature.name
                        st.session_state["view"] = "detailed_graph"
                        st.rerun()

        # Arrow between features
        if idx < len(workflow.features) - 1:
            st.markdown(
                "<div style='text-align:center;font-size:22px;color:#94a3b8;margin:-6px 0'>↓</div>",
                unsafe_allow_html=True,
            )
