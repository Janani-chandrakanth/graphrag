import streamlit as st
from graph.models import Feature


_STEP_COLORS = [
    "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
    "#f59e0b", "#ef4444", "#ec4899", "#14b8a6",
]


def _pill(text: str, color: str = "#64748b") -> str:
    """Return an inline HTML badge/pill for an actor or condition."""
    return (
        f'<span style="display:inline-block;padding:2px 9px;border-radius:12px;'
        f'background:{color}22;color:{color};border:1px solid {color}55;'
        f'font-size:11px;font-weight:600;margin:2px 3px 2px 0">{text}</span>'
    )


def render_feature_flow(feature: Feature):
    """
    Render the sequential flow of a single feature as a vertical step list.
    Each step card shows: step number, action label, actor/system pills,
    and any conditions as an indented note. Branches show both outgoing
    next-steps. At the bottom a button navigates to the detailed graph.
    """
    st.subheader(f"Feature Flow: {feature.name}")

    if not feature.steps:
        st.info("No steps detected for this feature.")
        if st.button("Back to Overview", key=f"btn_back_overview_empty_{feature.name}"):
            st.session_state["view"] = "overview"
            st.session_state["selected_feature"] = None
            st.rerun()
        return

    st.caption(f"{len(feature.steps)} step(s) in this workflow")
    st.write("")

    for i, step in enumerate(feature.steps):
        color = _STEP_COLORS[i % len(_STEP_COLORS)]

        with st.container(border=True):
            # Step header
            step_label = step.action or step.id
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px'>"
                f"<span style='background:{color};color:white;border-radius:50%;width:28px;"
                f"height:28px;display:flex;align-items:center;justify-content:center;"
                f"font-weight:700;font-size:13px;flex-shrink:0'>{i+1}</span>"
                f"<span style='font-size:15px;font-weight:600;color:#0f172a'>{step_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Actor / system pills
            pills_html = ""
            for actor in (step.actors or []):
                pills_html += _pill(f"Actor: {actor}", "#3b82f6")
            for sys in (step.systems or []):
                pills_html += _pill(f"{sys}", "#8b5cf6")
            for obj in (step.data_objects or []):
                pills_html += _pill(f"{obj}", "#06b6d4")
            if pills_html:
                st.markdown(pills_html, unsafe_allow_html=True)

            # Conditions
            for cond in (step.conditions or []):
                st.markdown(
                    f"<div style='margin-top:4px;padding:4px 10px;background:#fef9c3;"
                    f"border-left:3px solid #f59e0b;border-radius:4px;font-size:12px;"
                    f"color:#78350f'>⚡ {cond}</div>",
                    unsafe_allow_html=True,
                )

            # Branch indicator
            if len(step.next_steps) > 1:
                st.caption(f"↳ branches to: {', '.join(step.next_steps)}")

        # Arrow between steps (not after last)
        if i < len(feature.steps) - 1:
            st.markdown(
                "<div style='text-align:center;font-size:20px;color:#94a3b8;margin:-4px 0'>↓</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.divider()
    col_back, col_graph = st.columns(2)
    with col_back:
        if st.button("← Back to Overview", key=f"btn_back_overview_{feature.name}", use_container_width=True):
            st.session_state["view"] = "overview"
            st.session_state["selected_feature"] = None
            st.rerun()
    with col_graph:
        if st.button("Show Detailed Graph", key=f"go_detail_{feature.name}", use_container_width=True):
            st.session_state["view"] = "detailed_graph"
            st.rerun()
