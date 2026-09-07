"""
graph/failure_trace.py
 This module reuses the dominator
sets graph/flow_graph_analysis.py already computes for a completely
different purpose (and now exposes via analyze_flow()'s "dominators"
key) to build a root-cause candidate list for a failed test case.

Why dominators are the right tool for this: dominator(D) of node N means
"every path to N passes through D" (see flow_graph_analysis.py's own
docstring / the Dragon Book chapter this project is built on). If a test
case fails at node N, every node that dominates N is a step that
DEFINITELY executed successfully before the failure point -- so the
actual defect is either AT the failing node, or in something that is
NOT a dominator (a sibling branch, a structural dependency the flow
walk doesn't see). Ordering the dominator set by sequence position turns
that into a readable "these are the steps that ran before this broke"
trace, instead of a flat unordered set.

This module is READ-ONLY -- it never writes to Neo4j.
"""

from graph.flow_graph_analysis import analyze_flow
from graph.neo4j_manager import get_all_nodes, get_all_relationships


def _tc_node_id(tc_id: str) -> str:
    # Matches graph/test_case_writer.py's own id scheme exactly --
    # kept as a local copy rather than importing test_case_writer to
    # avoid a circular import (test_case_writer doesn't need this
    # module, but this module needs to speak its id format).
    return f"TESTCASE::{tc_id}"


def trace_failure(tc_id: str, nodes: list = None, relationships: list = None) -> dict:
    """
    Given a failed test case's id, returns a backward trace of every
    flow step that must have executed before the point(s) it validates.

    Args:
        tc_id: the test case id, e.g. "TC-COMBINED-req42-HYB-001"
            (NOT the internal "TESTCASE::..." graph node id).
        nodes, relationships: optionally supply the graph directly
            (e.g. the current Streamlit session's data) to avoid an
            extra live Neo4j round-trip. If omitted, pulls the full
            live graph via get_all_nodes()/get_all_relationships() --
            this needs to be the FULL graph, not just the current
            session's extraction batch, for the same reason
            graph/test_case_writer.py's resolution bug mattered: a
            test case's VALIDATES edges can point at nodes from any
            prior document ingest, not just the current session.

    Returns:
        {
            "tc_id": tc_id,
            "found": bool,                 # False if this tc_id has no
                                            # TestCase node in the graph
            "validated_nodes": [node_id, ...],   # what this test case VALIDATES
            "trace": [                     # one entry per validated node
                {
                    "failing_node": node_id,
                    "failing_node_name": str,
                    "in_flow_subgraph": bool,   # False if this node has
                                                 # no flow edges at all --
                                                 # dominator tracing does
                                                 # not apply, said explicitly
                                                 # rather than returning an
                                                 # empty list silently
                    "upstream_steps": [        # dominators, ordered by
                        {"id": ..., "name": ..., "sequence_position": int}
                        ...                     # sequence position in the
                    ],                          # walk -- earliest first,
                                                 # excludes failing_node itself
                },
                ...
            ],
        }
    """
    if nodes is None or relationships is None:
        nodes = get_all_nodes()
        relationships = get_all_relationships()

    node_by_id = {n["id"]: n for n in nodes}
    tc_node_id = _tc_node_id(tc_id)

    tc_node = node_by_id.get(tc_node_id, {})
    tc_type = tc_node.get("attributes", {}).get("type") or tc_node.get("type", "")
    tc_name = tc_node.get("name", tc_id)

    # Check if test case is failure/negative/edge path
    is_negative = any(kw in tc_id.upper() or kw in tc_name.upper() or kw in str(tc_type).upper()
                      for kw in ("NEG", "FAIL", "EDGE", "NEGATIVE", "ERROR", "INVALID"))

    if not is_negative:
        return {
            "tc_id": tc_id,
            "found": True,
            "is_negative": False,
            "error_message": "Traceback operations are restricted to failure/negative/edge test cases. The selected test case appears to be a positive happy path.",
            "validated_nodes": [],
            "trace": [],
        }

    validated_nodes = [
        r["to"] for r in relationships
        if r.get("from") == tc_node_id and r.get("type") == "VALIDATES"
    ]

    flow = analyze_flow(nodes, relationships)
    dominators = flow["dominators"]
    dfn = flow["dfn"]
    in_flow_subgraph = flow["in_flow_subgraph"]

    trace = []
    for target in validated_nodes:
        target_name = node_by_id.get(target, {}).get("name", target)

        # Find 1-hop structural neighbors attached to target or its predecessors
        target_and_preds = (dominators.get(target, set()) | {target}) if target in in_flow_subgraph else {target}
        supporting_structural = []
        for r in relationships:
            frm, to, rtype = r.get("from"), r.get("to"), r.get("type")
            if frm in target_and_preds and to not in target_and_preds:
                other = node_by_id.get(to)
                if other and other.get("type") in {"BusinessRule", "Condition", "Constraint", "DataObject", "SystemComponent"}:
                    supporting_structural.append({
                        "id": to,
                        "name": other.get("name", to),
                        "type": other.get("type"),
                        "relation": rtype,
                    })
            elif to in target_and_preds and frm not in target_and_preds:
                other = node_by_id.get(frm)
                if other and other.get("type") in {"BusinessRule", "Condition", "Constraint", "DataObject", "SystemComponent"}:
                    supporting_structural.append({
                        "id": frm,
                        "name": other.get("name", frm),
                        "type": other.get("type"),
                        "relation": rtype,
                    })

        upstream_ids = dominators.get(target, set()) - {target} if target in in_flow_subgraph else set()
        upstream_sorted = sorted(upstream_ids, key=lambda nid: dfn.get(nid, 0))

        trace.append({
            "failing_node": target,
            "failing_node_name": target_name,
            "in_flow_subgraph": target in in_flow_subgraph,
            "upstream_steps": [
                {
                    "id": nid,
                    "name": node_by_id.get(nid, {}).get("name", nid),
                    "sequence_position": dfn.get(nid),
                }
                for nid in upstream_sorted
            ],
            "supporting_structural_nodes": supporting_structural,
        })

    return {
        "tc_id": tc_id,
        "found": True,
        "is_negative": True,
        "validated_nodes": validated_nodes,
        "trace": trace,
    }


def format_trace_report(result: dict) -> str:
    """
    Human-readable version of trace_failure()'s output, for dropping
    straight into a Streamlit st.text() / st.markdown() call or a
    console print -- app.py wiring left to whoever adds the UI button,
    this is the pure-logic + formatting half.
    """
    if not result["found"]:
        return f"No TestCase node found for '{result['tc_id']}' -- has it been written to the graph yet?"

    lines = [f"Failure trace for {result['tc_id']}:"]
    if not result["trace"]:
        lines.append("  (this test case has no resolved VALIDATES edges to trace from)")

    for entry in result["trace"]:
        lines.append(f"\n  Touched: {entry['failing_node_name']} ({entry['failing_node']})")
        if not entry["in_flow_subgraph"]:
            lines.append("    -- not part of the sequential flow subgraph; no upstream steps to trace (this node is only reached via structural edges, e.g. CONTAINS/USES).")
            continue
        if not entry["upstream_steps"]:
            lines.append("    -- this IS the flow entry point; nothing runs before it.")
            continue
        lines.append("    Steps that ran before this, in order (check these first):")
        for step in entry["upstream_steps"]:
            lines.append(f"      {step['sequence_position']}. {step['name']}")

    return "\n".join(lines)