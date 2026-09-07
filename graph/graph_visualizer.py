"""
graph_visualizer.py

Drop-in replacement for the streamlit_agraph-based rendering in app.py.
Renders an interactive vis-network graph with:
  - hierarchical (top-down / left-right) layout instead of a tangled
    force-directed blob
  - click a node -> incoming nodes/edges highlight red, outgoing
    highlight green, everything else dims
  - an info panel showing the selected node's neighbors
  - a search box to locate a node by name
  - a legend built from the same NODE_COLORS map app.py already has

Usage (see PATCH_NOTES.md for the exact app.py diff):

    from graph.graph_visualizer import render_graph, render_cypher_graph

These two functions have the SAME signatures as the ones they replace,
so no other call sites in app.py need to change.
"""

import math
import json
import html as _html
import streamlit.components.v1 as components

from graph.flow_graph_analysis import analyze_flow, FLOW_RELATIONS

NODE_COLORS = {
    "Actor":                    "#0D9488",  # Teal
    "Role":                     "#0D9488",  # Teal
    "Feature":                  "#059669",  # Emerald Green
    "Requirement":              "#D97706",  # Amber
    "NonFunctionalRequirement": "#B45309",  # Dark Amber
    "BusinessRule":             "#7C3AED",  # Violet
    "Condition":                "#9333EA",  # Purple
    "Action":                   "#2563EB",  # Royal Blue (NO RED)
    "WorkflowStep":             "#1D4ED8",  # Deep Blue
    "WorkflowState":            "#4F46E5",  # Indigo State
    "Output":                   "#10B981",  # Mint / Success
    "SystemComponent":          "#475569",  # Slate
    "SoftwareSystem":           "#334155",  # Dark Slate
    "DataObject":               "#0284C7",  # Light Blue
    "Constraint":               "#D97706",  # Amber (NO RED)
    "Event":                    "#6366F1",  # Indigo Event
    "State":                    "#64748B",  # Neutral Slate
    "Screen":                   "#0284C7",  # UI Screen
    "UIElement":                "#0EA5E9",  # UI Element
    "Attribute":                "#94A3B8",  # Soft Slate
    "TestCase":                 "#CA8A04",  # Gold
    "Error":                    "#DC2626",  # Red - ONLY for actual errors/failures
    "Failure":                  "#DC2626",  # Red - ONLY for actual errors/failures
    "Rejected":                 "#DC2626",  # Red - ONLY for actual errors/failures
}
DEFAULT_COLOR = "#94A3B8"

NODE_SHAPES = {
    "Actor":                    "triangle",
    "Role":                     "triangle",
    "Feature":                  "hexagon",
    "Requirement":              "box",
    "NonFunctionalRequirement": "box",
    "BusinessRule":             "diamond",
    "Condition":                "diamond",
    "Action":                   "box",
    "WorkflowStep":             "box",
    "WorkflowState":            "ellipse",
    "Output":                   "star",
    "SystemComponent":          "database",
    "SoftwareSystem":           "database",
    "DataObject":               "database",
    "Constraint":               "diamond",
    "Event":                    "dot",
    "State":                    "ellipse",
    "Screen":                   "box",
    "UIElement":                "dot",
    "Attribute":                "dot",
    "TestCase":                 "star",
    "Error":                    "diamond",
    "Failure":                  "diamond",
    "Rejected":                 "diamond",
}
DEFAULT_SHAPE = "box"

# vis-network shape glyphs shown in the legend so the shape/type
# mapping is visible without hovering every node type — plain
# characters, not an icon font dependency.
_SHAPE_GLYPH = {
    "dot": "●", "square": "■", "box": "▭", "triangle": "▲",
    "triangleDown": "▼", "star": "★", "diamond": "◆",
    "hexagon": "⬡", "ellipse": "⬭", "database": "🗄",
}


def get_node_color(node_type):
    return NODE_COLORS.get(node_type, DEFAULT_COLOR)


def get_node_shape(node_type):
    return NODE_SHAPES.get(node_type, DEFAULT_SHAPE)

# ── Layout budget for the toolbar + legend strip above the network,
# used only to size the surrounding components.html() iframe so it's
# tall enough to show everything instead of clipping it. Does not
# affect the graph area itself (#net_KEY keeps its exact `height` as
# before) or any visual styling — legend/toolbar CSS is unchanged.
_TOOLBAR_H = 54
_LEGEND_ROW_H = 30
# Rough estimate of how many legend chips fit on one row at the
# container's typical rendered width — used only to guess how many
# rows the legend will wrap onto so the iframe is sized tall enough.
# If the real wrap needs more rows than guessed, scrolling=True (set
# where components.html() is called) means nothing is ever clipped
# and unreachable — worst case is an extra scroll, never lost content.
_ASSUMED_LEGEND_ITEMS_PER_ROW = 6


# Long node names (a BR/FR whose "name" is its full sentence, e.g. "The
# system must suggest currencies based on the combination of...") were
# rendering as raw overlapping text across the whole canvas — this was
# the single biggest cause of the "clumsy" look, more than layout. We
# truncate what's drawn on the node; the untruncated text always stays
# available in the hover tooltip and the click side-panel, so nothing
# is lost, just not force-fit onto a 20px dot.
_LABEL_LINE_LEN = 16
_LABEL_MAX_LINES = 2


def _wrap_label(text, line_len=_LABEL_LINE_LEN, max_lines=_LABEL_MAX_LINES):
    text = (text or "").strip()
    words = text.split()
    if not words:
        return text
    lines, cur_words, idx = [], [], 0
    while idx < len(words) and len(lines) < max_lines:
        trial = " ".join(cur_words + [words[idx]]).strip()
        if len(trial) <= line_len or not cur_words:
            cur_words.append(words[idx])
            idx += 1
        else:
            lines.append(" ".join(cur_words))
            cur_words = []
    if cur_words:
        lines.append(" ".join(cur_words))
    if idx < len(words) and lines:
        lines[-1] = lines[-1].rstrip() + "…"
    return "\n".join(lines) if lines else text[:line_len]


def _estimate_legend_rows(legend_types):
    if not legend_types:
        return 1
    return max(1, math.ceil(len(legend_types) / _ASSUMED_LEGEND_ITEMS_PER_ROW))


def _build_vis_payload(nodes, relationships, flow_only=False):
    """
    nodes: list of {"id","name","type", ...}
    relationships: list of {"from","to","type", ...}
    flow_only: if True, render ONLY the FLOW_RELATIONS subgraph (the
        actual sequential procedure) instead of the full entity graph.

        Why this exists: analyze_flow() already computes a correct
        sequence_order/entry/back-edge/loop analysis over the flow
        subset -- but rendering it as annotations laid on top of the
        FULL graph (structural + flow edges together) means the
        structural container edges (Requirement CONTAINS/PART_OF/USES
        many sub-entities -- which is CORRECT hub-shaped information,
        not a bug) always visually dominate a numbered flow step
        sitting right next to them. A BRD/user-story document is
        mostly structural content with a minority of true procedural
        steps inside it; trying to render both shapes on one canvas
        makes the correct sequence unreadable. flow_only gives a
        second, separate view that shows just the procedure.
    Returns (vis_nodes, vis_edges, legend_types) as JSON-serializable lists.
    """
    # degree count (ALL relation types) so hub nodes render slightly
    # bigger -- this is a display-density signal, unrelated to flow.
    degree = {}
    for r in relationships:
        degree[r["from"]] = degree.get(r["from"], 0) + 1
        degree[r["to"]] = degree.get(r["to"], 0) + 1

    # Real flow-graph analysis (graph/flow_graph_analysis.py), restricted
    # to the relation types that actually denote procedural sequence
    # (FLOW_RELATIONS) -- NOT "any node with zero incoming edges of any
    # type", which mismarked nodes only reachable via structural edges
    # like SUPPORTS/PART_OF as flow starts, then had nothing coherent
    # to walk from them since those were never step-transition edges.
    flow = analyze_flow(nodes, relationships)

    if flow_only:
        in_flow = flow["in_flow_subgraph"]
        nodes = [n for n in nodes if n["id"] in in_flow]
        relationships = [
            r for r in relationships
            if r.get("type") in FLOW_RELATIONS and r["from"] in in_flow and r["to"] in in_flow
        ]
        if not nodes:
            return [], [], []
    start_ids = set(flow["entry_ids"])
    seq_number = {nid: i + 1 for i, nid in enumerate(flow["sequence_order"])}
    back_edge_set = set(flow["back_edges"])
    # every node id that's part of at least one natural loop, for a
    # subtle "this is inside a loop" tooltip note independent of
    # whether THIS node happens to be the loop header
    loop_member_ids = set()
    for nl in flow["natural_loops"]:
        loop_member_ids |= nl["nodes"]

    vis_nodes = []
    seen_types = set()
    for n in nodes:
        ntype = n.get("type", "Unknown")
        seen_types.add(ntype)
        d = degree.get(n["id"], 0)
        full_name = n.get("name", n["id"])
        
        # User Feedback: "Instead of mentioning the node as system can we name it someother term"
        if full_name.lower() == "system":
            full_name = "End User"
            
        is_start = n["id"] in start_ids
        step_no = seq_number.get(n["id"])
        label_prefix = f"{step_no}. " if step_no else ""
        label = _wrap_label(("▶ " + full_name) if is_start else (label_prefix + full_name))
        tooltip_lines = [full_name, "", f"Type: {ntype}", f"ID: {n['id']}"]
        if is_start:
            tooltip_lines.append("\n▶ FLOW START" + (" (best-guess -- no Actor/zero-indegree entry found)" if flow.get("entry_is_guessed") else ""))
        elif step_no:
            tooltip_lines.append(f"\nFlow step #{step_no} (depth-first order)")
        if n["id"] in loop_member_ids:
            tooltip_lines.append("↻ part of a loop (natural loop / retry path)")
            
        node_entry = {
            "id": n["id"],
            "label": label,
            "group": ntype,
            "shape": get_node_shape(ntype),
            "color": get_node_color(ntype),
            "value": 8 + min(d, 12) * 2,   # visual size scaled by degree
            "title": "\n".join(tooltip_lines),
        }
        
        diff_status = n.get("diff_status")
        if diff_status:
            if diff_status == "added":
                node_entry["color"] = {
                    "background": "#E8F5E9",
                    "border": "#2ECC71",
                    "highlight": {"background": "#C8E6C9", "border": "#27AE60"}
                }
                node_entry["borderWidth"] = 4
                node_entry["label"] = "[+] " + node_entry["label"]
                tooltip_lines.append("\n★ STATUS: ADDED (New in this version)")
            elif diff_status == "removed":
                node_entry["color"] = {
                    "background": "#FFEBEE",
                    "border": "#E74C3C",
                    "highlight": {"background": "#FFCDD2", "border": "#C0392B"}
                }
                node_entry["borderWidth"] = 3
                node_entry["shapeProperties"] = {"borderDashes": True}
                node_entry["label"] = "[-] " + node_entry["label"]
                tooltip_lines.append("\n✖ STATUS: REMOVED (Deleted in this version)")
            elif diff_status == "modified":
                node_entry["color"] = {
                    "background": "#E3F2FD",
                    "border": "#3498DB",
                    "highlight": {"background": "#BBDEFB", "border": "#2980B9"}
                }
                node_entry["borderWidth"] = 4
                node_entry["label"] = "[~] " + node_entry["label"]
                tooltip_lines.append("\n✏ STATUS: MODIFIED (Properties changed)")
            node_entry["title"] = "\n".join(tooltip_lines)
        elif is_start:
            # gold ring around the node's own color, on top of the
            # ▶ label prefix — two cues so it still reads once you've
            # clicked into a dimmed/highlighted state elsewhere.
            node_entry["color"] = {"background": get_node_color(ntype), "border": "#F5B301"}
            node_entry["borderWidth"] = 4
            node_entry["isStart"] = True
            
        vis_nodes.append(node_entry)

    vis_edges = []
    for i, r in enumerate(relationships):
        edge_key = (r["from"], r["to"])
        is_back_edge = edge_key in back_edge_set
        edge_entry = {
            "id": f"e{i}",
            "from": r["from"],
            "to": r["to"],
            "label": r.get("type", ""),
        }
        
        diff_status = r.get("diff_status")
        if diff_status:
            if diff_status == "added":
                edge_entry["color"] = {"color": "#2ECC71", "opacity": 0.95}
                edge_entry["width"] = 3
                edge_entry["label"] = (edge_entry["label"] + " (+)").strip()
            elif diff_status == "removed":
                edge_entry["color"] = {"color": "#E74C3C", "opacity": 0.8}
                edge_entry["width"] = 2
                edge_entry["dashes"] = True
                edge_entry["label"] = (edge_entry["label"] + " (-)").strip()
        elif is_back_edge:
            # A verified loop-back edge (dominance-checked, not just
            # "points to something visited earlier") — rendered dashed
            # + amber + curved, so a loop reads as "this returns to an
            # earlier step" instead of looking like a broken/backwards
            # sequence in the layout.
            edge_entry["dashes"] = True
            edge_entry["color"] = {"color": "#F5B301", "opacity": 0.85}
            edge_entry["smooth"] = {"type": "curvedCW", "roundness": 0.3}
            edge_entry["label"] = (r.get("type", "") + "  ↻ loop back").strip()
            
        vis_edges.append(edge_entry)

    return vis_nodes, vis_edges, sorted(seen_types)


def _render_html(vis_nodes, vis_edges, legend_types, height=750, key="graph", detailed_analysis=None):
    if not vis_nodes:
        return None

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    detailed_json = json.dumps(detailed_analysis) if detailed_analysis else "null"

    legend_html = "".join(
        f'<span class="legend-item"><span class="dot" '
        f'style="background:{_html.escape(get_node_color(t))}"></span>'
        f'<span class="shape-glyph" style="color:{_html.escape(get_node_color(t))}">'
        f'{_html.escape(_SHAPE_GLYPH.get(get_node_shape(t), "●"))}</span>{_html.escape(t)}</span>'
        for t in legend_types
    )
    legend_html += (
        '<span class="legend-item legend-start-note">'
        '<span style="color:#F5B301;font-weight:700;">▶ gold ring</span>'
        ' = flow start &nbsp;|&nbsp; '
        '<span style="color:#F5B301;font-weight:700;">- - ↻</span>'
        ' dashed amber edge = loop-back (natural loop)</span>'
    )

    template = """
<div id="wrap_KEY" class="graphwrap">
  <style>
    .graphwrap { font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
    .toolbar_KEY {
      display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
      padding: 8px 10px; background: #f8fafc; border: 1px solid #e2e8f0;
      border-bottom: none; border-radius: 8px 8px 0 0; font-size: 13px;
    }
    .toolbar_KEY input[type=text] {
      padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 6px;
      font-size: 13px; width: 200px;
    }
    .toolbar_KEY select {
      padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px;
    }
    .toolbar_KEY button {
      padding: 5px 12px; border: 1px solid #cbd5e1; border-radius: 6px;
      background: white; cursor: pointer; font-size: 13px;
    }
    .toolbar_KEY button:hover { background: #f1f5f9; }
    .legend_KEY {
      display: flex; gap: 10px; flex-wrap: wrap; padding: 6px 10px;
      background: #f8fafc; border: 1px solid #e2e8f0; border-top: none;
      font-size: 12px; color: #475569;
    }
    .legend-item { display: flex; align-items: center; gap: 4px; }
    .shape-glyph { font-size: 11px; }
    .legend-start-note { color: #92400e; }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .main_KEY { display: flex; border: 1px solid #e2e8f0; border-top: none; border-radius: 0 0 8px 8px; overflow: hidden; }
    #net_KEY { flex: 1; height: HEIGHTpx; background: #ffffff; }
    #panel_KEY {
      width: 280px; height: HEIGHTpx; overflow-y: auto; padding: 12px;
      background: #f8fafc; border-left: 1px solid #e2e8f0; font-size: 13px; color: #334155;
    }
    #panel_KEY h4 { margin: 0 0 6px 0; font-size: 14px; color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; }
    #panel_KEY .sec { margin-top: 12px; }
    #panel_KEY .sec-title { font-weight: 600; margin-bottom: 4px; color: #475569; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
    #panel_KEY .in-item { color: #ef4444; margin-bottom: 4px; font-size: 12px; }
    #panel_KEY .out-item { color: #16a34a; margin-bottom: 4px; font-size: 12px; }
    #panel_KEY .hint { color: #94a3b8; font-size: 12px; }
  </style>

  <div class="toolbar_KEY">
    <input type="text" id="search_KEY" placeholder="Find node by name..." />
    <button onclick="doSearch_KEY()">Find</button>
    <select id="dir_KEY" onchange="setDirection_KEY()">
      <option value="UD" selected>Top &rarr; Down (Flowchart)</option>
      <option value="LR">Left &rarr; Right (Flowchart)</option>
      <option value="NONE">Free layout</option>
    </select>
    <button onclick="resetHighlight_KEY()">Reset highlight</button>
    <button onclick="fitAll_KEY()">Fit to screen</button>
    
    <!-- Sequential Flow Navigation Panel -->
    <select id="flow_select_KEY" onchange="changeFlow_KEY()" style="display:none; margin-left:12px;">
      <option value="-1">-- Select Actor Flow --</option>
    </select>
    <button id="prev_btn_KEY" onclick="prevStep_KEY()" style="display:none;">&larr; Previous</button>
    <button id="next_btn_KEY" onclick="nextStep_KEY()" style="display:none;">Next &rarr;</button>
    <button id="full_flow_btn_KEY" onclick="showFullFlow_KEY()" style="display:none;">Show Full Flow</button>
    <span id="step_info_KEY" style="margin-left: 8px; font-weight: 600; color: #475569;"></span>
  </div>
  <div class="legend_KEY">LEGENDHTML</div>
  <div class="main_KEY">
    <div id="net_KEY"></div>
    <div id="panel_KEY"><span class="hint">Click any node to see its incoming / outgoing connections.</span></div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/dist/vis-network.min.js"></script>
<script>
(function() {
  var rawNodes_KEY = NODESJSON;
  var rawEdges_KEY = EDGESJSON;
  var detailedAnalysis_KEY = DETAILED_JSON;

  var nodesDataSet_KEY = new vis.DataSet(rawNodes_KEY.map(function(n) {
    return Object.assign({}, n, { opacity: 1, borderWidth: 1.5 });
  }));
  var edgesDataSet_KEY = new vis.DataSet(rawEdges_KEY.map(function(e) {
    return Object.assign({}, e, {
      relType: e.label,
      label: "",
      arrows: { to: { enabled: true, scaleFactor: 0.7 } },
      color: { color: "#94a3b8", opacity: 0.95 },
      width: 1.5,
      font: { size: 10, color: "#475569", strokeWidth: 2, strokeColor: "#ffffff", align: "middle" }
    });
  }));

  var container_KEY = document.getElementById("net_KEY");
  var data_KEY = { nodes: nodesDataSet_KEY, edges: edgesDataSet_KEY };

  function hierOptions(direction) {
    return {
      layout: {
        hierarchical: {
          enabled: true,
          direction: direction,
          sortMethod: "directed",
          levelSeparation: 150,
          nodeSpacing: 180,
          treeSpacing: 200,
          blockShifting: true,
          edgeMinimization: true,
          parentCentralization: true
        }
      },
      physics: {
        hierarchicalRepulsion: { nodeDistance: 160, springLength: 120, avoidOverlap: 0.8 },
        solver: "hierarchicalRepulsion",
        stabilization: { enabled: true, iterations: 100, updateInterval: 25, fit: true }
      },
      edges: {
        smooth: {
          type: "cubicBezier",
          forceDirection: direction === "LR" ? "horizontal" : "vertical",
          roundness: 0.45
        }
      }
    };
  }

  function freeOptions() {
    return {
      layout: { hierarchical: { enabled: false } },
      physics: {
        solver: "forceAtlas2Based",
        forceAtlas2Based: {
          gravitationalConstant: -120,
          springLength: 200,
          springConstant: 0.02,
          avoidOverlap: 0
        },
        stabilization: { enabled: true, iterations: 120, updateInterval: 25, fit: false }
      },
      edges: { smooth: { type: "continuous" } }
    };
  }

  var baseOptions_KEY = {
    nodes: {
      shape: "box",
      margin: { top: 7, bottom: 7, left: 12, right: 12 },
      font: { size: 13, face: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color: "#0f172a", strokeWidth: 0 },
      borderWidth: 1.5,
      shadow: { enabled: true, color: "rgba(0,0,0,0.06)", size: 5, x: 1, y: 2 }
    },
    groups: {},
    interaction: {
      hover: true,
      tooltipDelay: 120,
      navigationButtons: true,
      keyboard: true,
      multiselect: false
    }
  };

  // ── Large-graph fast path: if > 150 nodes, skip physics entirely
  var LARGE_GRAPH = rawNodes_KEY.length > 150;
  var DEFAULT_DIRECTION_KEY = "UD";
  if (LARGE_GRAPH) {
    baseOptions_KEY = Object.assign({}, baseOptions_KEY, {
      physics: { enabled: false }
    });
  }

  function mergedOptions(direction) {
    var extra = direction === "NONE" ? freeOptions() : hierOptions(direction);
    var merged = Object.assign({}, baseOptions_KEY, extra);
    if (LARGE_GRAPH) {
      merged.physics = { enabled: false };
    }
    return merged;
  }

  var network_KEY = new vis.Network(container_KEY, data_KEY, mergedOptions(DEFAULT_DIRECTION_KEY));

  // ── Disable physics as soon as the network stabilises.
  // We re-attach this listener every time the layout changes so that
  // switching from Free ↔ LR ↔ UD never leaves physics running forever.
  function attachStabilizationOff_KEY() {
    network_KEY.once("stabilizationIterationsDone", function() {
      network_KEY.setOptions({ physics: { enabled: false } });
    });
  }
  attachStabilizationOff_KEY();

  window.setDirection_KEY = function() {
    var dir = document.getElementById("dir_KEY").value;
    var opts = mergedOptions(dir);
    network_KEY.setOptions(opts);
    if (!LARGE_GRAPH) {
      attachStabilizationOff_KEY();
    }
  };

  window.fitAll_KEY = function() {
    network_KEY.fit({ animation: true });
  };

  window.doSearch_KEY = function() {
    var q = document.getElementById("search_KEY").value.trim().toLowerCase();
    if (!q) return;
    var match = rawNodes_KEY.find(function(n) {
      return (n.label || "").toLowerCase().indexOf(q) !== -1;
    });
    if (match) {
      network_KEY.selectNodes([match.id]);
      network_KEY.focus(match.id, { scale: 1.1, animation: true });
      highlightNode_KEY(match.id);
    }
  };

  function neighborsOf(nodeId) {
    var incoming = [], outgoing = [], incomingNodeIds = new Set(), outgoingNodeIds = new Set();
    rawEdges_KEY.forEach(function(e) {
      if (e.to === nodeId) { incoming.push(e); incomingNodeIds.add(e.from); }
      if (e.from === nodeId) { outgoing.push(e); outgoingNodeIds.add(e.to); }
    });
    return { incoming: incoming, outgoing: outgoing, incomingNodeIds: incomingNodeIds, outgoingNodeIds: outgoingNodeIds };
  }

  function labelOf(id) {
    var n = nodesDataSet_KEY.get(id);
    return n ? n.label : id;
  }

  function highlightNode_KEY(nodeId) {
    var rel = neighborsOf(nodeId);

    var DIM_EDGE_FONT = { size: 10, color: "rgba(148,163,184,0.25)", strokeWidth: 0 };
    var LIT_EDGE_FONT = { size: 11, color: "#334155", strokeWidth: 3, strokeColor: "#ffffff" };
    var edgeUpdates = rawEdges_KEY.map(function(e) {
      var isIn = rel.incoming.indexOf(e) !== -1;
      var isOut = rel.outgoing.indexOf(e) !== -1;
      if (isIn) return { id: e.id, color: { color: "#ef4444", opacity: 1 }, width: 2.5, font: LIT_EDGE_FONT, label: e.relType || e.label };
      if (isOut) return { id: e.id, color: { color: "#16a34a", opacity: 1 }, width: 2.5, font: LIT_EDGE_FONT, label: e.relType || e.label };
      return { id: e.id, color: { color: "#e2e8f0", opacity: 0.25 }, width: 1, font: DIM_EDGE_FONT, label: "" };
    });
    edgesDataSet_KEY.update(edgeUpdates);

    var DIM_NODE_FONT = { size: 13, color: "rgba(30,41,59,0.15)", strokeWidth: 0 };
    var LIT_NODE_FONT = { size: 14, color: "#1e293b", strokeWidth: 4, strokeColor: "#ffffff" };
    var nodeUpdates = rawNodes_KEY.map(function(n) {
      if (n.id === nodeId) return { id: n.id, opacity: 1, borderWidth: 4, borderWidthSelected: 4, font: { size: 16, color: "#0f172a", strokeWidth: 4, strokeColor: "#ffffff" } };
      if (rel.incomingNodeIds.has(n.id)) return { id: n.id, opacity: 1, borderWidth: 3, font: LIT_NODE_FONT };
      if (rel.outgoingNodeIds.has(n.id)) return { id: n.id, opacity: 1, borderWidth: 3, font: LIT_NODE_FONT };
      return { id: n.id, opacity: 0.15, borderWidth: 1, font: DIM_NODE_FONT };
    });
    nodesDataSet_KEY.update(nodeUpdates);

    updatePanelForNode_KEY(nodeId, rel);
  }

  function updatePanelForNode_KEY(nodeId, rel) {
    if (!rel) {
      rel = neighborsOf(nodeId);
    }
    var n = nodesDataSet_KEY.get(nodeId);
    var panel = document.getElementById("panel_KEY");
    var html = "<h4>" + labelOf(nodeId) + "</h4>";
    
    html += '<div class="sec">';
    html += '<div><strong>Type:</strong> ' + (n.group || "Unknown") + '</div>';
    html += '<div><strong>Requirement:</strong> ' + (n.source || "N/A") + '</div>';
    html += '<div><strong>Confidence:</strong> ' + (n.confidence || "N/A") + '</div>';
    html += '</div>';

    if (n.isolated_reason) {
      html += '<div class="sec">';
      html += '<div class="sec-title" style="color:#ef4444;">Status: Isolated</div>';
      html += '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;padding:8px;margin-top:6px;color:#991b1b;font-size:11px;line-height:1.4;">';
      html += '<strong>Reason:</strong><br/>' + n.isolated_reason;
      html += '</div>';
      html += '</div>';
    }

    html += '<div class="sec"><div class="sec-title">Incoming (' + rel.incoming.length + ')</div>';
    if (rel.incoming.length === 0) html += '<span class="hint">none</span>';
    rel.incoming.forEach(function(e) {
      html += '<div class="in-item">&larr; ' + labelOf(e.from) + ' <span class="hint">[' + (e.relType||e.label||"") + ']</span></div>';
    });
    html += '</div>';
    html += '<div class="sec"><div class="sec-title">Outgoing (' + rel.outgoing.length + ')</div>';
    if (rel.outgoing.length === 0) html += '<span class="hint">none</span>';
    rel.outgoing.forEach(function(e) {
      html += '<div class="out-item">&rarr; ' + labelOf(e.to) + ' <span class="hint">[' + (e.relType||e.label||"") + ']</span></div>';
    });
    html += '</div>';
    panel.innerHTML = html;
  }

  window.resetHighlight_KEY = function() {
    edgesDataSet_KEY.update(rawEdges_KEY.map(function(e) {
      return {
        id: e.id, color: { color: "#cbd5e1", opacity: 0.9 }, width: 1, label: "",
        font: { size: 10, color: "#64748b", strokeWidth: 3, strokeColor: "#ffffff", align: "top" }
      };
    }));
    nodesDataSet_KEY.update(rawNodes_KEY.map(function(n) {
      return {
        id: n.id, opacity: 1, borderWidth: 2,
        font: { size: 14, color: "#1e293b", strokeWidth: 4, strokeColor: "#ffffff" }
      };
    }));
    document.getElementById("panel_KEY").innerHTML =
      '<span class="hint">Click any node to see its incoming / outgoing connections.</span>';
    network_KEY.unselectAll();
    currentFlowIndex = -1;
    updateFlowUI_KEY();
  };

  network_KEY.on("click", function(params) {
    if (params.nodes.length > 0) {
      // If we are currently in flow navigation, update the flow step index if the clicked node is in the flow
      if (currentFlowIndex !== -1) {
        var flow = detailedAnalysis_KEY.actor_flows[currentFlowIndex];
        var idx = flow.sequence.indexOf(params.nodes[0]);
        if (idx !== -1) {
          currentStepIndex = idx;
          var stepInfo = document.getElementById("step_info_KEY");
          if (stepInfo) stepInfo.textContent = "Step " + (currentStepIndex + 1) + " of " + flow.sequence.length;
          highlightFlowPath_KEY(flow.sequence, currentStepIndex);
          return;
        }
      }
      highlightNode_KEY(params.nodes[0]);
    } else {
      window.resetHighlight_KEY();
      if (currentFlowIndex !== -1) {
        currentStepIndex = -1;
        var stepInfo = document.getElementById("step_info_KEY");
        if (stepInfo) stepInfo.textContent = "Flow selected: click Next to start";
      }
    }
  });

  // Flow Navigation logic
  var currentFlowIndex = -1;
  var currentStepIndex = -1;

  if (detailedAnalysis_KEY && detailedAnalysis_KEY.actor_flows && detailedAnalysis_KEY.actor_flows.length > 0) {
    var flowSelector = document.getElementById("flow_select_KEY");
    if (flowSelector) {
      flowSelector.style.display = "inline-block";
      detailedAnalysis_KEY.actor_flows.forEach(function(flow, index) {
        var opt = document.createElement("option");
        opt.value = index;
        opt.textContent = flow.actor_name + " (" + flow.sequence.length + " steps)";
        flowSelector.appendChild(opt);
      });
    }
  }

  window.changeFlow_KEY = function() {
    var val = document.getElementById("flow_select_KEY").value;
    currentFlowIndex = parseInt(val);
    currentStepIndex = -1;
    updateFlowUI_KEY();
  };

  function updateFlowUI_KEY() {
    var prevBtn = document.getElementById("prev_btn_KEY");
    var nextBtn = document.getElementById("next_btn_KEY");
    var fullBtn = document.getElementById("full_flow_btn_KEY");
    var stepInfo = document.getElementById("step_info_KEY");

    if (currentFlowIndex === -1) {
      if (prevBtn) prevBtn.style.display = "none";
      if (nextBtn) nextBtn.style.display = "none";
      if (fullBtn) fullBtn.style.display = "none";
      if (stepInfo) stepInfo.textContent = "";
      window.resetHighlight_KEY();
      return;
    }

    if (prevBtn) prevBtn.style.display = "inline-block";
    if (nextBtn) nextBtn.style.display = "inline-block";
    if (fullBtn) fullBtn.style.display = "inline-block";

    var flow = detailedAnalysis_KEY.actor_flows[currentFlowIndex];
    if (currentStepIndex === -1) {
      if (stepInfo) stepInfo.textContent = "Flow selected: click Next to start";
      window.resetHighlight_KEY();
    } else {
      var nodeId = flow.sequence[currentStepIndex];
      if (stepInfo) stepInfo.textContent = "Step " + (currentStepIndex + 1) + " of " + flow.sequence.length;
      network_KEY.selectNodes([nodeId]);
      network_KEY.focus(nodeId, { scale: 1.1, animation: true });
      highlightFlowPath_KEY(flow.sequence, currentStepIndex);
    }
  }

  window.nextStep_KEY = function() {
    if (currentFlowIndex === -1) return;
    var flow = detailedAnalysis_KEY.actor_flows[currentFlowIndex];
    if (currentStepIndex < flow.sequence.length - 1) {
      currentStepIndex++;
      updateFlowUI_KEY();
    }
  };

  window.prevStep_KEY = function() {
    if (currentFlowIndex === -1) return;
    if (currentStepIndex > 0) {
      currentStepIndex--;
      updateFlowUI_KEY();
    }
  };

  window.showFullFlow_KEY = function() {
    if (currentFlowIndex === -1) return;
    var flow = detailedAnalysis_KEY.actor_flows[currentFlowIndex];
    highlightFlowPath_KEY(flow.sequence, flow.sequence.length - 1, true);
    network_KEY.fit({ nodes: flow.sequence, animation: true });
  };

  function highlightFlowPath_KEY(sequence, activeIndex, showAll) {
    var seqSet = new Set(sequence);
    var pathNodes = sequence.slice(0, activeIndex + 1);
    var pathSet = new Set(pathNodes);
    var activeNodeId = sequence[activeIndex];

    var DIM_NODE_FONT = { size: 13, color: "rgba(30,41,59,0.15)", strokeWidth: 0 };
    var LIT_NODE_FONT = { size: 14, color: "#1e293b", strokeWidth: 4, strokeColor: "#ffffff" };

    var nodeUpdates = rawNodes_KEY.map(function(n) {
      var isSeq = seqSet.has(n.id);
      var isPath = pathSet.has(n.id);

      if (n.id === activeNodeId) {
        return {
          id: n.id,
          opacity: 1,
          borderWidth: 4,
          borderWidthSelected: 4,
          font: { size: 16, color: "#0f172a", strokeWidth: 4, strokeColor: "#ffffff" }
        };
      } else if (isPath || (showAll && isSeq)) {
        return { id: n.id, opacity: 1, borderWidth: 2, font: LIT_NODE_FONT };
      } else {
        return { id: n.id, opacity: 0.15, borderWidth: 1, font: DIM_NODE_FONT };
      }
    });
    nodesDataSet_KEY.update(nodeUpdates);

    var DIM_EDGE_FONT = { size: 10, color: "rgba(148,163,184,0.1)", strokeWidth: 0 };
    var LIT_EDGE_FONT = { size: 11, color: "#334155", strokeWidth: 3, strokeColor: "#ffffff" };

    var edgeUpdates = rawEdges_KEY.map(function(e) {
      var fromIdx = sequence.indexOf(e.from);
      var toIdx = sequence.indexOf(e.to);
      var isFlowEdge = (fromIdx !== -1 && toIdx !== -1 && toIdx === fromIdx + 1);

      if (isFlowEdge && (fromIdx < activeIndex || showAll)) {
        return {
          id: e.id,
          color: { color: "#16a34a", opacity: 1 },
          width: 3,
          font: LIT_EDGE_FONT,
          label: e.relType || e.label
        };
      } else {
        return {
          id: e.id,
          color: { color: "#cbd5e1", opacity: 0.1 },
          width: 1,
          font: DIM_EDGE_FONT,
          label: ""
        };
      }
    });
    edgesDataSet_KEY.update(edgeUpdates);

    if (activeNodeId) {
      updatePanelForNode_KEY(activeNodeId);
    }
  }

  var MIN_INITIAL_SCALE = 0.55;
  function fitCapped_KEY(minScale) {
    var ids = nodesDataSet_KEY.getIds();
    if (ids.length === 0) return;
    var positions = network_KEY.getPositions(ids);
    var xs = ids.map(function(id) { return positions[id].x; });
    var ys = ids.map(function(id) { return positions[id].y; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
    var canvasW = container_KEY.clientWidth || 900;
    var canvasH = container_KEY.clientHeight || 600;
    var scale = Math.min(canvasW / (w + 240), canvasH / (h + 240));
    scale = Math.max(scale, minScale);
    scale = Math.min(scale, 1.3);
    network_KEY.moveTo({
      position: { x: (minX + maxX) / 2, y: (minY + maxY) / 2 },
      scale: scale,
      animation: false
    });
  }

  network_KEY.once("stabilizationIterationsDone", function() {
    fitCapped_KEY(MIN_INITIAL_SCALE);
  });
})();
</script>
"""
    out = template.replace("KEY", key)
    out = out.replace("NODESJSON", nodes_json)
    out = out.replace("EDGESJSON", edges_json)
    out = out.replace("DETAILED_JSON", detailed_json)
    out = out.replace("LEGENDHTML", legend_html)
    out = out.replace("HEIGHTpx", f"{height}px")
    return out


def render_graph(nodes, relationships, height=750, key="maingraph", flow_only=False, entry_ids=None):
    """
    Renders an interactive vis-network graph with flow analysis, isolated nodes reasoning,
    and a custom flow navigator panel.
    """
    import streamlit as st
    if not nodes:
        st.warning("No nodes to visualize.")
        return

    from graph.flow_graph_analysis import analyze_flow_detailed

    # Perform detailed analysis
    detailed = analyze_flow_detailed(nodes, relationships, entry_ids)

    # Annotate nodes with source requirement, confidence score, and isolation reasons
    annotated_nodes = []
    for n in nodes:
        n_copy = dict(n)
        nid = n["id"]
        n_copy["source"] = n.get("source") or "Unknown"
        n_copy["confidence"] = n.get("attributes", {}).get("confidence") or n.get("confidence") or "N/A"
        if nid in detailed["isolated_nodes"]:
            n_copy["isolated_reason"] = detailed["isolated_nodes"][nid]["reason"]
        annotated_nodes.append(n_copy)

    vis_nodes, vis_edges, legend_types = _build_vis_payload(annotated_nodes, relationships, flow_only=flow_only)

    if flow_only and not vis_nodes:
        st.info(
            "No sequential procedure detected in this graph — every relationship "
            "here is structural (USES/PART_OF/CONTAINS/etc.), not a step-to-step "
            "flow."
        )
        return

    html_str = _render_html(vis_nodes, vis_edges, legend_types, height=height, key=key, detailed_analysis=detailed)
    legend_rows = _estimate_legend_rows(legend_types)
    extra_height = _TOOLBAR_H + (legend_rows * _LEGEND_ROW_H) + 40
    components.html(html_str, height=height + extra_height, scrolling=True)


def render_diff_graph(diff, version_a, version_b, height=750, key="comparegraph"):
    """
    Renders a unified diff graph comparing version A (older) and version B (newer) on a single canvas.
    - Added elements: bright green border, green fill
    - Removed elements: bright red border, red dashes, semi-transparent overlay
    - Modified elements: blue border, light blue fill
    - Unchanged elements: default styling
    """
    import streamlit as st
    if not version_b.get("nodes") and not version_a.get("nodes"):
        st.warning("No version nodes to compare.")
        return

    # Track status of nodes
    added_ids = {n["id"] for n in diff.get("added_nodes", [])}
    removed_ids = {n["id"] for n in diff.get("removed_nodes", [])}
    modified_ids = {n["id"] for n in diff.get("modified_nodes", [])}

    # Track status of relationships
    def rel_key(r):
        return (r.get("from"), r.get("to"), r.get("type"))

    added_rel_keys = {rel_key(r) for r in diff.get("added_relationships", [])}
    removed_rel_keys = {rel_key(r) for r in diff.get("removed_relationships", [])}

    # Combine nodes and assign diff_status
    combined_nodes = []
    seen_ids = set()

    for n in version_b.get("nodes", []):
        nid = n["id"]
        n_copy = dict(n)
        if nid in added_ids:
            n_copy["diff_status"] = "added"
        elif nid in modified_ids:
            n_copy["diff_status"] = "modified"
        else:
            n_copy["diff_status"] = "unchanged"
        combined_nodes.append(n_copy)
        seen_ids.add(nid)

    for n in version_a.get("nodes", []):
        nid = n["id"]
        if nid in removed_ids and nid not in seen_ids:
            n_copy = dict(n)
            n_copy["diff_status"] = "removed"
            combined_nodes.append(n_copy)
            seen_ids.add(nid)

    # Combine relationships and assign diff_status
    combined_rels = []
    for r in version_b.get("relationships", []):
        r_copy = dict(r)
        if rel_key(r) in added_rel_keys:
            r_copy["diff_status"] = "added"
        else:
            r_copy["diff_status"] = "unchanged"
        combined_rels.append(r_copy)

    for r in version_a.get("relationships", []):
        if rel_key(r) in removed_rel_keys:
            r_copy = dict(r)
            r_copy["diff_status"] = "removed"
            combined_rels.append(r_copy)

    # Render combined diff using our vis builder
    vis_nodes, vis_edges, legend_types = _build_vis_payload(combined_nodes, combined_rels, flow_only=False)

    html_str = _render_html(vis_nodes, vis_edges, legend_types, height=height, key=key, detailed_analysis=None)
    legend_rows = _estimate_legend_rows(legend_types)
    extra_height = _TOOLBAR_H + (legend_rows * _LEGEND_ROW_H) + 40
    components.html(html_str, height=height + extra_height, scrolling=True)


def render_cypher_graph(records, height=650, key="cyphergraph"):
    """
    Renders cypher query results with standard entity coloring.
    """
    from neo4j.graph import Node as NeoNode, Relationship as NeoRel

    seen_nodes = {}
    rels = []
    for record in records:
        for value in record.values():
            if isinstance(value, NeoNode):
                node_id = str(value.id)
                if node_id not in seen_nodes:
                    seen_nodes[node_id] = {
                        "id": node_id,
                        "name": value.get("name", node_id),
                        "type": value.get("type", "Unknown"),
                    }
            elif isinstance(value, NeoRel):
                rels.append({
                    "from": str(value.start_node.id),
                    "to": str(value.end_node.id),
                    "type": value.type,
                })

    if not seen_nodes:
        return False

    vis_nodes, vis_edges, legend_types = _build_vis_payload(list(seen_nodes.values()), rels)
    html_str = _render_html(vis_nodes, vis_edges, legend_types, height=height, key=key)
    legend_rows = _estimate_legend_rows(legend_types)
    extra_height = _TOOLBAR_H + (legend_rows * _LEGEND_ROW_H) + 40
    components.html(html_str, height=height + extra_height, scrolling=True)
    return True