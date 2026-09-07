"""
graph/flow_graph_analysis.py — real flow-graph analysis (dominators,
depth-first spanning tree, back-edge / natural-loop detection),
applied ONLY to the subset of this project's relationship types that
actually denote procedural sequence.

WHY THIS MODULE EXISTS

graph_visualizer.py's original "flow start = zero incoming edges"
heuristic treated every relationship type in graph/schemas.py's
ALLOWED_RELATIONS as if it meant "this happens, then that happens."
It doesn't: USES/PART_OF/SUPPORTS/DEPENDS_ON/ASSOCIATED_WITH/
REFERENCES/... are structural or justificatory edges (dependency,
composition, evidence), not control-flow edges. A node can have zero
incoming SUPPORTS/USES edges and still not be where a user's journey
begins — which is exactly the bug: a node reachable only by an
outgoing SUPPORTS edge got marked as a flow start, then "the next
step" didn't follow, because SUPPORTS was never a step transition to
begin with.

This module is the fix: restrict analysis to FLOW_RELATIONS (the
subset that genuinely means "leads to the next state/action"), then
apply the actual algorithms from Aho/Lam/Sethi/Ullman ("Compilers:
Principles, Techniques, and Tools") Section 9.6, "Loops in Flow
Graphs" — the same chapter this was requested from:

  - Depth-first search / depth-first spanning tree (Algorithm 9.41)
  - Edge classification: tree / forward (advancing), back / cross
    (retreating vs cross, per the DFST ancestor test in 9.6.3)
  - Dominators, via the classic iterative data-flow algorithm
    (Algorithm 9.38): D(entry) = {entry}; D(n) = {n} UNION
    (intersection of D(p) for every predecessor p of n), iterated to
    a fixed point.
  - True back edges: a retreating edge a -> b is only a genuine BACK
    edge (and therefore the start of a natural LOOP) if b actually
    DOMINATES a (b in D(a)) — not just "b is an ancestor in this one
    DFST", which is what distinguishes a real loop from an artifact of
    DFS traversal order (the chapter's "reducibility" discussion).
  - Natural loops, via Algorithm 9.46: for back edge n -> d, the loop
    is {d} plus every node that can reach n without passing through d.

None of this runs on the full heterogeneous entity graph — only on
the flow subgraph. Nodes/edges outside FLOW_RELATIONS are left alone
by this module entirely; graph_visualizer.py still draws them, they
just don't get a step number or "flow start" marker, because they
were never really flow edges.
"""

# Relation types that denote an actual state/action transition — "this
# happens, then that happens" — as opposed to a structural,
# compositional, or evidentiary relationship. Kept as an explicit
# allow-list (not "everything except a deny-list") so a newly added
# relationship type in graph/schemas.py defaults to being treated as
# non-sequential until someone deliberately decides otherwise — the
# safer default, since silently mis-classifying a structural edge as
# "flow" is exactly the bug this module exists to fix.
FLOW_RELATIONS = {
    "TRIGGERS", "LEADS_TO", "CAUSES", "CREATES", "UPDATES", "GENERATES",
    "NOTIFIES", "EXECUTES", "PRODUCES", "PERFORMS", "AUTHENTICATES",
    "PROCESSES", "VALIDATES", "MONITORS",
}


def _build_flow_adjacency(nodes: list, relationships: list) -> dict:
    """Returns {node_id: [successor_id, ...]} restricted to
    FLOW_RELATIONS, for every node in `nodes` (isolated nodes get an
    empty successor list, not omitted).

    Successors are ordered by the prompt-provided `sequence` attribute
    (prompts/kg_extraction_prompt.py's SEQUENCE / FLOW EXTRACTION
    RULES) when the extraction supplied one, ascending, with
    un-sequenced successors sorted after sequenced ones (stable sort,
    so ties/unknowns keep their original relationship-list order).
    This is what lets the DFS below actually walk the order the source
    text described, instead of whatever order the LLM happened to list
    the relationships in.

    Excludes any edge touching a TestCase node, regardless of relation
    type. Reason: graph/test_case_writer.py's traceability edges
    (TestCase -[VALIDATES]-> domain_node) intentionally reuse the
    domain schema's existing "VALIDATES" relationship type (see
    graph/schemas.py) rather than inventing a new one -- which
    collides with FLOW_RELATIONS' own "VALIDATES" (used for real
    domain semantics like "OTP validates login"). Without this
    exclusion, every TestCase node written to the graph would get
    treated as a flow participant, corrupting entry detection and
    dominators for the ENTIRE graph, not just the test case's own
    corner of it -- discovered via graph/failure_trace.py's own test
    fixture, which is exactly the feature this would have silently
    broken. TestCase nodes are traceability metadata, never real
    steps in a business flow, so excluding them here is correct
    regardless of which specific relation type connects them.
    """
    METADATA_NODE_TYPES = {"testcase", "documentmetadata", "author", "pageartifact", "metadata", "versionhistory"}
    excluded_ids = {n["id"] for n in nodes if str(n.get("type", "")).lower() in METADATA_NODE_TYPES}
    adj = {n["id"]: [] for n in nodes if str(n.get("type", "")).lower() not in METADATA_NODE_TYPES}

    for r in relationships:
        if r.get("type") not in FLOW_RELATIONS:
            continue
        if r["from"] in excluded_ids or r["to"] in excluded_ids:
            continue
        if r["from"] in adj and r["to"] in adj:
            adj[r["from"]].append(r["to"])


    seq_by_id = {}
    for n in nodes:
        raw = (n.get("attributes") or {}).get("sequence")
        try:
            seq_by_id[n["id"]] = int(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            seq_by_id[n["id"]] = None

    for nid, succs in adj.items():
        adj[nid] = sorted(
            succs,
            key=lambda s: (seq_by_id.get(s) is None, seq_by_id.get(s) if seq_by_id.get(s) is not None else 0),
        )
    return adj


def _predecessors(adj: dict) -> dict:
    preds = {nid: [] for nid in adj}
    for nid, succs in adj.items():
        for s in succs:
            preds[s].append(nid)
    return preds


def _find_entry_nodes(adj: dict, preds: dict, nodes: list) -> list:
    """
    Chapter 9.6's algorithms all take ENTRY as a GIVEN INPUT (Algorithm
    9.38's signature is literally "INPUT: A flow graph G ... and entry
    node ENTRY") — a compiler always knows its function's entry block
    ahead of time. A knowledge graph doesn't come with that label, so
    this has to be inferred — UNLESS the extraction prompt already
    said so directly.

    Three-tier heuristic, highest-confidence first:
      1. Nodes the extraction prompt itself tagged
         attributes["is_flow_entry"] == "true" (prompts/
         kg_extraction_prompt.py's SEQUENCE / FLOW EXTRACTION RULES
         section) — this is the LLM reading the source text
         top-to-bottom and saying directly "this is where the
         described procedure begins," which is a real signal no
         amount of post-hoc topology analysis can recover once it's
         been lost. Always trusted first when present.
      2. Nodes of type "Actor" with an outgoing flow edge — in this
         project's domain, the human/system that initiates a
         requirement flow IS the natural starting point, absent an
         explicit tag.
      3. Genuine zero-in-degree nodes in the flow subgraph — works
         fine for acyclic/partial flows (common for a single short
         requirement's extracted sequence).
      None found -> caller (analyze_flow) falls back further to
      max-out-degree, clearly documented there as a last resort.
    """
    tagged_entries = [
        n["id"] for n in nodes
        if n["id"] in adj and adj[n["id"]]
        and str((n.get("attributes") or {}).get("is_flow_entry", "")).strip().lower() == "true"
    ]
    if tagged_entries:
        return tagged_entries

    actor_types = {n["id"] for n in nodes if n.get("type") == "Actor"}
    actor_entries = [nid for nid in adj if nid in actor_types and adj[nid]]
    if actor_entries:
        return actor_entries
    return [nid for nid in adj if adj[nid] and not preds[nid]]


def _dfs_spanning_forest(adj: dict, entries: list):
    """
    Algorithm 9.41 (iterative, to avoid Python recursion-depth limits
    on a large graph). Runs one DFS per entry node, then — so that
    every node still gets a depth-first number even if it's not
    reachable from any detected entry (e.g. two disjoint flow islands
    where one island's "entry" wasn't picked up) — continues over any
    still-unvisited nodes as additional roots, same fallback spirit as
    the old code's "if nothing found, don't just drop the rest."

    Returns:
        dfn: {node_id: preorder discovery index}, 0-based, in the
             order nodes were first visited.
        finish_order: {node_id: postorder finish index}, used for edge
             classification below.
        parent: {node_id: parent_id_in_DFST or None}
        tree_edges: set of (u, v) that are DFST tree edges.
        roots: the actual node ids DFS started from (entries, plus any
             fallback roots needed to cover disconnected remainders).
    """
    visited = set()
    finished = set()
    dfn, finish_order, parent = {}, {}, {}
    tree_edges = set()
    counter = [0]
    finish_counter = [0]
    roots = []

    def dfs_from(root):
        roots.append(root)
        stack = [(root, iter(adj[root]))]
        visited.add(root)
        dfn[root] = counter[0]; counter[0] += 1
        parent[root] = None
        while stack:
            node, it = stack[-1]
            advanced = False
            for succ in it:
                if succ not in visited:
                    visited.add(succ)
                    dfn[succ] = counter[0]; counter[0] += 1
                    parent[succ] = node
                    tree_edges.add((node, succ))
                    stack.append((succ, iter(adj[succ])))
                    advanced = True
                    break
            if not advanced:
                finish_order[node] = finish_counter[0]; finish_counter[0] += 1
                finished.add(node)
                stack.pop()

    for e in entries:
        if e not in visited:
            dfs_from(e)
    # cover any remaining unvisited nodes so every node still gets a
    # dfn/finish (needed for edge classification below to be total)
    for nid in adj:
        if nid not in visited:
            dfs_from(nid)

    return dfn, finish_order, parent, tree_edges, roots


def _classify_edges(adj: dict, dfn: dict, finish_order: dict, tree_edges: set) -> dict:
    """
    Per Section 9.6.3: for a non-tree edge (m, n),
      - if dfn[n] > dfn[m] and n hasn't finished before m started in a
        way that makes it a descendant -> forward (advancing) edge
      - if dfn[n] < dfn[m] and finish_order[n] > finish_order[m]
        (n is still "open"/an ancestor when we found the edge) -> back
        (retreating) edge — a CANDIDATE; whether it's a TRUE back edge
        is decided by dominance in classify_back_edges() below, not
        here (this function only reproduces the DFST-relative
        forward/back/cross split, same as the book's raw definition).
      - otherwise -> cross edge

    Returns {"tree": [...], "forward": [...], "back": [...], "cross": [...]}
    as lists of (u, v).
    """
    out = {"tree": list(tree_edges), "forward": [], "back": [], "cross": []}
    for u, succs in adj.items():
        for v in succs:
            if (u, v) in tree_edges:
                continue
            if u not in dfn or v not in dfn:
                continue
            if dfn[v] > dfn[u]:
                out["forward"].append((u, v))
            elif finish_order.get(v, -1) > finish_order.get(u, -1):
                # v started before u and hadn't finished yet when u ran
                # -> v is an ancestor of u in the DFST -> retreating
                out["back"].append((u, v))
            else:
                out["cross"].append((u, v))
    return out


def _compute_dominators(adj: dict, preds: dict, roots: list) -> dict:
    """
    Algorithm 9.38, iterative worklist version, run per weakly-connected
    component (each with its own root from `roots`) since the classic
    algorithm assumes a single entry node.

    Returns {node_id: set(node_ids that dominate it)}.
    """
    all_ids = set(adj.keys())
    dom = {nid: set(all_ids) for nid in all_ids}
    for root in roots:
        dom[root] = {root}

    changed = True
    while changed:
        changed = False
        for nid in all_ids:
            if nid in roots:
                continue
            p_list = preds[nid]
            if not p_list:
                continue
            new_dom = set.intersection(*(dom[p] for p in p_list)) | {nid}
            if new_dom != dom[nid]:
                dom[nid] = new_dom
                changed = True
    return dom


def _natural_loop(adj: dict, preds: dict, n: str, d: str) -> set:
    """Algorithm 9.46: natural loop of back edge n -> d. Reverse
    reachability from n, stopping at d."""
    loop = {n, d}
    stack = [n]
    seen = {n, d}
    while stack:
        cur = stack.pop()
        for p in preds.get(cur, []):
            if p not in seen:
                seen.add(p)
                loop.add(p)
                stack.append(p)
    return loop


def analyze_flow(nodes: list, relationships: list, entry_ids: list = None) -> dict:
    """
    Main entry point. Returns:
        {
          "entry_ids": [...],                # true flow entries (§ find_entry_nodes)
          "dfn": {node_id: int},              # depth-first discovery order
          "sequence_order": [node_id, ...],   # nodes sorted by dfn — "Step 1, 2, 3..."
          "back_edges": [(u, v), ...],        # TRUE back edges (dominance-verified)
          "natural_loops": [{"header": d, "nodes": {...}, "back_edges": [(n, d), ...]}, ...],
          "in_flow_subgraph": set(node_id),   # nodes that have >=1 flow edge (in or out)
          "dominators": {node_id: set(node_id)},  # every node that must be
                                               # passed through to reach this
                                               # one -- the basis for
                                               # backward failure-tracing
                                               # (see graph/failure_trace.py)
        }
    Nodes/edges with no flow-relevant relationships at all are simply
    absent from sequence_order / never marked as entries — they are
    NOT flow-classified as "no start found", they're correctly left
    alone, since there's nothing in FLOW_RELATIONS to say where they'd
    sit in a sequence.

    Args:
        entry_ids: override _find_entry_nodes()'s heuristic with an
            explicit, known entry set. Chapter 9.6's own algorithms
            take ENTRY as a given input rather than deriving it (see
            _find_entry_nodes()'s docstring) — this parameter exists
            so callers that DO know the true entry (or test fixtures
            validating against the book's own worked example, where
            "entry node 1" is stated up front, not computed) can supply
            it directly instead of going through the heuristic.
    """
    adj = _build_flow_adjacency(nodes, relationships)
    preds = _predecessors(adj)
    in_flow_subgraph = {nid for nid in adj if adj[nid] or preds[nid]}

    entries = entry_ids if entry_ids is not None else _find_entry_nodes(adj, preds, nodes)
    entry_is_guessed = False
    if not entries and in_flow_subgraph:
        # Nothing matched either heuristic tier (e.g. no Actor node AND
        # every flow node has some incoming flow edge) — pick the
        # highest-out-degree node in the flow subgraph as a pragmatic
        # stand-in so analysis still runs, same "don't just give up"
        # fallback spirit as the rest of this project. This is
        # explicitly a heuristic of last resort, not a claim of
        # correctness — flagged to the caller via "entry_is_guessed" so
        # the UI can show it with lower confidence than a real Actor-
        # or zero-in-degree-derived entry.
        entries = [max(in_flow_subgraph, key=lambda nid: len(adj[nid]))]
        entry_is_guessed = True

    dfn, finish_order, parent, tree_edges, roots = _dfs_spanning_forest(adj, entries)
    edge_classes = _classify_edges(adj, dfn, finish_order, tree_edges)
    dominators = _compute_dominators(adj, preds, roots)

    true_back_edges = [
        (u, v) for (u, v) in edge_classes["back"]
        if v in dominators.get(u, set())
    ]

    natural_loops_raw = []
    for (n, d) in true_back_edges:
        loop_nodes = _natural_loop(adj, preds, n, d)
        natural_loops_raw.append({"header": d, "nodes": loop_nodes, "back_edge": (n, d)})

    # Per the chapter's explicit rule: natural loops sharing the same
    # header are combined into a single loop rather than kept separate
    # (Example 9.48 — "we shall combine them into a single loop").
    # A loop nested inside another with a DIFFERENT header stays
    # separate; only a same-header collision gets merged here.
    loops_by_header = {}
    for loop in natural_loops_raw:
        h = loop["header"]
        if h not in loops_by_header:
            loops_by_header[h] = {"header": h, "nodes": set(loop["nodes"]), "back_edges": [loop["back_edge"]]}
        else:
            loops_by_header[h]["nodes"] |= loop["nodes"]
            loops_by_header[h]["back_edges"].append(loop["back_edge"])
    natural_loops = list(loops_by_header.values())

    sequence_order = [nid for nid in sorted(dfn, key=lambda k: dfn[k]) if nid in in_flow_subgraph]

    return {
        "entry_ids": entries,
        "entry_is_guessed": entry_is_guessed,
        "dfn": dfn,
        "sequence_order": sequence_order,
        "back_edges": true_back_edges,
        "natural_loops": natural_loops,
        "in_flow_subgraph": in_flow_subgraph,
        "dominators": dominators,
    }


def analyze_flow_detailed(nodes: list, relationships: list, entry_ids: list = None) -> dict:
    """
    Performs detailed workflow analysis including per-actor flow separation,
    isolated node classification with explanations, branches/merges, and health metrics.
    """
    flow_analysis = analyze_flow(nodes, relationships, entry_ids)
    
    # 1. Adjacency for flow and full graph
    adj_flow = _build_flow_adjacency(nodes, relationships)
    preds_flow = _predecessors(adj_flow)
    in_flow_subgraph = flow_analysis["in_flow_subgraph"]
    
    # Adjacency for ALL relationships (to detect true isolation)
    adj_full = {n["id"]: [] for n in nodes}
    preds_full = {n["id"]: [] for n in nodes}
    for r in relationships:
        f, t = r.get("from"), r.get("to")
        if f in adj_full and t in adj_full:
            adj_full[f].append(t)
            preds_full[t].append(f)
            
    # 2. Identify isolated nodes and assign reasons
    isolated_nodes = {}
    actor_node_ids = {n["id"] for n in nodes if n.get("type") == "Actor"}
    
    for n in nodes:
        nid = n["id"]
        ntype = n.get("type", "Unknown")
        if nid in in_flow_subgraph:
            continue
            
        # Node has absolutely no connections in the input relationships list
        if not adj_full[nid] and not preds_full[nid]:
            isolated_nodes[nid] = {
                "name": n.get("name", nid),
                "type": ntype,
                "reason": "No relationships of any type are connected to this node. This can happen if the entity was extracted but no connections were established, or if related relationships were dropped during deduplication/validation.",
                "source": n.get("source", "N/A")
            }
        else:
            # Node has connections, but none are flow-relevant
            if ntype in ("TestCase", "Attribute", "DataObject", "SystemComponent", "NonFunctionalRequirement", "BusinessRule", "Constraint"):
                reason = f"This node is of type '{ntype}', which represents supporting information/metadata rather than a sequential business/process workflow step."
            else:
                reason = "This node has structural or dependency connections in the graph, but no sequential workflow transitions (like LEADS_TO or TRIGGERS) connect it to the active business flow."
            
            isolated_nodes[nid] = {
                "name": n.get("name", nid),
                "type": ntype,
                "reason": reason,
                "source": n.get("source", "N/A")
            }
            
    # 3. Detect branches and merges
    branches = []
    merges = []
    for nid in in_flow_subgraph:
        if len(adj_flow.get(nid, [])) > 1:
            branches.append(nid)
        if len(preds_flow.get(nid, [])) > 1:
            merges.append(nid)
            
    # 4. Tracing per-actor flows
    actor_flows = []
    for entry in flow_analysis["entry_ids"]:
        if entry not in in_flow_subgraph:
            continue
        
        # Traverse reachable subgraph in flow-only graph
        visited = set()
        queue = [entry]
        visited.add(entry)
        seq = []
        flow_branches = []
        flow_merges = []
        
        while queue:
            curr = queue.pop(0)
            seq.append(curr)
            succs = adj_flow.get(curr, [])
            if len(succs) > 1:
                flow_branches.append(curr)
            for s in succs:
                if s not in visited:
                    visited.add(s)
                    if len(preds_flow.get(s, [])) > 1:
                        flow_merges.append(s)
                    queue.append(s)
                    
        # Identify associated actor
        actor_name = None
        # Look for direct actor node starting the flow
        if entry in actor_node_ids:
            actor_name = next((n.get("name") for n in nodes if n["id"] == entry), entry)
        else:
            # Look for Actor node that has a relationship to the entry
            for r in relationships:
                if r.get("to") == entry and r.get("from") in actor_node_ids:
                    actor_name = next((n.get("name") for n in nodes if n["id"] == r["from"]), r["from"])
                    break
        if not actor_name:
            actor_name = "Standard / Main Flow"
            
        actor_flows.append({
            "entry_id": entry,
            "actor_name": actor_name,
            "sequence": seq,
            "branches": flow_branches,
            "merges": flow_merges
        })
        
    # 5. Connected components in flow graph (undirected)
    visited_cc = set()
    cc_count = 0
    # build undirected flow adjacency
    undir_adj = {nid: set() for nid in in_flow_subgraph}
    for u, succs in adj_flow.items():
        if u not in in_flow_subgraph:
            continue
        for v in succs:
            if v in in_flow_subgraph:
                undir_adj[u].add(v)
                undir_adj[v].add(u)
                
    for nid in in_flow_subgraph:
        if nid not in visited_cc:
            cc_count += 1
            # BFS to visit entire component
            queue = [nid]
            visited_cc.add(nid)
            while queue:
                curr = queue.pop(0)
                for neighbor in undir_adj[curr]:
                    if neighbor not in visited_cc:
                        visited_cc.add(neighbor)
                        queue.append(neighbor)
                        
    # 6. Overall health score
    total_nodes = len(nodes)
    isolated_count = len(isolated_nodes)
    if total_nodes > 0:
        isolated_ratio = isolated_count / total_nodes
        if isolated_ratio == 0:
            health = "Excellent"
        elif isolated_ratio < 0.15:
            health = "Good"
        elif isolated_ratio < 0.35:
            health = "Fair"
        else:
            health = "Needs Review"
    else:
        health = "Empty Graph"
        
    # Convert sets to lists in flow_analysis for JSON serialization
    serialized_flow_analysis = {
        "entry_ids": flow_analysis["entry_ids"],
        "entry_is_guessed": flow_analysis["entry_is_guessed"],
        "dfn": flow_analysis["dfn"],
        "sequence_order": flow_analysis["sequence_order"],
        "back_edges": flow_analysis["back_edges"],
        "in_flow_subgraph": list(flow_analysis["in_flow_subgraph"]),
        "dominators": {k: list(v) for k, v in flow_analysis["dominators"].items()},
        "natural_loops": [
            {
                "header": loop["header"],
                "nodes": list(loop["nodes"]),
                "back_edges": loop["back_edges"]
            }
            for loop in flow_analysis["natural_loops"]
        ]
    }
        
    return {
        "connected_components_count": cc_count,
        "isolated_nodes": isolated_nodes,
        "branches_count": len(branches),
        "merges_count": len(merges),
        "loops_count": len(flow_analysis["natural_loops"]),
        "actor_flows": actor_flows,
        "health_score": health,
        "flow_analysis": serialized_flow_analysis
    }