"""
tests/test_semantic_classifier.py
Phase 1 unit tests for graph/semantic_classifier.py

Run with:  python tests/test_semantic_classifier.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.semantic_classifier import (
    classify_nodes,
    classify_edges,
    get_source_coverage_universe,
    classify_graph,
)

# ── Synthetic test graph ──────────────────────────────────────────────
NODES = [
    {"id": "user",            "type": "Actor",       "name": "User"},
    {"id": "login",           "type": "Action",      "name": "Login"},
    {"id": "dashboard",       "type": "State",       "name": "Dashboard"},
    {"id": "search",          "type": "Action",      "name": "Search"},
    {"id": "product",         "type": "Feature",     "name": "Product"},
    {"id": "cart",            "type": "Action",      "name": "Add to Cart"},
    {"id": "payment",         "type": "Action",      "name": "Payment"},
    {"id": "success",         "type": "State",       "name": "Success"},
    {"id": "email",           "type": "DataObject",  "name": "Email"},
    {"id": "password",        "type": "DataObject",  "name": "Password"},
    {"id": "auth_rule",       "type": "BusinessRule","name": "Authentication Rule"},
    {"id": "req_001",         "type": "Requirement", "name": "Login Requirement"},
    {"id": "login_screen",    "type": "Screen",      "name": "Login Screen"},
    {"id": "currency",        "type": "Attribute",   "name": "Currency"},
    # TestCase artefact — must NEVER enter coverage universe
    {"id": "TESTCASE::TC-001","type": "TestCase",    "name": "TC-001"},
    {"id": "testcase::tc-002","type": "TestCase",    "name": "TC-002"},
]

RELATIONSHIPS = [
    {"from": "user",    "to": "login",      "type": "PERFORMS"},
    {"from": "login",   "to": "dashboard",  "type": "LEADS_TO"},
    {"from": "dashboard","to": "search",    "type": "TRIGGERS"},
    {"from": "search",  "to": "product",   "type": "LEADS_TO"},
    {"from": "product", "to": "cart",      "type": "LEADS_TO"},
    {"from": "cart",    "to": "payment",   "type": "TRIGGERS"},
    {"from": "payment", "to": "success",   "type": "LEADS_TO"},
    # Supporting / attribute edges
    {"from": "login",   "to": "email",     "type": "USES"},
    {"from": "login",   "to": "password",  "type": "USES"},
    {"from": "login",   "to": "auth_rule", "type": "CONSTRAINS"},
    {"from": "payment", "to": "currency",  "type": "TRACKS"},
    # TestCase traceability edges — must be excluded from transitions
    {"from": "TESTCASE::TC-001", "to": "login", "type": "VALIDATES"},
]


def _sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_deterministic_node_roles():
    _sep("TEST: Deterministic node classification")
    nc = classify_nodes(NODES, RELATIONSHIPS, use_llm=False)

    checks = {
        "user":              "actor",
        "login":             "workflow_step",
        "dashboard":         "workflow_state",
        "success":           "workflow_state",
        "email":             "input",
        "password":          "input",
        "currency":          "metadata",
        "auth_rule":         "business_rule",
        "req_001":           "requirement",
        "login_screen":      "ui_element",
        "TESTCASE::TC-001":  "artifact",
        "testcase::tc-002":  "artifact",
    }

    passed = True
    for nid, expected_role in checks.items():
        actual = nc.get(nid, {}).get("role", "MISSING")
        status = "PASS" if actual == expected_role else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  [{status}] {nid:30s} expected={expected_role:18s} got={actual}")

    assert passed, "One or more node classification checks failed"
    print("\n  [OK] All deterministic node role checks passed")


def test_artifact_nodes_never_testable():
    _sep("TEST: Artefact nodes are never testable")
    nc = classify_nodes(NODES, RELATIONSHIPS, use_llm=False)
    for nid in ("TESTCASE::TC-001", "testcase::tc-002"):
        cls = nc.get(nid, {})
        testable = cls.get("testable", True)
        assert not testable, f"{nid} should not be testable"
        print(f"  [PASS] {nid} testable=False")
    print("\n  ✓ Artefact testability check passed")


def test_edge_classification():
    _sep("TEST: Edge classification")
    nc = classify_nodes(NODES, RELATIONSHIPS, use_llm=False)
    ec = classify_edges(RELATIONSHIPS, nc)

    # Workflow transitions
    transitions_expected = [
        ("login",   "dashboard", "LEADS_TO"),
        ("cart",    "payment",   "TRIGGERS"),
        ("payment", "success",   "LEADS_TO"),
    ]
    for (frm, to, rtype) in transitions_expected:
        role = ec.get((frm, to, rtype), {}).get("role", "MISSING")
        status = "PASS" if role == "workflow_transition" else "FAIL"
        print(f"  [{status}] {frm}→{to} ({rtype}) expected=workflow_transition got={role}")
        assert role == "workflow_transition", f"Edge ({frm},{to},{rtype}) should be workflow_transition"

    # Attribute edge
    key = ("payment", "currency", "TRACKS")
    role = ec.get(key, {}).get("role", "MISSING")
    assert role == "attribute", f"TRACKS should be attribute, got {role}"
    print(f"  [PASS] payment→currency (TRACKS) = attribute")

    # TestCase traceability edge must be metadata (touches artefact)
    tc_key = ("TESTCASE::TC-001", "login", "VALIDATES")
    role = ec.get(tc_key, {}).get("role", "MISSING")
    assert role == "metadata", f"TC traceability edge should be metadata, got {role}"
    print(f"  [PASS] TESTCASE::TC-001→login (VALIDATES) = metadata (artefact edge)")

    print("\n  ✓ All edge classification checks passed")


def test_coverage_universe_excludes_artifacts():
    _sep("TEST: Coverage universe excludes artefact nodes")
    nc = classify_nodes(NODES, RELATIONSHIPS, use_llm=False)
    ec = classify_edges(RELATIONSHIPS, nc)
    universe = get_source_coverage_universe(NODES, RELATIONSHIPS, nc, ec)

    # No artefact should appear in any universe list
    all_universe_ids = {n["id"] for n in universe["all_source_nodes"]}
    tc_ids = {"TESTCASE::TC-001", "testcase::tc-002"}
    overlap = tc_ids & all_universe_ids
    assert not overlap, f"Artefact nodes found in coverage universe: {overlap}"
    print(f"  [PASS] No artefact nodes in all_source_nodes")

    wf_ids = {n["id"] for n in universe["workflow_nodes"]}
    overlap = tc_ids & wf_ids
    assert not overlap, f"Artefact nodes in workflow_nodes: {overlap}"
    print(f"  [PASS] No artefact nodes in workflow_nodes")

    # Workflow transitions must not include TC traceability edge
    tc_edges = [
        r for r in universe["workflow_transitions"]
        if r.get("from", "").lower().startswith("testcase")
    ]
    assert not tc_edges, f"TC traceability edges in workflow_transitions: {tc_edges}"
    print(f"  [PASS] No TC traceability edges in workflow_transitions")

    wf_count = len(universe["workflow_nodes"])
    tr_count = len(universe["workflow_transitions"])
    print(f"\n  Workflow nodes in universe:       {wf_count}")
    print(f"  Workflow transitions in universe: {tr_count}")
    print(f"  All source nodes:                 {len(universe['all_source_nodes'])}")
    print(f"  Requirement nodes:                {len(universe['requirement_nodes'])}")
    print(f"  Business rule nodes:              {len(universe['business_rule_nodes'])}")
    print(f"  Input nodes:                      {len(universe['input_nodes'])}")
    print("\n  ✓ Coverage universe exclusion check passed")


def test_workflow_step_sequence():
    _sep("TEST: Workflow step and transition detection")
    nc = classify_nodes(NODES, RELATIONSHIPS, use_llm=False)
    ec = classify_edges(RELATIONSHIPS, nc)
    universe = get_source_coverage_universe(NODES, RELATIONSHIPS, nc, ec)

    wf_ids = {n["id"] for n in universe["workflow_nodes"]}
    expected_wf = {"login", "dashboard", "search", "product", "cart", "payment", "success"}
    # user is actor, not workflow; email/password are inputs, not steps
    assert expected_wf.issubset(wf_ids), f"Missing workflow nodes: {expected_wf - wf_ids}"
    assert "user"     not in wf_ids, "Actor 'user' should not be a workflow node"
    assert "email"    not in wf_ids, "Input 'email' should not be a workflow node"
    assert "currency" not in wf_ids, "Metadata 'currency' should not be a workflow node"
    print(f"  [PASS] Expected workflow nodes present: {sorted(expected_wf)}")
    print(f"  [PASS] Actor/input/metadata excluded from workflow nodes")

    tr_tuples = {(r.get("from"), r.get("to")) for r in universe["workflow_transitions"]}
    expected_transitions = {
        ("login",    "dashboard"),
        ("dashboard","search"),
        ("cart",     "payment"),
        ("payment",  "success"),
    }
    missing = expected_transitions - tr_tuples
    assert not missing, f"Missing workflow transitions: {missing}"
    print(f"  [PASS] All expected workflow transitions present")
    print("\n  ✓ Workflow step/transition detection passed")


def test_cache_key_stability():
    _sep("TEST: Cache key stability")
    k1 = classify_graph.__wrapped__(NODES, RELATIONSHIPS) if hasattr(classify_graph, "__wrapped__") else None
    from graph.semantic_classifier import build_classification_cache_key
    k1 = build_classification_cache_key(NODES, RELATIONSHIPS)
    k2 = build_classification_cache_key(NODES, RELATIONSHIPS)
    assert k1 == k2, "Cache key should be stable across calls"
    print(f"  [PASS] Cache key is stable: {k1}")

    # Adding a node changes the key
    nodes2 = NODES + [{"id": "extra_node", "type": "Action", "name": "Extra"}]
    k3 = build_classification_cache_key(nodes2, RELATIONSHIPS)
    assert k1 != k3, "Cache key should change when nodes change"
    print(f"  [PASS] Cache key changes when graph changes: {k3}")
    print("\n  ✓ Cache key stability check passed")


if __name__ == "__main__":
    test_deterministic_node_roles()
    test_artifact_nodes_never_testable()
    test_edge_classification()
    test_coverage_universe_excludes_artifacts()
    test_workflow_step_sequence()
    test_cache_key_stability()

    print("\n" + "="*60)
    print("  ALL PHASE 1 TESTS PASSED")
    print("="*60)
