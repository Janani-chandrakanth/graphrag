"""
tests/test_feature_extractor.py
Phase 2 unit tests for graph/feature_extractor.py

Run with: python tests/test_feature_extractor.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.feature_extractor import extract_features, get_feature_by_id, get_feature_node_ids
from graph.semantic_classifier import classify_graph

# ── Synthetic test graph ──────────────────────────────────────────────
NODES = [
    # Feature 1: Login
    {"id": "user",            "type": "Actor",       "name": "User"},
    {"id": "login",           "type": "Action",      "name": "Login"},
    {"id": "dashboard",       "type": "State",       "name": "Dashboard"},
    {"id": "email",           "type": "DataObject",  "name": "Email"},
    {"id": "password",        "type": "DataObject",  "name": "Password"},

    # Feature 2: Checkout (Branching)
    {"id": "cart",            "type": "Action",      "name": "Add to Cart"},
    {"id": "checkout",        "type": "Action",      "name": "Checkout"},
    {"id": "payment",         "type": "Action",      "name": "Payment"},
    {"id": "success",         "type": "State",       "name": "Success"},
    {"id": "rejected",        "type": "State",       "name": "Rejected"},
    {"id": "currency",        "type": "Attribute",   "name": "Currency"},
    {"id": "req_002",         "type": "Requirement", "name": "Payment Requirement"},

    # Feature 3: Single Node Island
    {"id": "logout",          "type": "Action",      "name": "Logout"},

    # Artifact (excluded)
    {"id": "TESTCASE::TC-001","type": "TestCase",    "name": "TC-001"},
]

RELATIONSHIPS = [
    # Login Flow
    {"from": "user",    "to": "login",      "type": "PERFORMS"},
    {"from": "login",   "to": "dashboard",  "type": "LEADS_TO"},
    {"from": "login",   "to": "email",      "type": "USES"},
    {"from": "login",   "to": "password",   "type": "USES"},

    # Checkout Flow
    {"from": "cart",    "to": "checkout",   "type": "LEADS_TO"},
    {"from": "checkout","to": "payment",    "type": "TRIGGERS"},
    {"from": "payment", "to": "success",    "type": "LEADS_TO"},
    {"from": "payment", "to": "rejected",   "type": "LEADS_TO"}, # Branch!
    {"from": "payment", "to": "currency",   "type": "TRACKS"},
    {"from": "payment", "to": "req_002",    "type": "REALIZES"},

    # Traceability
    {"from": "TESTCASE::TC-001", "to": "login", "type": "VALIDATES"},
]

def _sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_feature_extraction():
    _sep("TEST: Extract Features")
    
    # We set use_llm=False so the tests run deterministically and fast, relying on heuristics
    result = extract_features(NODES, RELATIONSHIPS, use_llm=False)
    features = result.get("features", [])
    
    # We expect 2 features (Login, Checkout). Logout is an isolated node. If _MIN_STEPS_FOR_FEATURE=1, it might be 3.
    # Let's inspect the output.
    print(f"  Found {len(features)} features.")
    
    feature_names = [f["name"] for f in features]
    print(f"  Feature names: {feature_names}")
    
    # Check Login Feature
    login_f = next((f for f in features if "Login" in f["name"]), None)
    assert login_f, "Login feature missing"
    print(f"  [PASS] Found Login feature")
    assert login_f["entry_point_id"] == "login", "Login entry point is wrong"
    print(f"  [PASS] Login entry point is 'login'")
    assert "dashboard" in login_f["step_node_ids"], "Dashboard missing from Login feature"
    print(f"  [PASS] Dashboard is in Login feature")
    assert "User" in login_f["actors"], "User missing from Login feature actors"
    print(f"  [PASS] User actor assigned to Login feature")
    assert "email" in login_f["supporting_node_ids"], "Email missing from supporting nodes"
    print(f"  [PASS] Email is in supporting context")
    
    # Check Checkout Feature (Branching)
    checkout_f = next((f for f in features if "Add to Cart" in f["name"] or "cart" in f["entry_point_id"]), None)
    assert checkout_f, "Checkout feature missing"
    print(f"  [PASS] Found Checkout feature")
    assert checkout_f["entry_point_id"] == "cart", "Checkout entry point should be cart"
    print(f"  [PASS] Checkout entry point is 'cart'")
    
    # Check branching logic
    assert "payment" in checkout_f["decision_node_ids"], "Payment should be a decision node"
    print(f"  [PASS] Payment identified as a decision node")
    
    branches = checkout_f["branches"]
    assert len(branches) == 1, "Should have 1 branch point"
    assert branches[0]["from"] == "payment"
    assert "success" in branches[0]["to_list"] and "rejected" in branches[0]["to_list"]
    print(f"  [PASS] Branches correctly identified for payment (success, rejected)")
    
    assert "success" in checkout_f["end_state_ids"] and "rejected" in checkout_f["end_state_ids"], "Missing end states"
    print(f"  [PASS] End states correctly identified (success, rejected)")

    # Check Requirements linkage in supporting context
    assert "req_002" in checkout_f["requirement_ids"], "Payment Requirement not found in supporting context"
    print(f"  [PASS] Requirement properly placed in supporting context")
    
    print("\n  [OK] Feature Extraction passed")

if __name__ == "__main__":
    test_feature_extraction()
    print("\n" + "="*60)
    print("  ALL PHASE 2 TESTS PASSED")
    print("="*60)
