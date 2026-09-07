"""
graph/semantic_classifier.py
Phase 1 of the Feature Flow Enhancement.

Hybrid deterministic + LLM approach to classifying source graph nodes and
relationships into semantic roles. This is the foundation for:

  * Feature Flow feature extraction (Phase 2)
  * Correct evaluation coverage universe (Phase 6/7)

Design principles:
  - Deterministic rules run FIRST and cover the majority of clear-cut cases.
  - LLM is called only ONCE, in batch, for genuinely ambiguous nodes/edges.
  - TestCase artifact nodes are always "artifact" — never enter the
    source workflow coverage universe.
  - No per-node, no per-click LLM calls.

Node roles:
  actor           – a person, role, or system acting in the workflow
  workflow_step   – an executable action in a user/business workflow
  workflow_state  – a stable state the system/process can be in
  input           – data/value provided by an actor or system
  business_rule   – a policy or constraint governing the workflow
  requirement     – a formal requirement node (FR, NFR, BR, ...)
  module          – a software module, system component, or screen
  ui_element      – a UI widget, form, button, page
  metadata        – label, description, attribute, identifier
  artifact        – generated artefacts (TestCase nodes, etc.)
  other           – anything that does not fit above

Edge roles:
  workflow_transition – "X then Y" — an actual state/action transition
  dependency          – "X requires/depends on Y"
  business_rule       – "X is governed by rule Y"
  attribute           – "X has attribute/property Y"
  semantic_relation   – meaningful but not directly executable link
  metadata            – structural/document-level link (NEXT_IN_DOCUMENT, etc.)
  other               – anything that does not fit above
"""

import json
import hashlib
import logging
from typing import List, Dict, Any, Tuple

from graph.flow_graph_analysis import FLOW_RELATIONS
from parser.llm_client import call_ollama, extract_json_block
from config import EXTRACTION_MODEL, EXTRACTION_MODEL_URL

logger = logging.getLogger(__name__)

# ── Classifier version — bump when prompt/rules change to bust caches ─
_CLASSIFIER_VERSION = "v1.2"

# ── Deterministic node-type → role mapping ────────────────────────────
# These are high-confidence rules that don't need LLM confirmation.
_TYPE_TO_ROLE: Dict[str, str] = {
    # Actors
    "Actor":                "actor",
    "Role":                 "actor",
    "User":                 "actor",
    "Persona":              "actor",
    # Workflow
    "Action":               "workflow_step",
    "WorkflowStep":         "workflow_step",
    "Step":                 "workflow_step",
    "Process":              "workflow_step",
    "Task":                 "workflow_step",
    "Operation":            "workflow_step",
    "Feature":              "module",          # Feature = top-level module/container, not individual step
    "State":                "workflow_state",
    "WorkflowState":        "workflow_state",
    "Status":               "workflow_state",
    "Outcome":              "workflow_state",
    "Event":                "workflow_state",
    # Inputs
    "Input":                "input",
    "DataObject":           "input",
    "Data":                 "input",
    "Parameter":            "input",
    "Field":                "input",
    "Attribute":            "metadata",
    "Property":             "metadata",
    # Rules
    "BusinessRule":         "business_rule",
    "Rule":                 "business_rule",
    "Policy":               "business_rule",
    "Constraint":           "business_rule",
    "Condition":            "business_rule",
    # Requirements
    "Requirement":          "requirement",
    "FunctionalRequirement":"requirement",
    "NonFunctionalRequirement": "requirement",
    "UserStory":            "requirement",
    # Modules / System
    "SystemComponent":      "module",
    "Module":               "module",
    "System":               "module",
    "Service":              "module",
    "Screen":               "ui_element",
    "Page":                 "ui_element",
    "View":                 "ui_element",
    "Form":                 "ui_element",
    "Button":               "ui_element",
    "Component":            "ui_element",
    # Artefacts & Document Metadata
    "TestCase":             "artifact",
    "DocumentMetadata":     "artifact",
    "Author":               "artifact",
    "PageArtifact":         "artifact",
    "Metadata":             "artifact",
    "VersionHistory":       "artifact",
    # Meta
    "Output":               "workflow_state",  # output of an action = a state
}


# ── Deterministic edge-type → role mapping ────────────────────────────
# FLOW_RELATIONS = actual state/action transitions (from flow_graph_analysis.py)
_FLOW_RELATIONS_SET: frozenset = frozenset(FLOW_RELATIONS)

_REL_TYPE_TO_ROLE: Dict[str, str] = {
    # workflow transitions — covered by FLOW_RELATIONS + explicit extras
    "TRIGGERS":         "workflow_transition",
    "LEADS_TO":         "workflow_transition",
    "CAUSES":           "workflow_transition",
    "CREATES":          "workflow_transition",
    "UPDATES":          "workflow_transition",
    "GENERATES":        "workflow_transition",
    "NOTIFIES":         "workflow_transition",
    "EXECUTES":         "workflow_transition",
    "PRODUCES":         "workflow_transition",
    "PERFORMS":         "workflow_transition",
    "AUTHENTICATES":    "workflow_transition",
    "PROCESSES":        "workflow_transition",
    "VALIDATES":        "workflow_transition",
    "MONITORS":         "workflow_transition",
    "NAVIGATES_TO":     "workflow_transition",
    "SHOWS":            "workflow_transition",
    # dependencies
    "HAS_STEP":         "dependency",
    "HAS_FEATURE":      "dependency",
    "DEPENDS_ON":       "dependency",
    "REQUIRED_FOR":     "dependency",
    "USES":             "dependency",
    "SUPPORTS":         "dependency",
    "CONTAINS":         "dependency",
    "PART_OF":          "dependency",
    "RETRIEVES":        "dependency",
    "STORES":           "dependency",
    # business rules
    "CONSTRAINS":       "business_rule",
    "DEFINES":          "business_rule",
    # attributes
    "TRACKS":           "attribute",
    "OWNS":             "attribute",
    # semantic relations
    "ASSOCIATED_WITH":  "semantic_relation",
    "RELATED_TO":       "semantic_relation",
    "RELATES_TO":       "semantic_relation",
    "REFERENCES":       "semantic_relation",
    "CONTRIBUTES_TO":   "semantic_relation",
    "ACTOR_OF":         "semantic_relation",
    "VERIFIED_BY":      "semantic_relation",
    "VERIFIES":         "semantic_relation",
    "REALIZES":         "semantic_relation",
    "REALIZED_BY":      "semantic_relation",
    "THREATENS":        "semantic_relation",
    "VIEWS":            "semantic_relation",
    # metadata / document structure
    "NEXT_IN_DOCUMENT": "metadata",
}

# ── Artefact node ID prefix (test_case_writer.py convention) ─────────
_ARTIFACT_ID_PREFIXES = ("testcase::", "TESTCASE::", "TC::")

# ── Ambiguous node types that benefit from LLM refinement ────────────
_AMBIGUOUS_TYPES = {
    "Entity", "Item", "Object", "Concept", "Element",
    "Info", "Detail", "Node", "Record", None, "",
}

# ── LLM batch size — don't send more than this many nodes at once ─────
_LLM_BATCH_SIZE = 60


# =============================================================================
# Public API
# =============================================================================

def classify_nodes(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    use_llm: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Classify every node into a semantic role.

    Returns:
        {
            node_id: {
                "role":       str,          # one of the node roles above
                "testable":   bool,         # included in executable test coverage
                "confidence": float,        # 0.0–1.0
                "reason":     str,
                "source":     "deterministic" | "llm" | "fallback",
            },
            ...
        }
    """
    result: Dict[str, Dict[str, Any]] = {}
    needs_llm: List[Dict[str, Any]] = []

    for node in nodes:
        nid   = node.get("id", "")
        ntype = node.get("type", "")
        nname = node.get("name", "")

        # 1. Artefact detection — always wins regardless of type
        if _is_artifact(nid, ntype):
            result[nid] = {
                "role": "artifact", "testable": False,
                "confidence": 1.0, "reason": "Generated test-case artefact node",
                "source": "deterministic",
            }
            continue

        # 2. Deterministic type lookup
        role = _TYPE_TO_ROLE.get(ntype)
        if role:
            result[nid] = {
                "role": role,
                "testable": _is_testable(role),
                "confidence": 0.95,
                "reason": f"Node type '{ntype}' maps deterministically to '{role}'",
                "source": "deterministic",
            }
            continue

        # 3. Name-based heuristics for common patterns
        name_role = _classify_by_name(nname)
        if name_role:
            result[nid] = {
                "role": name_role,
                "testable": _is_testable(name_role),
                "confidence": 0.80,
                "reason": f"Node name '{nname}' pattern matched '{name_role}'",
                "source": "deterministic",
            }
            continue

        # 4. Ambiguous — queue for LLM batch
        needs_llm.append(node)

    # 5. Batch LLM for ambiguous nodes
    if needs_llm and use_llm:
        llm_results = _classify_nodes_with_llm(needs_llm, relationships)
        result.update(llm_results)
    elif needs_llm:
        # LLM disabled: fallback to "other"
        for node in needs_llm:
            nid = node.get("id", "")
            result[nid] = {
                "role": "other", "testable": False,
                "confidence": 0.3,
                "reason": "LLM disabled; type could not be determined deterministically",
                "source": "fallback",
            }

    return result


def classify_edges(
    relationships: List[Dict[str, Any]],
    node_classifications: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Classify every relationship into a semantic edge role.

    Key: (from_id, to_id, rel_type)

    Returns:
        {
            (from_id, to_id, rel_type): {
                "role":       str,
                "testable":   bool,
                "confidence": float,
                "reason":     str,
                "source":     "deterministic" | "llm" | "fallback",
            },
            ...
        }
    """
    result: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for rel in relationships:
        frm   = rel.get("from", "")
        to    = rel.get("to", "")
        rtype = rel.get("type", "")
        key   = (frm, to, rtype)

        # 1. Any edge touching an artefact node → metadata
        from_cls = node_classifications.get(frm, {})
        to_cls   = node_classifications.get(to, {})
        if from_cls.get("role") == "artifact" or to_cls.get("role") == "artifact":
            result[key] = {
                "role": "metadata", "testable": False,
                "confidence": 1.0,
                "reason": "Edge connects to a test-case artefact node",
                "source": "deterministic",
            }
            continue

        # 2. Deterministic rel-type lookup
        role = _REL_TYPE_TO_ROLE.get(rtype)
        if role:
            # Refine: if one endpoint is metadata/attribute, downgrade transition
            if role == "workflow_transition":
                frm_role = from_cls.get("role", "other")
                to_role  = to_cls.get("role", "other")
                if frm_role in {"metadata", "artifact"} or to_role in {"metadata", "artifact"}:
                    role = "attribute"
            result[key] = {
                "role": role,
                "testable": (role == "workflow_transition"),
                "confidence": 0.95,
                "reason": f"Relationship type '{rtype}' maps to '{role}'",
                "source": "deterministic",
            }
            continue

        # 3. Fallback by FLOW_RELATIONS membership
        if rtype in _FLOW_RELATIONS_SET:
            result[key] = {
                "role": "workflow_transition", "testable": True,
                "confidence": 0.85,
                "reason": f"'{rtype}' is in FLOW_RELATIONS",
                "source": "deterministic",
            }
            continue

        # 4. Unknown rel type — use node roles to infer
        inferred = _infer_edge_role_from_node_roles(from_cls, to_cls)
        result[key] = {
            "role": inferred,
            "testable": (inferred == "workflow_transition"),
            "confidence": 0.50,
            "reason": f"Unknown rel type '{rtype}'; inferred from node roles",
            "source": "fallback",
        }

    return result


def build_classification_cache_key(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> str:
    """
    Stable cache key for the classification result.
    Includes node IDs, types, relationship types + CLASSIFIER_VERSION.
    Changing a node's type or adding/removing a relationship invalidates the cache.
    """
    node_sig = sorted((n.get("id", ""), n.get("type", "")) for n in nodes)
    rel_sig  = sorted((r.get("from", ""), r.get("to", ""), r.get("type", "")) for r in relationships)
    raw = json.dumps({"nodes": node_sig, "rels": rel_sig, "version": _CLASSIFIER_VERSION})
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_testable_nodes(
    node_classifications: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return only the nodes classified as testable (workflow_step / workflow_state)."""
    return {nid: cls for nid, cls in node_classifications.items() if cls.get("testable")}


def get_workflow_transition_edges(
    edge_classifications: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Return only the edges classified as workflow_transition."""
    return {
        key: cls for key, cls in edge_classifications.items()
        if cls.get("role") == "workflow_transition"
    }


def get_source_coverage_universe(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    node_classifications: Dict[str, Dict[str, Any]],
    edge_classifications: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the source evaluation universe — the correct denominator for
    coverage calculations.  Artefact and metadata nodes/edges are excluded.

    Returns:
        {
            "workflow_nodes":       [...],  # nodes with role workflow_step/state
            "workflow_transitions": [...],  # edges with role workflow_transition
            "requirement_nodes":    [...],
            "business_rule_nodes":  [...],
            "input_nodes":          [...],
            "all_source_nodes":     [...],  # all non-artefact nodes
            "all_source_edges":     [...],  # all non-metadata edges
        }
    """
    node_by_id = {n["id"]: n for n in nodes}

    workflow_nodes:   List[Dict] = []
    requirement_nodes:List[Dict] = []
    br_nodes:         List[Dict] = []
    input_nodes:      List[Dict] = []
    all_source_nodes: List[Dict] = []

    for nid, cls in node_classifications.items():
        role = cls.get("role", "other")
        if role == "artifact":
            continue
        n = node_by_id.get(nid)
        if not n:
            continue
        all_source_nodes.append(n)
        if role in ("workflow_step", "workflow_state"):
            workflow_nodes.append(n)
        elif role == "requirement":
            requirement_nodes.append(n)
        elif role == "business_rule":
            br_nodes.append(n)
        elif role == "input":
            input_nodes.append(n)

    rel_lookup = {
        (r.get("from", ""), r.get("to", ""), r.get("type", "")): r
        for r in relationships
    }
    workflow_transitions: List[Dict] = []
    all_source_edges:     List[Dict] = []
    for key, cls in edge_classifications.items():
        if cls.get("role") == "metadata" or cls.get("role") == "artifact":
            continue
        rel = rel_lookup.get(key)
        if not rel:
            continue
        all_source_edges.append(rel)
        if cls.get("role") == "workflow_transition":
            workflow_transitions.append(rel)

    return {
        "workflow_nodes":       workflow_nodes,
        "workflow_transitions": workflow_transitions,
        "requirement_nodes":    requirement_nodes,
        "business_rule_nodes":  br_nodes,
        "input_nodes":          input_nodes,
        "all_source_nodes":     all_source_nodes,
        "all_source_edges":     all_source_edges,
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _is_artifact(node_id: str, node_type: str) -> bool:
    """True if a node is a generated test-case, document metadata, or other non-functional artifact."""
    nid = (node_id or "").lower()
    ntype = (node_type or "").lower()
    if ntype in ("testcase", "artifact", "test_case", "documentmetadata", "author", "pageartifact", "metadata", "versionhistory"):
        return True
    for prefix in _ARTIFACT_ID_PREFIXES:
        if nid.startswith(prefix.lower()):
            return True
    return False



def _is_testable(role: str) -> bool:
    return role in ("workflow_step", "workflow_state")


def _classify_by_name(name: str) -> str | None:
    """Heuristic name-pattern matching for common node names."""
    if not name:
        return None
    name_lower = name.lower()
    # Common UI / screen keywords
    if any(kw in name_lower for kw in ("screen", "page", "view", "panel", "dashboard", "portal")):
        return "ui_element"
    # Input/data keywords
    if any(kw in name_lower for kw in ("email", "password", "otp", "pin", "input", "field", "form", "entry", "amount", "currency")):
        return "input"
    # Action/step keywords
    if any(kw in name_lower for kw in ("login", "logout", "register", "signup", "checkout", "payment", "submit", "click", "upload", "download", "search", "select", "add", "remove", "create", "delete", "update", "verify", "authenticate", "authorize", "navigate")):
        return "workflow_step"
    # State keywords
    if any(kw in name_lower for kw in ("success", "failure", "error", "rejected", "approved", "pending", "completed", "cancelled", "confirmed", "delivered", "shipped")):
        return "workflow_state"
    # Rule/constraint keywords
    if any(kw in name_lower for kw in ("rule", "policy", "constraint", "restriction", "limit", "condition", "requirement", "validation")):
        return "business_rule"
    return None


def _infer_edge_role_from_node_roles(
    from_cls: Dict[str, Any],
    to_cls:   Dict[str, Any],
) -> str:
    """When the rel type is unknown, use the endpoint roles to guess edge role."""
    frm_role = from_cls.get("role", "other")
    to_role  = to_cls.get("role", "other")
    # workflow → workflow = transition
    if frm_role in ("workflow_step", "workflow_state", "actor") and \
       to_role  in ("workflow_step", "workflow_state"):
        return "workflow_transition"
    # anything → input/attribute = attribute
    if to_role in ("input", "metadata"):
        return "attribute"
    # anything → business_rule = business_rule
    if to_role == "business_rule":
        return "business_rule"
    # workflow → requirement = semantic
    if to_role == "requirement":
        return "semantic_relation"
    return "other"


def _classify_nodes_with_llm(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Single batched LLM call to classify ambiguous nodes.
    Returns the same shape as classify_nodes().
    Falls back gracefully if Ollama is unavailable.
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Process in batches to avoid huge prompts
    for batch_start in range(0, len(nodes), _LLM_BATCH_SIZE):
        batch = nodes[batch_start: batch_start + _LLM_BATCH_SIZE]
        node_catalog = [
            {
                "id":   n.get("id", ""),
                "type": n.get("type", ""),
                "name": n.get("name", ""),
                "desc": (n.get("description") or "")[:120],
            }
            for n in batch
        ]
        catalog_json = json.dumps(node_catalog, indent=2)

        prompt = f"""You are a semantic classifier for a Requirements Knowledge Graph.

Classify each node below into exactly ONE of these roles:

actor           - a person, role, or organisational unit acting in the workflow
workflow_step   - an executable action step in a user or business workflow
workflow_state  - a stable system or process state (e.g. Success, Pending, Rejected)
input           - data or value provided by an actor or system (e.g. email, password)
business_rule   - a policy, constraint, or rule governing the workflow
requirement     - a formal requirement (functional, non-functional, user story)
module          - a software module, system, service, or integration
ui_element      - a screen, page, form, widget, or UI component
metadata        - a label, description, identifier, or structural attribute
artifact        - a generated artefact (test case, report, output document)
other           - anything that does not fit above

Nodes to classify:
{catalog_json}

Return ONLY a valid JSON array. Each element must have:
  "id"         : the node id (exactly as given)
  "role"       : one of the roles above
  "confidence" : float 0.0–1.0
  "reason"     : one sentence explaining the classification

Example:
[
  {{"id": "login_action", "role": "workflow_step", "confidence": 0.96, "reason": "Represents an executable login action in the user workflow."}},
  {{"id": "email_field", "role": "input", "confidence": 0.93, "reason": "An input data field for user email address."}}
]
"""
        llm_result = call_ollama(prompt, timeout=90, num_ctx=4096,
                                  model=EXTRACTION_MODEL, base_url=EXTRACTION_MODEL_URL)

        if llm_result.get("error"):
            logger.warning("LLM classifier call failed: %s", llm_result["error"])
            for n in batch:
                nid = n.get("id", "")
                result[nid] = {
                    "role": "other", "testable": False,
                    "confidence": 0.3,
                    "reason": f"LLM unavailable: {llm_result['error']}",
                    "source": "fallback",
                }
            continue

        try:
            classifications = extract_json_block(llm_result["raw"])
            if not isinstance(classifications, list):
                raise ValueError("LLM returned non-list")

            classified_ids = set()
            for item in classifications:
                nid = item.get("id", "")
                role = item.get("role", "other")
                if role not in _TYPE_TO_ROLE.values() and role not in (
                    "actor", "workflow_step", "workflow_state", "input",
                    "business_rule", "requirement", "module", "ui_element",
                    "metadata", "artifact", "other"
                ):
                    role = "other"
                classified_ids.add(nid)
                result[nid] = {
                    "role": role,
                    "testable": _is_testable(role),
                    "confidence": float(item.get("confidence", 0.7)),
                    "reason": item.get("reason", "LLM classified"),
                    "source": "llm",
                }

            # Nodes LLM didn't include in its response → fallback
            for n in batch:
                nid = n.get("id", "")
                if nid not in classified_ids:
                    result[nid] = {
                        "role": "other", "testable": False,
                        "confidence": 0.3,
                        "reason": "LLM did not classify this node",
                        "source": "fallback",
                    }

        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("LLM classifier returned invalid JSON: %s", e)
            for n in batch:
                nid = n.get("id", "")
                result[nid] = {
                    "role": "other", "testable": False,
                    "confidence": 0.3,
                    "reason": "LLM returned malformed JSON; using fallback",
                    "source": "fallback",
                }

    return result


# =============================================================================
# Convenience: classify and build coverage universe in one call
# =============================================================================

def classify_graph(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Full classification pipeline in one call.

    Returns:
        {
            "node_classifications":  {node_id: {...}},
            "edge_classifications":  {(from,to,type): {...}},
            "coverage_universe":     {...},   # see get_source_coverage_universe()
            "cache_key":             str,
        }
    """
    cache_key = build_classification_cache_key(nodes, relationships)
    node_cls  = classify_nodes(nodes, relationships, use_llm=use_llm)
    edge_cls  = classify_edges(relationships, node_cls)
    universe  = get_source_coverage_universe(nodes, relationships, node_cls, edge_cls)

    return {
        "node_classifications": node_cls,
        "edge_classifications": edge_cls,
        "coverage_universe":    universe,
        "cache_key":            cache_key,
    }
