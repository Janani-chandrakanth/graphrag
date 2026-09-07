"""
tests/test_issue1_metadata_filter.py
Unit test for Issue 1: verifying document control tables and repeating footer text
are detected as metadata, tagged appropriately, and excluded from flow graph features.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.entity_extractor import is_document_metadata_text, extract_entities
from graph.semantic_classifier import classify_nodes, _is_artifact
from graph.flow_graph_analysis import _build_flow_adjacency


def test_document_metadata_detection():
    # Sample Document Control / Version History table text
    doc_control_text = """
    | Version | Date       | Author    | Description / Change History |
    | 1.0     | 2024-01-15 | Parichita | Initial Draft Release       |
    | 1.1     | 2024-02-01 | Admin     | Revised requirements section |
    """
    
    is_meta, reason = is_document_metadata_text(doc_control_text)
    assert is_meta is True, "Document control table should be detected as metadata"
    assert reason == "document_control_table"

    # Sample repeating header / footer text
    footer_text = "Forvis Mazars 5"
    is_footer_meta, footer_reason = is_document_metadata_text(footer_text)
    assert is_footer_meta is True, "Footer text 'Forvis Mazars 5' should be detected as metadata"
    assert footer_reason == "page_header_footer"


def test_metadata_nodes_tagged_and_classified():
    nodes = [
        {"id": "author_parichita", "type": "Author", "name": "Parichita"},
        {"id": "footer_artifact", "type": "PageArtifact", "name": "Forvis Mazars 5"},
        {"id": "doc_ctrl", "type": "DocumentMetadata", "name": "Document Control Table"},
        {"id": "user_login", "type": "Action", "name": "User Login"},
    ]
    
    # Verify _is_artifact returns True for metadata node types
    assert _is_artifact("author_parichita", "Author") is True
    assert _is_artifact("footer_artifact", "PageArtifact") is True
    assert _is_artifact("doc_ctrl", "DocumentMetadata") is True
    assert _is_artifact("user_login", "Action") is False

    classified = classify_nodes(nodes, [])
    assert classified["author_parichita"]["role"] == "artifact"
    assert classified["footer_artifact"]["role"] == "artifact"
    assert classified["doc_ctrl"]["role"] == "artifact"
    assert classified["user_login"]["role"] in ("workflow_transition", "workflow_step") or classified["user_login"]["testable"] is True


def test_flow_graph_adjacency_excludes_metadata():
    nodes = [
        {"id": "parichita", "type": "Author", "name": "Parichita"},
        {"id": "footer", "type": "PageArtifact", "name": "Forvis Mazars 5"},
        {"id": "step_1", "type": "Action", "name": "Enter Credentials"},
        {"id": "step_2", "type": "Action", "name": "Submit Login"},
    ]
    rels = [
        {"from": "parichita", "to": "step_1", "type": "TRIGGERS"},
        {"from": "step_1", "to": "footer", "type": "LEADS_TO"},
        {"from": "step_1", "to": "step_2", "type": "LEADS_TO"},
    ]

    adj = _build_flow_adjacency(nodes, rels)
    
    # Metadata nodes must not be keys in adjacency
    assert "parichita" not in adj
    assert "footer" not in adj
    # Real steps must remain connected
    assert adj.get("step_1") == ["step_2"]


if __name__ == "__main__":
    test_document_metadata_detection()
    test_metadata_nodes_tagged_and_classified()
    test_flow_graph_adjacency_excludes_metadata()
    print("ALL ISSUE 1 METADATA FILTERING TESTS PASSED!")
