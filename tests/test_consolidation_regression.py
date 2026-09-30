import pytest
from graph.graph_analysis import evaluate_structure, evaluate_sequence, trace_failure, format_trace_report
from parser.requirement_linker import link_requirements
from parser.markdown_block_parser import parse_markdown_to_blocks
from parser.normalization import DocumentStructure, Block
from graph.version_manager import diff_graphs
from graph.evaluation.test_case_contracts import preprocess_test_cases


def test_trace_failure_not_found():
    nodes = [{"id": "N1", "name": "Step 1", "type": "Action"}]
    rels = []
    res = trace_failure("TC_DOES_NOT_EXIST_NEG", nodes=nodes, relationships=rels)
    assert res["found"] is False
    assert "No TestCase node found" in res["error_message"]
    report = format_trace_report(res)
    assert "No TestCase node found for 'TC_DOES_NOT_EXIST_NEG'" in report


def test_evaluate_structure():
    nodes = [
        {"id": "req_1", "name": "Req 1", "type": "Requirement", "source": "REQ-001"},
        {"id": "step_1", "name": "Step 1", "type": "Action", "source": "REQ-001"}
    ]
    rels = [{"from": "req_1", "to": "step_1", "type": "CONTAINS"}]
    items_with_links = [{"id": "REQ-001", "family": "REQ"}]

    report = evaluate_structure(nodes, rels, items_with_links)
    assert report["node_count"] == 2
    assert report["relationship_count"] == 1
    assert len(report["isolated_nodes"]) == 0
    assert len(report["requirements_with_no_extracted_content"]) == 0


def test_requirement_linker():
    items = [
        {"id": "FR-001", "family": "FR", "content": "The system shall implement BR-101."},
        {"id": "BR-101", "family": "BR", "content": "Business rule for validation."},
    ]
    result = link_requirements(items)
    assert result["summary"]["total_links"] == 1
    assert result["links"][0]["source_id"] == "FR-001"
    assert result["links"][0]["target_id"] == "BR-101"
    assert result["links"][0]["link_type"] == "REALIZES"


def test_markdown_block_parser_and_to_markdown():
    md = "# Overview\n\nThis is a sample document.\n\n- Bullet 1\n- Bullet 2\n"
    blocks = parse_markdown_to_blocks(md)
    assert len(blocks) == 4
    assert blocks[0].type == "heading"
    assert blocks[0].level == 1

    doc = DocumentStructure(source_filename="test.md", source_type="txt", blocks=blocks)
    exported = doc.to_markdown()
    assert "# Overview" in exported
    assert "- Bullet 1" in exported


def test_diff_graphs():
    ver_a = {
        "nodes": [{"id": "A", "type": "Action", "description": "old"}],
        "relationships": [{"from": "A", "to": "B", "type": "LEADS_TO"}]
    }
    ver_b = {
        "nodes": [
            {"id": "A", "type": "Action", "description": "new"},
            {"id": "C", "type": "Action", "description": "added"}
        ],
        "relationships": [{"from": "A", "to": "C", "type": "LEADS_TO"}]
    }
    diff = diff_graphs(ver_a, ver_b)
    assert diff["summary"]["added_nodes"] == 1
    assert diff["summary"]["modified_nodes"] == 1
    assert diff["summary"]["removed_relationships"] == 1
    assert diff["summary"]["added_relationships"] == 1


def test_preprocess_test_cases_dict_and_mapping():
    class FakeNode:
        def __init__(self, data):
            self._data = data
        def items(self):
            return self._data.items()
        def get(self, k, default=None):
            return self._data.get(k, default)

    fake_node = FakeNode({"id": "TC-101", "title": "Check Login", "steps": ["Step 1"]})
    regular_dict = {"tc_id": "TC-102", "name": "Check Logout"}

    processed = preprocess_test_cases([fake_node, regular_dict, "invalid_str"])
    assert len(processed) == 2
    assert processed[0]["id"] == "TC-101"
    assert processed[1]["id"] == "TC-102"
    assert "_full_text" in processed[0]
    assert "_full_text" in processed[1]
