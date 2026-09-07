"""
graph/story_exports.py — export the generated Gherkin suite as a raw
.feature file or an Excel workbook (one row per test step), for
hand-off to QA tools that don't speak Gherkin natively.

Ported from graphrag2/generation/exports.py.
"""
import io

from openpyxl import Workbook


def to_feature_file_bytes(gherkin_text: str) -> bytes:
    return gherkin_text.encode("utf-8")


def to_excel_bytes(cases: list) -> bytes:
    """
    cases: output of graph.gherkin_test_case_generator.parse_gherkin_to_cases()
    One row per test step, with the scenario title repeated for grouping.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Cases"
    ws.append(["Scenario", "Step #", "Action", "Expected Result"])

    for case in cases:
        for step in case["steps"]:
            ws.append([case["title"], step["stepIndex"], step["action"], step["expectedResult"]])

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
