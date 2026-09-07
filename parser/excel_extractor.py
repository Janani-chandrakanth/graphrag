"""
Structured Data Extractor — functional-spec spreadsheets (openpyxl).

Same {"nodes": [...], "relationships": [...]} output shape as
graph/entity_extractor.py, so it plugs into the existing
graph.graph_builder.build_graph() unchanged. No LLM call needed here --
the data is already structured; relationships between these new
Requirement nodes and everything else are left to
graph/cross_reference_linker.py's existing pass.
"""
import re

import openpyxl

_COLUMN_ALIASES = {
    "req_id": ["requirement id", "req id", "reqid", "id"],
    "title": ["title", "requirement title", "name"],
    "description": ["description", "requirement description", "details"],
    "module": ["module", "component", "area"],
}


def _norm_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", value.lower().strip())


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace("_", " ").replace("-", " ")


def _build_column_map(headers: list) -> dict:
    normalized = {i: _normalize_header(h) for i, h in enumerate(headers) if h}
    col_map = {}
    for field, aliases in _COLUMN_ALIASES.items():
        for idx, norm_header in normalized.items():
            if norm_header in aliases:
                col_map[field] = idx
                break
    return col_map


def extract_excel_requirements(file_path: str, source: str = None, sheet_name: str = None) -> dict:
    """
    Returns {"nodes": [...], "relationships": [], "error": Optional[str]}
    """
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active
    except Exception as e:
        return {"nodes": [], "relationships": [], "error": f"Could not open spreadsheet: {e}"}

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {"nodes": [], "relationships": [], "error": "Spreadsheet is empty."}

    headers = [str(h) if h is not None else "" for h in rows[0]]
    col_map = _build_column_map(headers)

    if "description" not in col_map:
        return {
            "nodes": [], "relationships": [],
            "error": (
                f"Could not find a description-like column. Headers found: {headers}. "
                f"Expected one of {_COLUMN_ALIASES['description']}."
            ),
        }

    nodes = []
    for row_num, row in enumerate(rows[1:], start=2):
        description = row[col_map["description"]] if col_map.get("description") is not None else None
        if not description:
            continue

        raw_id = (row[col_map["req_id"]] if "req_id" in col_map and row[col_map["req_id"]] else f"ROW-{row_num}")
        title = (row[col_map["title"]] if "title" in col_map and row[col_map["title"]] else str(description)[:60])
        module = (row[col_map["module"]] if "module" in col_map and row[col_map["module"]] else "Unspecified")

        nodes.append({
            "id": _norm_id(str(raw_id)),
            "type": "Requirement",
            "name": str(title).strip(),
            "description": str(description).strip(),
            "source": source,
            "attributes": {"module": str(module).strip()},
            "aliases": [],
        })

    return {"nodes": nodes, "relationships": [], "error": None}
