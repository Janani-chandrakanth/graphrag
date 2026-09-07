"""
DOCX Structure Extractor

Walks a .docx file in true document order (paragraphs, tables, and inline
images interleaved as they actually appear) and produces a DocumentStructure.

Uses Document.iter_inner_content() (python-docx >= 1.1) which yields
Paragraph and Table objects in document order — this is what lets us
keep tables in the right place relative to surrounding text instead of
dumping all tables at the end.
"""

from docx import Document
from docx.oxml.ns import qn
from parser.structure_preserver import Block, DocumentStructure

# Namespace for inline drawing elements (embedded images)
_DRAWING_TAG = qn("w:drawing")


def _heading_level(paragraph) -> int:
    """Return 1-6 if paragraph is a heading style, else 0."""
    style_name = (paragraph.style.name or "") if paragraph.style else ""
    style_name = style_name.strip().lower()
    if style_name.startswith("heading"):
        digits = "".join(ch for ch in style_name if ch.isdigit())
        return int(digits) if digits else 1
    if style_name in ("title",):
        return 1
    return 0


def _list_info(paragraph):
    """Return (is_list, level, ordered) based on style name / numbering props."""
    style_name = (paragraph.style.name or "") if paragraph.style else ""
    style_lower = style_name.lower()

    pPr = paragraph._p.pPr
    has_numpr = pPr is not None and pPr.numPr is not None

    if "list bullet" in style_lower or "list number" in style_lower \
            or "list paragraph" in style_lower or has_numpr:
        # Determining true bullet-vs-numbered requires resolving numbering.xml;
        # as a heuristic we trust the style name when present.
        ordered = "number" in style_lower
        # rough indent level from style name suffix, e.g. "List Bullet 2"
        digits = "".join(ch for ch in style_lower if ch.isdigit())
        level = int(digits) if digits else 1
        return True, level, ordered

    return False, 0, False


def _has_inline_image(paragraph) -> bool:
    return paragraph._p.find(f".//{_DRAWING_TAG}") is not None


def _extract_table(table) -> list:
    rows = []
    for row in table.rows:
        rows.append([cell.text.strip() for cell in row.cells])
    return rows


def extract_docx_structure(docx_file, filename: str = "document.docx") -> DocumentStructure:
    document = Document(docx_file)
    blocks = []
    image_index = 0

    for item in document.iter_inner_content():
        cls_name = item.__class__.__name__

        if cls_name == "Paragraph":
            paragraph = item

            if _has_inline_image(paragraph):
                image_index += 1
                blocks.append(Block(
                    type="image",
                    content={"alt": f"figure_{image_index}", "index": image_index},
                ))

            text = paragraph.text.strip()
            if not text:
                continue

            level = _heading_level(paragraph)
            if level:
                blocks.append(Block(type="heading", content=text, level=level))
                continue

            is_list, list_level, ordered = _list_info(paragraph)
            if is_list:
                blocks.append(Block(
                    type="list_item", content=text, level=list_level or 1,
                    meta={"ordered": ordered},
                ))
                continue

            blocks.append(Block(type="paragraph", content=text))

        elif cls_name == "Table":
            rows = _extract_table(item)
            if any(any(cell for cell in row) for row in rows):
                blocks.append(Block(type="table", content=rows))

    return DocumentStructure(
        source_filename=filename,
        source_type="docx",
        blocks=blocks,
    )
