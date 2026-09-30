"""
Markdown -> Block list parser.

Docling's export_to_markdown() (see parser/pdf_extractor.py) already
produces real, well-formed Markdown -- proper "#"-heading levels, real
pipe tables with a header-separator row, standard "-"/"1." lists --
much better structural signal than the old pdfplumber font-size
heuristics ever had to reconstruct by hand. This module's only job is
turning that Markdown text into this project's format-agnostic
Block/DocumentStructure representation (parser/structure_preserver.py),
the same contract docx/txt extractors already produce, so nothing
downstream (Document Type Detector, Template Normalizer, chunker, ...)
needs to know or care that a PDF was involved at all.
"""

import re
from parser.normalization import Block

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_BULLET_RE = re.compile(r"^\s*[-\*+•●▪]\s+(.*)")
_NUMBERED_RE = re.compile(r"^\s*\d{1,3}[\.\)]\s+(.*)")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-]+\|[\s:\-|]*\|?\s*$")
_IMAGE_MARKER_RE = re.compile(r"^\[Image (\d+) content\]:\s*$")
_IMAGE_MD_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)")


def _split_table_row(line: str):
    cells = line.strip()
    if cells.startswith("|"):
        cells = cells[1:]
    if cells.endswith("|"):
        cells = cells[:-1]
    return [c.strip() for c in cells.split("|")]


def parse_markdown_to_blocks(md_text: str) -> list:
    """
    Args:
        md_text: Markdown text (headings, lists, pipe tables, and
                  optionally "[Image N content]:\\n<ocr text>" markers --
                  see parser/pdf_extractor.py's _merge_ocr_into_markdown).

    Returns:
        List[Block] in document order.
    """
    lines = md_text.replace("\r\n", "\n").split("\n")
    blocks = []
    i = 0
    n = len(lines)
    para_buffer = []

    def flush_paragraph():
        if para_buffer:
            text = " ".join(l.strip() for l in para_buffer if l.strip())
            if text:
                blocks.append(Block(type="paragraph", content=text))
            para_buffer.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        heading = _ATX_HEADING_RE.match(stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            blocks.append(Block(type="heading", content=heading.group(2).strip(), level=level))
            i += 1
            continue

        img_marker = _IMAGE_MARKER_RE.match(stripped)
        if img_marker:
            # OCR'd figure content, inserted inline by the PDF extractor
            # at the position the image originally appeared -- kept as
            # its own "image" block (with the OCR text in meta) rather
            # than folded into a plain paragraph, so downstream code can
            # still tell "this came from a figure" apart from body text.
            flush_paragraph()
            idx = int(img_marker.group(1))
            i += 1
            ocr_lines = []
            while i < n and lines[i].strip():
                ocr_lines.append(lines[i].strip())
                i += 1
            blocks.append(Block(
                type="image",
                content={"alt": f"image_{idx}", "index": idx},
                meta={"ocr_text": "\n".join(ocr_lines)},
            ))
            continue

        img_md = _IMAGE_MD_RE.match(stripped)
        if img_md:
            flush_paragraph()
            blocks.append(Block(type="image", content={"alt": img_md.group(1) or "image", "index": None}))
            i += 1
            continue

        # A pipe-table: this line + the next line is a header-separator
        # row ("| --- | --- |"). Collect rows until a non-pipe/blank line.
        if "|" in stripped and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            flush_paragraph()
            rows = [_split_table_row(stripped)]
            i += 2  # skip header + separator row
            while i < n and lines[i].strip() and "|" in lines[i]:
                rows.append(_split_table_row(lines[i]))
                i += 1
            blocks.append(Block(type="table", content=rows))
            continue

        bullet = _BULLET_RE.match(stripped)
        if bullet:
            flush_paragraph()
            blocks.append(Block(type="list_item", content=bullet.group(1).strip(),
                                 level=1, meta={"ordered": False}))
            i += 1
            continue

        numbered = _NUMBERED_RE.match(stripped)
        if numbered:
            flush_paragraph()
            blocks.append(Block(type="list_item", content=numbered.group(1).strip(),
                                 level=1, meta={"ordered": True}))
            i += 1
            continue

        para_buffer.append(stripped)
        i += 1

    flush_paragraph()
    return blocks