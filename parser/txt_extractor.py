"""
TXT Structure Extractor

Plain text has no built-in structure, but requirement docs saved/exported
as .txt very commonly still use conventions we can detect:

  - "# Heading" / "## Heading"          -> heading (markdown-style)
  - "SECTION IN ALL CAPS" (short line)  -> heading (level 2 heuristic)
  - "- item" / "* item" / "1. item"     -> list_item
  - "col1 | col2 | col3" repeated lines -> table
  - everything else, blank-separated    -> paragraph
"""

import re
from parser.structure_preserver import Block, DocumentStructure

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_BULLET_RE = re.compile(r"^\s*[-\*•]\s+(.*)")
_NUMBERED_RE = re.compile(r"^\s*\d{1,3}[\.\)]\s+(.*)")
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-/&:]{2,60}$")


def _is_table_row(line: str) -> bool:
    # >=1 pipe catches 2-column tables too. This is a heuristic — a stray
    # "|" in prose could false-positive as a 1-row table, but a single
    # misclassified row is a far smaller loss than dropping real tables.
    return line.count("|") >= 1


def _split_pipe_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def extract_txt_structure(txt_file, filename: str = "document.txt") -> DocumentStructure:
    raw = txt_file.read()
    if isinstance(raw, bytes):
        # "utf-8-sig" (not plain "utf-8") strips a leading BOM if present
        # and is otherwise identical — plenty of .txt exports (Windows
        # Notepad, Excel "Save As Unicode Text", ...) carry one. Left as
        # plain "utf-8", the BOM survives decoding as a literal U+FEFF
        # character glued onto the first line, which silently fails
        # every start-of-string regex match on that line — "Title:",
        # "FR-001:", anything — since \ufeff isn't whitespace and every
        # ID/field-label pattern here is anchored with ^\s*.
        raw = raw.decode("utf-8-sig", errors="replace")

    lines = raw.split("\n")
    blocks = []
    table_buffer = []

    def flush_table():
        nonlocal table_buffer
        if table_buffer:
            rows = [r for r in table_buffer if any(c for c in r)]
            if len(rows) >= 2:
                blocks.append(Block(type="table", content=rows))
            else:
                # Not enough rows to be confident it's a real table —
                # treat the lone line(s) as plain paragraphs instead.
                for r in rows:
                    blocks.append(Block(type="paragraph", content=" | ".join(r)))
        table_buffer = []

    for line in lines:
        stripped = line.strip()

        if _is_table_row(stripped):
            table_buffer.append(_split_pipe_row(stripped))
            continue
        else:
            flush_table()

        if not stripped:
            continue

        md = _MD_HEADING_RE.match(stripped)
        if md:
            level = len(md.group(1))
            blocks.append(Block(type="heading", content=md.group(2).strip(), level=level))
            continue

        bullet = _BULLET_RE.match(stripped)
        if bullet:
            blocks.append(Block(type="list_item", content=bullet.group(1).strip(),
                                 level=1, meta={"ordered": False}))
            continue

        numbered = _NUMBERED_RE.match(stripped)
        if numbered:
            blocks.append(Block(type="list_item", content=numbered.group(1).strip(),
                                 level=1, meta={"ordered": True}))
            continue

        if len(stripped) <= 60 and _ALLCAPS_HEADING_RE.match(stripped) and len(stripped.split()) <= 8:
            blocks.append(Block(type="heading", content=stripped.title(), level=2))
            continue

        blocks.append(Block(type="paragraph", content=stripped))

    flush_table()

    return DocumentStructure(source_filename=filename, source_type="txt", blocks=blocks)
