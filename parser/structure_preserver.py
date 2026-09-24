"""
parser/structure_preserver.py

Defines format-agnostic structured document representations (headings, paragraphs, lists, tables, images)
and exports them to markdown.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class Block:
    type: str                      # "heading" | "paragraph" | "list_item" | "table" | "image"
    content: Union[str, List[List[str]], Dict[str, Any]]
    level: Optional[int] = None    # heading level (1-6) or list indent depth
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content,
            "level": self.level,
            "meta": self.meta,
        }


@dataclass
class DocumentStructure:
    source_filename: str
    source_type: str               # "pdf" | "docx" | "txt"
    blocks: List[Block] = field(default_factory=list)

    # ── Serialization ──────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "source_filename": self.source_filename,
            "source_type": self.source_type,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    # ── Convenience accessors (used by Document Type Detector etc.) ──
    def headings(self) -> List[Block]:
        return [b for b in self.blocks if b.type == "heading"]

    def tables(self) -> List[Block]:
        return [b for b in self.blocks if b.type == "table"]

    def images(self) -> List[Block]:
        return [b for b in self.blocks if b.type == "image"]

    def full_text(self) -> str:
        """Plain concatenated text with no structural markers at all.
        Kept only for callers that truly just want a text blob (e.g. an
        embedding call on the whole document). Prefer to_markdown() for
        anything that needs to reason about structure."""
        parts = []
        for b in self.blocks:
            if b.type == "table":
                for row in b.content:
                    parts.append(" | ".join(row))
            elif b.type == "image":
                continue
            else:
                parts.append(b.content)
        return "\n".join(parts)

    # ── Markdown rendering ─────────────────────────────────
    def to_markdown(self) -> str:
        """
        Render blocks back to structure-preserving Markdown.
        This is what Document Type Detector / Template Normalizer /
        Structural Chunker should consume — it keeps headings, list
        nesting, and tables intact so regex/LLM steps downstream can
        still find "FR-001" whether it's in a table cell or a heading.
        """
        lines = []
        ordered_counters = {}  # indent level -> running count, reset on any break in that level's run
        for b in self.blocks:
            if b.type == "heading":
                level = b.level or 1
                lines.append(f"{'#' * min(level, 6)} {b.content}".rstrip())
                lines.append("")
                ordered_counters = {}

            elif b.type == "list_item":
                item_level = b.level or 1
                indent = "  " * max(item_level - 1, 0)
                if b.meta.get("ordered"):
                    ordered_counters[item_level] = ordered_counters.get(item_level, 0) + 1
                    bullet = f"{ordered_counters[item_level]}."
                else:
                    bullet = "-"
                    ordered_counters.pop(item_level, None)
                lines.append(f"{indent}{bullet} {b.content}")

            elif b.type == "paragraph":
                if b.content.strip():
                    lines.append(b.content)
                    lines.append("")
                ordered_counters = {}

            elif b.type == "table":
                lines.extend(_table_to_markdown(b.content))
                lines.append("")
                ordered_counters = {}

            elif b.type == "image":
                alt = b.content.get("alt") or f"image_{b.content.get('index', '')}"
                lines.append(f"![{alt}](embedded-image)")
                lines.append("")

        # Collapse 3+ blank lines down to 1 for readability
        out = []
        blank_run = 0
        for ln in lines:
            if ln == "":
                blank_run += 1
                if blank_run > 1:
                    continue
            else:
                blank_run = 0
            out.append(ln)
        return "\n".join(out).strip() + "\n"


def _table_to_markdown(rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    # Normalize row lengths
    width = max(len(r) for r in rows)
    norm_rows = [r + [""] * (width - len(r)) for r in rows]

    def esc(cell: str) -> str:
        return (cell or "").replace("\n", " ").replace("|", "\\|").strip()

    header, *body = norm_rows
    lines = ["| " + " | ".join(esc(c) for c in header) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in body:
        lines.append("| " + " | ".join(esc(c) for c in row) + " |")
    return lines
