"""
Document Parser — orchestrator

Dispatches to the format-specific structure extractors (docx/pdf/txt) and
returns a DocumentStructure (Structure Preserver output).

Backward compatibility: `extract_text()` is kept because app.py and the
existing chunker import it directly. It now returns structure-preserving
Markdown (headings/lists/tables intact) instead of flat text — a strict
upgrade, since every downstream regex that used to match plain text still
matches the same text inside Markdown, but headings/lists/tables are no
longer lost.

New code (Document Type Detector, Template Normalizer, etc.) should call
`parse_document()` directly to get the full DocumentStructure rather than
just the flattened Markdown string.
"""

from parser.normalization import DocumentStructure
from parser.extractors import extract_docx_structure
from parser.extractors import extract_pdf_structure
from parser.extractors import extract_txt_structure


def parse_document(uploaded_file) -> DocumentStructure:
    """
    Parse an uploaded file into a structure-preserving DocumentStructure.

    Args:
        uploaded_file: file-like object with a `.name` attribute
                       (Streamlit's UploadedFile, or any object exposing
                       .name and standard read()/seek()).

    Returns:
        DocumentStructure with blocks in original document order.
    """
    filename = getattr(uploaded_file, "name", "document")
    extension = filename.split(".")[-1].lower()

    if extension == "docx":
        return extract_docx_structure(uploaded_file, filename)
    elif extension == "pdf":
        return extract_pdf_structure(uploaded_file, filename)
    elif extension == "txt":
        return extract_txt_structure(uploaded_file, filename)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def extract_text(uploaded_file) -> str:
    """
    Backward-compatible entry point used by app.py / chunker.py.

    Returns structure-preserving Markdown (not flat text). Existing
    regex-based ID detection (FR-001, TC-001, etc.) still works fine on
    Markdown since it's still plain text with the same lines — it just
    also carries heading levels, list markers, and real Markdown tables
    now instead of a flattened " | " join.
    """
    structure = parse_document(uploaded_file)
    return structure.to_markdown()
