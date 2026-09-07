"""
PDF Structure Extractor -- Docling + RapidOCR backend

Replaces the old pdfplumber font-size-heuristic reconstruction with
Docling's own document model (real heading levels, real table
structure via TableFormer, real picture regions) plus RapidOCR for
text embedded *inside* images/figures (diagrams, screenshots) that
Docling's own layout model finds but can't read text out of by
itself. OCR'd figure text is spliced back into the Markdown at the
exact position the image appeared, then the whole thing is parsed
into this project's Block/DocumentStructure representation.

Falls back to plain pypdf text extraction (no structure, no OCR) if
docling/rapidocr aren't installed, so the app doesn't hard-fail on a
missing dependency -- same fallback philosophy the pdfplumber version
had.
"""

import os
import re
import shutil
import tempfile

from parser.structure_preserver import Block, DocumentStructure
from parser.markdown_block_parser import parse_markdown_to_blocks

_BULLET_RE = re.compile(r"^\s*([•●▪◦‣∙\-\*])\s+")
_NUMBERED_RE = re.compile(r"^\s*(\d{1,3}[\.\)]|[a-zA-Z][\.\)])\s+")

# Lazily constructed -- importing docling/rapidocr is expensive
# (loads onnxruntime models), so it only happens the first time a PDF
# actually needs parsing, not at module import time / app startup.
_ocr_engine = None
_converter = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, TableStructureOptions, TableFormerMode,
        )
        from docling.datamodel.base_models import InputFormat

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        opts.images_scale = 2.0
        opts.generate_picture_images = True
        opts.table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _converter


def _ocr_image(image_path: str) -> str:
    """Extract visible text from an image using RapidOCR."""
    try:
        engine = _get_ocr_engine()
        result, _ = engine(image_path)
        if not result:
            return ""
        return "\n".join(item[1] for item in result)
    except Exception as e:
        print(f"OCR error on {image_path}: {e}")
        return ""


def _extract_and_ocr_images(doc, temp_image_dir: str) -> list:
    """Save each picture Docling found to a temp dir and OCR it, in order."""
    ocr_results = []
    for i, picture in enumerate(getattr(doc, "pictures", [])):
        ocr_text = ""
        try:
            img = picture.get_image(doc)
            if img:
                img_path = os.path.join(temp_image_dir, f"image_{i}.png")
                img.save(img_path)
                ocr_text = _ocr_image(img_path)
        except Exception as e:
            print(f"Image {i} processing error: {e}")
        ocr_results.append(ocr_text)
    return ocr_results


def _merge_ocr_into_markdown(md_text: str, ocr_results: list) -> str:
    """Replace each <!-- image --> placeholder with OCR'd text, in order,
    so figure text lands at the position the image appeared instead of
    being dropped."""

    def replace_placeholder(match, counter=[0]):
        idx = counter[0]
        counter[0] += 1
        if idx < len(ocr_results):
            text = ocr_results[idx].strip()
            if text:
                return f"\n[Image {idx} content]:\n{text}\n"
            return ""  # nothing OCR'd -- drop the placeholder
        return match.group(0)

    return re.sub(r"<!--\s*image\s*-->", replace_placeholder, md_text)


def _extract_with_docling(pdf_file, filename: str) -> DocumentStructure:
    temp_dir = tempfile.mkdtemp(prefix="doc_parse_")
    temp_image_dir = os.path.join(temp_dir, "images")
    os.makedirs(temp_image_dir, exist_ok=True)
    safe_name = os.path.basename(filename) or "document.pdf"
    temp_pdf_path = os.path.join(temp_dir, safe_name)

    try:
        data = pdf_file.read()
        with open(temp_pdf_path, "wb") as f:
            f.write(data)

        converter = _get_converter()
        result = converter.convert(temp_pdf_path)
        doc = result.document

        md_text = doc.export_to_markdown()
        ocr_results = _extract_and_ocr_images(doc, temp_image_dir)
        final_text = _merge_ocr_into_markdown(md_text, ocr_results)

        blocks = parse_markdown_to_blocks(final_text)
        return DocumentStructure(source_filename=filename, source_type="pdf", blocks=blocks)
    finally:
        # Only our own temp copy is deleted -- the caller's uploaded
        # file object (pdf_file) is left alone, unlike the standalone
        # DocumentParser this was adapted from, which owned and deleted
        # its own input file.
        shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_with_pypdf_fallback(pdf_file, filename: str) -> DocumentStructure:
    """No structure, no OCR -- every page becomes plain paragraph/list
    blocks. Only used when docling/rapidocr aren't installed."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_file)
    blocks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            if line.strip():
                if _BULLET_RE.match(line) or _NUMBERED_RE.match(line):
                    blocks.append(Block(type="list_item", content=line.strip(), level=1))
                else:
                    blocks.append(Block(type="paragraph", content=line.strip()))
    return DocumentStructure(source_filename=filename, source_type="pdf", blocks=blocks)


# Disable TorchDynamo compilation to prevent 'cl is not found' errors on Windows
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def extract_pdf_structure(pdf_file, filename: str = "document.pdf") -> DocumentStructure:
    try:
        return _extract_with_docling(pdf_file, filename)
    except Exception as e:
        print(f"Docling extraction failed ({e}), falling back to pypdf...")
        # Reset file pointer just in case _extract_with_docling advanced it before failing
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        return _extract_with_pypdf_fallback(pdf_file, filename)
