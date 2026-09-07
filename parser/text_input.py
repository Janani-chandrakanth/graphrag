"""
Pasted-text input adapter.

Lets the UI (Build Graph / Update Graph tabs) accept typed or pasted
text the same way it accepts an uploaded .txt file, by wrapping the
string in a file-like object exposing the same `.name` + `.read()`
contract that parser/parser.py's parse_document() dispatches on
(Streamlit's UploadedFile has the same shape) -- no changes needed to
parser/parser.py or parser/txt_extractor.py at all.
"""

import io
from datetime import datetime, timezone


class PastedTextFile(io.BytesIO):
    """BytesIO with a `.name` attribute."""

    def __init__(self, text: str, filename: str = None):
        super().__init__((text or "").encode("utf-8"))
        self.name = filename or f"pasted_text_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.txt"


def wrap_pasted_text(text: str, filename: str = None) -> PastedTextFile:
    """Wrap raw text so it can be passed straight into
    parser.parser.parse_document() / parse_document's callers."""
    return PastedTextFile(text, filename)
