from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF."""

    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_txt(txt_file):
    """Extract text from TXT."""

    return txt_file.read().decode("utf-8")


def extract_text(uploaded_file):
    """Extract text based on file type."""

    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "pdf":
        return extract_text_from_pdf(uploaded_file)

    elif extension == "txt":
        return extract_text_from_txt(uploaded_file)

    else:
        raise ValueError(f"Unsupported file type: {extension}")