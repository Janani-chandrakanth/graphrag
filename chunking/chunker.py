import re


def create_chunks(text):

    chunks = []

    pattern = r'(FR-\d{3}:.*?)(?=FR-\d{3}:|NFR-\d{3}:|BR-\d{3}|$)'

    fr_chunks = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    chunks.extend(fr_chunks)

    pattern = r'(NFR-\d{3}:.*?)(?=NFR-\d{3}:|BR-\d{3}|$)'

    nfr_chunks = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    chunks.extend(nfr_chunks)

    pattern = r'(BR-\d{3}.*?)(?=BR-\d{3}|$)'

    br_chunks = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    chunks.extend(br_chunks)

    return chunks