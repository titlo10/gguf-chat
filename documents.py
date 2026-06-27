import os

import constants as const


def _read_text_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in const.DOC_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _extract_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(path):
    import docx
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def extract_document_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == const.PDF_EXT:
        return _extract_pdf(path)
    if ext == const.DOCX_EXT:
        return _extract_docx(path)
    return _read_text_file(path)
