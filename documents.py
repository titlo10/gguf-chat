import codecs
import functools
import os

import constants as const


def _read_text_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
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


@functools.lru_cache(maxsize=1)
def _extract_document_text(path, _mtime_ns, _size):
    ext = os.path.splitext(path)[1].lower()
    if ext == const.PDF_EXT:
        return _extract_pdf(path)
    if ext == const.DOCX_EXT:
        return _extract_docx(path)
    return _read_text_file(path)


def extract_document_text(path):
    stat = os.stat(path)
    return _extract_document_text(path, stat.st_mtime_ns, stat.st_size)
