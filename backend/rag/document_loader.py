"""Extract raw text from company documents (PDF, TXT, DOCX)."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_DOC_EXTENSIONS = {".pdf": "pdf", ".txt": "txt", ".md": "txt", ".docx": "docx"}


class DocumentLoadError(ValueError):
    pass


def detect_doc_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_DOC_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported document type '{ext}'. Only PDF, TXT, and DOCX are supported."
        )
    return SUPPORTED_DOC_EXTENSIONS[ext]


def extract_text(path: Path, doc_type: str) -> str:
    try:
        if doc_type == "pdf":
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[page {i + 1}]\n{text}")
            text = "\n\n".join(pages)
        elif doc_type == "txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif doc_type == "docx":
            from docx import Document as DocxDocument

            doc = DocxDocument(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:  # pragma: no cover - guarded by detect_doc_type
            raise DocumentLoadError(f"Unsupported document type '{doc_type}'.")
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Failed to extract text from {doc_type.upper()} file: {exc}") from exc

    if not text.strip():
        raise DocumentLoadError("No extractable text found in document (it may be a scanned/image-only file).")
    return text
