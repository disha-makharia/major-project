"""Split extracted document text into overlapping chunks for embedding."""
from __future__ import annotations

import re

from backend.config import settings


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Character-based sliding-window chunking that tries to break on paragraph/sentence
    boundaries so chunks stay semantically coherent for retrieval."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
            buffer = buffer[-overlap:] if overlap else ""

        if len(para) <= chunk_size:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        else:
            # Paragraph itself is too long: hard-split with overlap.
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap if overlap else end
            buffer = ""

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c.strip()]
