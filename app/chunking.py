from __future__ import annotations

import hashlib
import re

from app.document_loader import LoadedPage
from app.models import ChunkMetadata, DocumentChunk


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def chunk_pages(
    pages: list[LoadedPage],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[DocumentChunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue
        for position, chunk_text in enumerate(_split_text(text, chunk_size, overlap)):
            chunk_id = _chunk_id(page.source_name, page.page_number, position, chunk_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        source_name=page.source_name,
                        page_number=page.page_number,
                    ),
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        if len(text) - start <= chunk_size:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        hard_end = min(start + chunk_size, len(text))
        end = _best_boundary(text, start, hard_end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _best_boundary(text: str, start: int, hard_end: int) -> int:
    window = text[start:hard_end]
    for marker in (". ", "? ", "! ", "\n\n", "; "):
        offset = window.rfind(marker)
        if offset >= int(len(window) * 0.55):
            return start + offset + len(marker)
    space = window.rfind(" ")
    if space >= int(len(window) * 0.7):
        return start + space
    return hard_end


def _chunk_id(
    source_name: str,
    page_number: int | None,
    position: int,
    text: str,
) -> str:
    digest = hashlib.sha256(
        f"{source_name}:{page_number}:{position}:{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"chk_{digest}"
