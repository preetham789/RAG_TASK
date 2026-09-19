from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class DocumentLoadError(ValueError):
    """Raised when an uploaded document cannot be parsed."""


@dataclass(frozen=True)
class LoadedPage:
    source_name: str
    text: str
    page_number: int | None = None


def load_document(filename: str, content: bytes) -> list[LoadedPage]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return _load_txt(filename, content)
    if suffix == ".pdf":
        return _load_pdf(filename, content)
    raise DocumentLoadError("Only .pdf and .txt documents are supported.")


def _load_txt(filename: str, content: bytes) -> list[LoadedPage]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("cp1252")
    if not text.strip():
        raise DocumentLoadError(f"{filename} did not contain readable text.")
    return [LoadedPage(source_name=filename, text=text)]


def _load_pdf(filename: str, content: bytes) -> list[LoadedPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError("PDF support requires pypdf. Install requirements.txt.") from exc

    try:
        import io

        reader = PdfReader(io.BytesIO(content))
        pages: list[LoadedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    LoadedPage(
                        source_name=filename,
                        text=text,
                        page_number=index,
                    )
                )
        if not pages:
            raise DocumentLoadError(
                f"{filename} did not contain extractable text. Scanned PDFs need OCR."
            )
        return pages
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Could not read {filename} as a PDF.") from exc

