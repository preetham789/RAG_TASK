from app.chunking import chunk_pages
from app.document_loader import LoadedPage


def test_chunk_pages_creates_overlapping_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(220))
    chunks = chunk_pages(
        [LoadedPage(source_name="notes.txt", text=text)],
        chunk_size=180,
        overlap=40,
    )

    assert len(chunks) > 1
    assert len(chunks) < 20
    assert all(len(chunk.text) <= 180 for chunk in chunks)
    assert chunks[0].metadata.source_name == "notes.txt"
    assert chunks[0].chunk_id.startswith("chk_")


def test_short_text_stays_single_chunk() -> None:
    chunks = chunk_pages(
        [LoadedPage(source_name="short.txt", text="A short fact about basil.")],
        chunk_size=900,
        overlap=150,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "A short fact about basil."
