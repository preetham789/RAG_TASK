from __future__ import annotations

import json
import math
from pathlib import Path

from app.models import ChunkHit, DocumentChunk, StoreStats, StoredChunk


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be read or written."""


class LocalVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: list[StoredChunk] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._chunks = []
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._chunks = [StoredChunk.model_validate(item) for item in payload]
        except Exception as exc:
            raise VectorStoreError(f"Could not load vector store at {self.path}.") from exc

    def save(self) -> None:
        try:
            payload = [chunk.model_dump() for chunk in self._chunks]
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            raise VectorStoreError(f"Could not write vector store at {self.path}.") from exc

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts did not match.")

        existing_ids = {chunk.chunk_id for chunk in self._chunks}
        added = 0
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.chunk_id in existing_ids:
                continue
            self._chunks.append(
                StoredChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    embedding=embedding,
                )
            )
            existing_ids.add(chunk.chunk_id)
            added += 1
        self.save()
        return added

    def search(self, query_embedding: list[float], top_k: int) -> list[ChunkHit]:
        scored: list[tuple[float, StoredChunk]] = []
        for chunk in self._chunks:
            if len(chunk.embedding) != len(query_embedding):
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            ChunkHit(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=round(score, 4),
            )
            for score, chunk in scored[:top_k]
        ]

    def reset(self) -> None:
        self._chunks = []
        self.save()

    def stats(self) -> StoreStats:
        sources = sorted({chunk.metadata.source_name for chunk in self._chunks})
        return StoreStats(chunk_count=len(self._chunks), sources=sources)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

