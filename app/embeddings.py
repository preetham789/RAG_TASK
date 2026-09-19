from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


class EmbeddingError(RuntimeError):
    """Raised when the embedding provider fails."""


class EmbeddingClient(Protocol):
    provider_name: str
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class OpenAIEmbeddingClient:
    model_name: str = "text-embedding-3-small"
    batch_size: int = 64
    provider_name: str = "openai"

    def __post_init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise EmbeddingError("OPENAI_API_KEY is required for OpenAI embeddings.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError("OpenAI embeddings require the openai package.") from exc
        self._client = OpenAI()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        start = time.perf_counter()
        try:
            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset : offset + self.batch_size]
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                embeddings.extend([item.embedding for item in response.data])
        except Exception as exc:
            raise EmbeddingError(
                f"OpenAI embedding call failed after {time.perf_counter() - start:.2f}s."
            ) from exc
        if len(embeddings) != len(texts):
            raise EmbeddingError("Embedding provider returned an unexpected count.")
        return embeddings


@dataclass
class LocalHashEmbeddingClient:
    dimensions: int = 384
    provider_name: str = "local"
    model_name: str = "hashing-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def create_embedding_client(
    provider: str,
    model_name: str,
    local_dimensions: int,
) -> EmbeddingClient:
    if provider == "openai":
        return OpenAIEmbeddingClient(model_name=model_name)
    if provider == "local":
        return LocalHashEmbeddingClient(dimensions=local_dimensions)
    raise EmbeddingError(f"Unknown embedding provider: {provider}")

