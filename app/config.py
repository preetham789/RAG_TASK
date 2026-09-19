from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class Settings:
    vector_store_path: Path
    embedding_provider: str
    generation_provider: str
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    default_min_score: float
    local_embedding_dimensions: int

    @classmethod
    def from_env(cls) -> "Settings":
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        has_groq_key = bool(os.getenv("GROQ_API_KEY"))
        embedding_provider = os.getenv(
            "RAG_EMBEDDING_PROVIDER", "openai" if has_openai_key else "local"
        ).lower()
        default_generation_provider = "extractive"
        if has_groq_key:
            default_generation_provider = "groq"
        elif has_openai_key:
            default_generation_provider = "openai"
        generation_provider = os.getenv(
            "RAG_GENERATION_PROVIDER", default_generation_provider
        ).lower()
        chunk_size = _env_int("RAG_CHUNK_SIZE", 900)
        chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", 150)
        if chunk_overlap >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

        return cls(
            vector_store_path=Path(
                os.getenv("RAG_VECTOR_STORE_PATH", "data/vector_store/index.json")
            ),
            embedding_provider=embedding_provider,
            generation_provider=generation_provider,
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            chat_model=os.getenv("RAG_CHAT_MODEL", _default_chat_model(generation_provider)),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            default_top_k=_env_int("RAG_TOP_K", 4),
            default_min_score=_env_float("RAG_MIN_SCORE", 0.18),
            local_embedding_dimensions=_env_int("RAG_LOCAL_EMBEDDING_DIMENSIONS", 384),
        )


def _default_chat_model(provider: str) -> str:
    if provider == "groq":
        return "openai/gpt-oss-20b"
    if provider == "openai":
        return "gpt-4.1-mini"
    return "token-overlap-sentences"
