from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    source_name: str
    page_number: int | None = None


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class StoredChunk(DocumentChunk):
    embedding: list[float]


class ChunkHit(DocumentChunk):
    score: float = Field(ge=-1.0, le=1.0)


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_added: int
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    latency_ms: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    min_score: float = Field(default=0.20, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class QueryMetrics(BaseModel):
    total_latency_ms: float
    embedding_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    retrieved_chunks: list[ChunkHit]
    metrics: QueryMetrics
    embedding_provider: str
    generation_provider: str


class StoreStats(BaseModel):
    chunk_count: int
    sources: list[str]

