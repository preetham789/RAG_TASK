from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.answering import GenerationError, create_answer_generator
from app.config import Settings
from app.document_loader import DocumentLoadError
from app.embeddings import EmbeddingError, create_embedding_client
from app.models import IngestResponse, QueryRequest, QueryResponse, StoreStats
from app.rag import RAGService
from app.vector_store import LocalVectorStore, VectorStoreError


app = FastAPI(
    title="Transparent RAG QA",
    version="0.1.0",
    description="Upload TXT/PDF documents, retrieve chunks, and answer from them.",
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    settings = get_settings()
    try:
        embedder = create_embedding_client(
            settings.embedding_provider,
            settings.embedding_model,
            settings.local_embedding_dimensions,
        )
        answerer = create_answer_generator(
            settings.generation_provider,
            settings.chat_model,
        )
    except (EmbeddingError, GenerationError) as exc:
        raise RuntimeError(str(exc)) from exc

    return RAGService(
        store=LocalVectorStore(settings.vector_store_path),
        embedder=embedder,
        answerer=answerer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@app.get("/health")
def health() -> dict[str, str | int | float]:
    settings = get_settings()
    stats = LocalVectorStore(settings.vector_store_path).stats()
    return {
        "status": "ok",
        "chunks": stats.chunk_count,
        "embedding_provider": settings.embedding_provider,
        "generation_provider": settings.generation_provider,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


@app.post("/documents", response_model=IngestResponse)
async def ingest_documents(files: list[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one document.")

    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        filename = upload.filename or "uploaded"
        if not filename.lower().endswith((".pdf", ".txt")):
            raise HTTPException(
                status_code=400,
                detail=f"{filename}: only .pdf and .txt files are supported.",
            )
        payloads.append((filename, await upload.read()))

    try:
        return get_service().ingest_files(payloads)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbeddingError, VectorStoreError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return get_service().answer_question(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except (EmbeddingError, GenerationError, VectorStoreError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/chunks", response_model=StoreStats)
def chunks() -> StoreStats:
    settings = get_settings()
    return LocalVectorStore(settings.vector_store_path).stats()


@app.delete("/documents", response_model=StoreStats)
def reset_documents() -> StoreStats:
    service = get_service()
    service.store.reset()
    return service.store.stats()

