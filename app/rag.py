from __future__ import annotations

import time

from app.answering import UNKNOWN_ANSWER, AnswerGenerator
from app.chunking import chunk_pages
from app.document_loader import load_document
from app.embeddings import EmbeddingClient
from app.models import IngestResponse, QueryMetrics, QueryResponse
from app.vector_store import LocalVectorStore


class RAGService:
    def __init__(
        self,
        *,
        store: LocalVectorStore,
        embedder: EmbeddingClient,
        answerer: AnswerGenerator,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.answerer = answerer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_files(self, files: list[tuple[str, bytes]]) -> IngestResponse:
        start = time.perf_counter()
        all_chunks = []
        for filename, content in files:
            pages = load_document(filename, content)
            all_chunks.extend(
                chunk_pages(
                    pages,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )
            )

        embeddings = self.embedder.embed_texts([chunk.text for chunk in all_chunks])
        added = self.store.add(all_chunks, embeddings)
        return IngestResponse(
            documents_ingested=len(files),
            chunks_added=added,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_provider=self.embedder.provider_name,
            embedding_model=self.embedder.model_name,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    def answer_question(
        self,
        *,
        question: str,
        top_k: int,
        min_score: float,
    ) -> QueryResponse:
        total_start = time.perf_counter()

        embed_start = time.perf_counter()
        query_embedding = self.embedder.embed_texts([question])[0]
        embedding_ms = (time.perf_counter() - embed_start) * 1000

        retrieval_start = time.perf_counter()
        hits = self.store.search(query_embedding, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        grounded = bool(hits and hits[0].score >= min_score)
        generation_ms = 0.0
        if grounded:
            generation_start = time.perf_counter()
            answer = self.answerer.answer(question, hits)
            generation_ms = (time.perf_counter() - generation_start) * 1000
            if answer.strip() == UNKNOWN_ANSWER:
                grounded = False
        else:
            answer = UNKNOWN_ANSWER

        return QueryResponse(
            question=question,
            answer=answer,
            grounded=grounded,
            retrieved_chunks=hits,
            metrics=QueryMetrics(
                total_latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
                embedding_latency_ms=round(embedding_ms, 2),
                retrieval_latency_ms=round(retrieval_ms, 2),
                generation_latency_ms=round(generation_ms, 2),
            ),
            embedding_provider=self.embedder.provider_name,
            generation_provider=self.answerer.provider_name,
        )

