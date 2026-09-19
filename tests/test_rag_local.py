from pathlib import Path

from app.answering import ExtractiveAnswerGenerator, UNKNOWN_ANSWER
from app.embeddings import LocalHashEmbeddingClient
from app.rag import RAGService
from app.vector_store import LocalVectorStore


def build_service(tmp_path: Path) -> RAGService:
    return RAGService(
        store=LocalVectorStore(tmp_path / "index.json"),
        embedder=LocalHashEmbeddingClient(dimensions=128),
        answerer=ExtractiveAnswerGenerator(),
        chunk_size=300,
        chunk_overlap=60,
    )


def test_answers_grounded_question_from_txt(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    service.ingest_files(
        [
            (
                "garden.txt",
                b"The basil is watered every Monday morning. The gate code is 1234.",
            )
        ]
    )

    response = service.answer_question(
        question="When is the basil watered?",
        top_k=2,
        min_score=0.05,
    )

    assert response.grounded is True
    assert "Monday" in response.answer
    assert response.retrieved_chunks


def test_says_unknown_when_retrieval_score_is_too_low(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    service.ingest_files([("garden.txt", b"The basil is watered every Monday morning.")])

    response = service.answer_question(
        question="What is the budget for the robotics team?",
        top_k=2,
        min_score=0.95,
    )

    assert response.grounded is False
    assert response.answer == UNKNOWN_ANSWER

