from __future__ import annotations

import os

import streamlit as st

from app.answering import GenerationError, create_answer_generator
from app.config import Settings
from app.document_loader import DocumentLoadError
from app.embeddings import EmbeddingError, create_embedding_client
from app.rag import RAGService
from app.vector_store import LocalVectorStore, VectorStoreError


st.set_page_config(page_title="RAG QA", layout="wide")


def build_service() -> RAGService:
    settings = Settings.from_env()
    embedder = create_embedding_client(
        settings.embedding_provider,
        settings.embedding_model,
        settings.local_embedding_dimensions,
    )
    answerer = create_answer_generator(settings.generation_provider, settings.chat_model)
    return RAGService(
        store=LocalVectorStore(settings.vector_store_path),
        embedder=embedder,
        answerer=answerer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def sidebar_settings() -> str:
    st.sidebar.header("Settings")
    generation_provider = st.sidebar.selectbox(
        "Answer generator",
        ["groq", "openai", "extractive"],
        index=["groq", "openai", "extractive"].index(
            os.getenv("RAG_GENERATION_PROVIDER", "groq")
        )
        if os.getenv("RAG_GENERATION_PROVIDER", "groq") in {"groq", "openai", "extractive"}
        else 0,
    )
    os.environ["RAG_GENERATION_PROVIDER"] = generation_provider

    if generation_provider == "groq":
        groq_key = st.sidebar.text_input(
            "GROQ_API_KEY",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        os.environ["RAG_CHAT_MODEL"] = st.sidebar.text_input(
            "Groq model",
            value=os.getenv("RAG_CHAT_MODEL", "openai/gpt-oss-20b"),
        )
    elif generation_provider == "openai":
        openai_key = st.sidebar.text_input(
            "OPENAI_API_KEY",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["RAG_CHAT_MODEL"] = st.sidebar.text_input(
            "OpenAI chat model",
            value=os.getenv("RAG_CHAT_MODEL", "gpt-4.1-mini"),
        )

    embedding_provider = st.sidebar.selectbox(
        "Embedding provider",
        ["local", "openai"],
        index=0 if os.getenv("RAG_EMBEDDING_PROVIDER", "local") == "local" else 1,
    )
    os.environ["RAG_EMBEDDING_PROVIDER"] = embedding_provider
    if embedding_provider == "openai":
        openai_key = st.sidebar.text_input(
            "OpenAI key for embeddings",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            key="openai_embedding_key",
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    os.environ["RAG_TOP_K"] = str(st.sidebar.slider("Top K", 1, 10, int(os.getenv("RAG_TOP_K", "4"))))
    os.environ["RAG_MIN_SCORE"] = str(
        st.sidebar.slider(
            "Minimum retrieval score",
            0.0,
            1.0,
            float(os.getenv("RAG_MIN_SCORE", "0.20")),
            0.01,
        )
    )
    return generation_provider


def main() -> None:
    generation_provider = sidebar_settings()

    st.title("RAG Question Answering")
    st.caption("Upload PDF/TXT documents, retrieve matching chunks, and answer from those chunks.")

    if generation_provider == "groq" and not os.getenv("GROQ_API_KEY"):
        st.info("Enter a Groq API key in the sidebar to use Groq answer generation.")
        st.stop()

    try:
        service = build_service()
    except (EmbeddingError, GenerationError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Documents")
        uploads = st.file_uploader(
            "Upload reference documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        ingest = st.button("Ingest documents", type="primary", disabled=not uploads)
        if ingest and uploads:
            try:
                files = [(upload.name, upload.getvalue()) for upload in uploads]
                response = service.ingest_files(files)
                st.success(
                    f"Ingested {response.documents_ingested} document(s), "
                    f"added {response.chunks_added} chunk(s) in {response.latency_ms} ms."
                )
            except (DocumentLoadError, EmbeddingError, VectorStoreError) as exc:
                st.error(str(exc))

        stats = service.store.stats()
        st.metric("Stored chunks", stats.chunk_count)
        if stats.sources:
            st.write("Sources")
            st.dataframe({"source": stats.sources}, hide_index=True, use_container_width=True)

        if st.button("Reset local vector store"):
            service.store.reset()
            st.rerun()

    with right:
        st.subheader("Ask")
        question = st.text_area("Question", placeholder="What is RAG?")
        ask = st.button("Answer", type="primary", disabled=not question.strip())
        if ask:
            try:
                settings = Settings.from_env()
                response = service.answer_question(
                    question=question,
                    top_k=settings.default_top_k,
                    min_score=settings.default_min_score,
                )
                if response.grounded:
                    st.success(response.answer)
                else:
                    st.warning(response.answer)
                st.write("Metrics")
                st.json(response.metrics.model_dump())
                st.write("Retrieved chunks")
                for chunk in response.retrieved_chunks:
                    with st.expander(
                        f"{chunk.chunk_id} | score {chunk.score} | {chunk.metadata.source_name}",
                        expanded=False,
                    ):
                        st.write(chunk.text)
            except (EmbeddingError, GenerationError, VectorStoreError) as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
