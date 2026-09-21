# Transparent RAG Question Answering

Transparent RAG Question Answering is a Python application for answering questions from uploaded reference documents. It accepts PDF and TXT files, splits document text into searchable chunks, embeds those chunks, retrieves the most relevant context for a question, and generates an answer grounded in the retrieved material.

The project is intentionally lightweight and inspectable. It uses explicit document loading, chunking, embedding, vector search, and answer-generation modules instead of hiding the core retrieval flow inside a large RAG framework.

## Features

- Upload TXT files, text-based PDFs, and scanned/image-only PDFs.
- Extract text from regular PDFs with `pypdf`.
- OCR scanned PDFs with Groq vision OCR or local Tesseract OCR.
- Chunk documents with configurable chunk size and overlap.
- Embed chunks with OpenAI embeddings or deterministic local hash embeddings.
- Store embeddings in a local JSON vector store.
- Retrieve relevant chunks with cosine similarity.
- Generate grounded answers with Groq, OpenAI, or a local extractive fallback.
- Return retrieved chunks, source metadata, similarity scores, grounding status, and latency metrics.
- Respond with `I don't know from the provided documents.` when the retrieved material does not support an answer.
- Run as either a FastAPI service or a Streamlit app.

## Tech Stack

- Python
- FastAPI
- Streamlit
- Pydantic
- Groq API
- OpenAI API
- pypdf
- PyMuPDF
- pytesseract
- NumPy
- pytest

## Project Structure

```text
app/
  answering.py        Answer generation with Groq, OpenAI, or extractive fallback
  chunking.py         Text normalization and overlapping chunk creation
  config.py           Environment-based runtime settings
  document_loader.py  TXT, PDF, and OCR document loading
  embeddings.py       OpenAI and local hash embedding clients
  main.py             FastAPI routes
  models.py           Pydantic request/response models
  rag.py              Ingestion and query orchestration
  vector_store.py     Local JSON vector store and cosine search

streamlit_app.py      Streamlit user interface
requirements.txt      Python dependencies
sample_documents/     Sample input documents
tests/                Unit tests
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

The app is configured through environment variables.

Groq answer generation with local embeddings:

```powershell
$env:GROQ_API_KEY = "gsk_..."
$env:RAG_EMBEDDING_PROVIDER = "local"
$env:RAG_GENERATION_PROVIDER = "groq"
$env:RAG_CHAT_MODEL = "openai/gpt-oss-20b"
```

OpenAI embeddings and OpenAI answer generation:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:RAG_EMBEDDING_PROVIDER = "openai"
$env:RAG_GENERATION_PROVIDER = "openai"
```

Offline local mode:

```powershell
$env:RAG_EMBEDDING_PROVIDER = "local"
$env:RAG_GENERATION_PROVIDER = "extractive"
```

Scanned PDF OCR with Groq:

```powershell
$env:GROQ_API_KEY = "gsk_..."
$env:RAG_OCR_PROVIDER = "groq"
```

Scanned PDF OCR with local Tesseract:

```powershell
$env:RAG_OCR_PROVIDER = "tesseract"
```

## Run The Streamlit App

```powershell
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

If port `8501` is already in use:

```powershell
streamlit run streamlit_app.py --server.port 8502
```

## Run The FastAPI Service

```powershell
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

## API Usage

Upload documents:

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents `
  -F "files=@sample_documents/garden_guide.txt"
```

Ask a grounded question:

```powershell
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"When is greenhouse basil watered?\",\"top_k\":4,\"min_score\":0.18}"
```

Ask a question that is not answered by the documents:

```powershell
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Who won the 2026 World Cup?\",\"top_k\":4,\"min_score\":0.18}"
```

Reset the local vector store:

```powershell
curl.exe -X DELETE http://127.0.0.1:8000/documents
```

## API Endpoints

### `POST /documents`

Uploads one or more `.txt` or `.pdf` documents.

Response includes:

- Number of documents ingested
- Number of chunks added
- Chunk size and overlap
- Embedding provider and model
- Ingestion latency

### `POST /query`

Answers a question using retrieved document chunks.

Example request:

```json
{
  "question": "When is greenhouse basil watered?",
  "top_k": 4,
  "min_score": 0.18
}
```

Response includes:

- Answer
- Grounding status
- Retrieved chunks
- Source metadata
- Similarity scores
- Latency metrics

### `GET /chunks`

Returns the number of stored chunks and source document names.

### `DELETE /documents`

Clears the local vector store.

## Design Notes

Chunking uses a 900-character window with 150 characters of overlap by default. This keeps chunks focused while preserving enough surrounding context for facts that appear near chunk boundaries.

The vector store is a local JSON file at `data/vector_store/index.json`. This keeps the project easy to run locally and makes stored chunks inspectable without a separate database.

Retrieval uses cosine similarity over stored embeddings. The query endpoint accepts `top_k` and `min_score` so retrieval strictness can be tuned per request.

Answer generation is grounded by passing only retrieved chunks to the model. If retrieval does not produce a sufficiently relevant chunk, the app returns an explicit unknown answer instead of guessing.

## Testing

```powershell
pytest
```

The test suite uses local hash embeddings and the extractive answerer so it can run without external API calls.

## Limitations

- OCR quality depends on scan quality and page layout.
- Large scanned PDFs can take time to process because each page may require OCR.
- Local Tesseract OCR requires Tesseract to be installed separately on the machine.
- Groq OCR and model-based answer generation require a valid `GROQ_API_KEY`.
- OpenAI embeddings and OpenAI answer generation require a valid `OPENAI_API_KEY`.
- The local JSON vector store is intended for small local workloads, not large production-scale indexes.
- Citation verification is limited to returning the chunks used for retrieval.
