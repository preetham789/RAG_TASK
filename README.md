# Transparent RAG Question Answering

This is a small RAG question-answering system. A user uploads PDF or TXT reference documents, the app chunks and embeds them, stores vectors in a local JSON index, retrieves relevant chunks for a question, and answers only from those chunks.

The code avoids black-box RAG frameworks on purpose. The interesting pieces are visible in `app/chunking.py`, `app/embeddings.py`, `app/vector_store.py`, and `app/answering.py`.

## What Works

- Upload `.txt`, text-based `.pdf`, and scanned/image-only `.pdf` documents.
- Chunk documents with a configurable character window and overlap.
- Embed chunks with OpenAI (`text-embedding-3-small` by default) or deterministic local hash embeddings for offline tests.
- Store vectors locally in `data/vector_store/index.json`.
- Retrieve top-k chunks with cosine similarity.
- Generate grounded answers with Groq (`openai/gpt-oss-20b` by default), OpenAI, or a simple extractive fallback for offline demos.
- Run either as a FastAPI API or as a Streamlit app.
- Return the answer, grounding status, source chunks, scores, and latency metrics.
- Say `I don't know from the provided documents.` when retrieval is below the configured threshold or the answerer cannot ground the response.

## Current Limits

- OCR quality depends on scan quality.
- Local OCR needs Tesseract installed on the machine. Without Tesseract, scanned PDF OCR uses Groq when `GROQ_API_KEY` is configured.
- Large scanned PDFs can be slow or costly because OCR may process every page.
- No authentication, deployment, or hosted database.
- No robust citation verifier beyond returning the retrieved chunks and instructing the answerer to cite chunk ids.
- Retrieval quality was checked with a tiny smoke set, not a full evaluation suite.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the Groq answer-generation path, set a Groq API key. Local embeddings are the default so you can use Groq without needing a separate embedding API:

```powershell
$env:GROQ_API_KEY = "gsk_..."
$env:RAG_EMBEDDING_PROVIDER = "local"
$env:RAG_GENERATION_PROVIDER = "groq"
$env:RAG_CHAT_MODEL = "openai/gpt-oss-20b"
```

For the OpenAI embedding/generation path, set an OpenAI API key:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:RAG_EMBEDDING_PROVIDER = "openai"
$env:RAG_GENERATION_PROVIDER = "openai"
```

For offline development without an API key:

```powershell
$env:RAG_EMBEDDING_PROVIDER = "local"
$env:RAG_GENERATION_PROVIDER = "extractive"
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Open docs at `http://127.0.0.1:8000/docs`.

Start the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

Open Streamlit at `http://localhost:8501`.

For scanned PDFs, use Groq OCR:

```powershell
$env:GROQ_API_KEY = "gsk_..."
$env:RAG_OCR_PROVIDER = "groq"
```

Or install local Tesseract OCR and use:

```powershell
$env:RAG_OCR_PROVIDER = "tesseract"
```

## Usage

Upload a TXT file:

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

Ask something not in the documents:

```powershell
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Who won the 2026 World Cup?\",\"top_k\":4,\"min_score\":0.18}"
```

Reset the local index:

```powershell
curl.exe -X DELETE http://127.0.0.1:8000/documents
```

## API Shape

`POST /documents`

- Form-data field: `files`
- Accepts one or more `.txt` or `.pdf` files.
- Returns chunk count, chunking settings, embedding provider/model, and ingest latency.

`POST /query`

```json
{
  "question": "When is greenhouse basil watered?",
  "top_k": 4,
  "min_score": 0.18
}
```

Returns:

- `answer`
- `grounded`
- `retrieved_chunks` with `chunk_id`, source metadata, text, and score
- latency metrics for embedding, retrieval, generation, and total request time

## Design Notes

Chunking uses 900 characters with 150 characters of overlap. I chose character-based chunking because it is transparent, dependency-light, and easy to explain. The boundary finder tries sentence endings first so chunks usually end at a natural place, then falls back to spaces or a hard cut.

The vector store is a JSON file instead of a database because the assignment does not require production infrastructure. This keeps setup simple and makes it easy to inspect stored chunks.

The default retrieval threshold is `0.18`. With OpenAI embeddings, that keeps loosely related chunks out while still allowing paraphrased questions to match. It is configurable per request because real document sets need tuning.

## Tests

```powershell
pytest
```

The tests use local hash embeddings and the extractive answerer so they run without network calls.
