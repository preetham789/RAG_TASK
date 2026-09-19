# Transparent RAG Question Answering

This is a small FastAPI RAG service for the interview task. A user uploads PDF or TXT reference documents, the service chunks and embeds them, stores vectors in a local JSON index, retrieves relevant chunks for a question, and answers only from those chunks.

The code avoids black-box RAG frameworks on purpose. The interesting pieces are visible in `app/chunking.py`, `app/embeddings.py`, `app/vector_store.py`, and `app/answering.py`.

## What Works

- Upload `.txt` and text-based `.pdf` documents.
- Chunk documents with a configurable character window and overlap.
- Embed chunks with OpenAI (`text-embedding-3-small` by default) or deterministic local hash embeddings for offline tests.
- Store vectors locally in `data/vector_store/index.json`.
- Retrieve top-k chunks with cosine similarity.
- Generate grounded answers with OpenAI (`gpt-4.1-mini` by default) or a simple extractive fallback for offline demos.
- Return the answer, grounding status, source chunks, scores, and latency metrics.
- Say `I don't know from the provided documents.` when retrieval is below the configured threshold or the answerer cannot ground the response.

## What Does Not Work Yet

- No OCR for scanned PDFs. PDFs must contain extractable text.
- No frontend, authentication, deployment, or hosted database.
- No robust citation verifier beyond returning the retrieved chunks and instructing the answerer to cite chunk ids.
- Retrieval quality was checked with a tiny smoke set, not a full evaluation suite.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the intended AI path, set an OpenAI API key:

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

## Walkthrough Video Checklist

For the 3-5 minute video:

1. Start the API and upload a new TXT or PDF that is not in this repo.
2. Ask a question with an answer in the document and show returned chunks.
3. Ask a question whose answer is absent and show the honest unknown response.
4. Walk through `app/chunking.py` and `app/rag.py`.
5. Show `prompts/grounded_answer_prompt.txt` as the prompt used for grounded answering.

