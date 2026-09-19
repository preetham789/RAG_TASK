# Explanation

**Design parameter.** I chose a 900-character chunk size with 150 characters of overlap. That value kept each retrieved chunk small enough to be specific, while the overlap preserved surrounding context when a fact landed near a boundary. I used character chunks instead of a larger framework so the chunking behavior is easy to inspect and tune.

**Observed failure.** My first smoke test turned a 430-character sample file into 58 chunks. The cause was a chunking bug near the tail of the document: after choosing a natural boundary, the next start position could move forward by only one character when the remaining text was shorter than the overlap. I fixed it by treating any final remainder smaller than `chunk_size` as one last chunk.

**Metric tracked.** I tracked request latency in the API response: embedding, retrieval, generation, and total time. In a local FastAPI smoke test, ingest took 3.75 ms and a grounded query took 0.85 ms total, with retrieval at 0.25 ms. That showed the local JSON vector store is fine for small assignments. The expensive part will be external API calls once OpenAI mode is enabled, so the code batches document embeddings.

**Not finished / next.** I did not add OCR for scanned PDFs, a frontend, auth, deployment, or a serious evaluation dataset. Next I would add a small labeled retrieval eval set, tune the similarity threshold against that set, add citation verification, and add OCR only if scanned documents are in scope.
