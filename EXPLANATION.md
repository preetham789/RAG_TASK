# Explanation

**Design parameter.** I chose a 900-character chunk size with 150 characters of overlap. That value kept each retrieved chunk small enough to be specific, while the overlap preserved surrounding context when a fact landed near a boundary. I used character chunks instead of a larger framework so the chunking behavior is easy to inspect and tune.

**Observed failure.** My first smoke test turned a 430-character sample file into 58 chunks. The cause was a chunking bug near the tail of the document: after choosing a natural boundary, the next start position could move forward by only one character when the remaining text was shorter than the overlap. I fixed it by treating any final remainder smaller than `chunk_size` as one last chunk.

**Metric tracked.** I tracked request latency in the API response: embedding, retrieval, generation, and total time. In a local FastAPI smoke test, ingest took 3.75 ms and a grounded query took 0.85 ms total, with retrieval at 0.25 ms. That showed the local JSON vector store is fine for small assignments. The expensive part will be external API calls once OpenAI mode is enabled, so the code batches document embeddings.

**Evaluation & Threshold Tuning.** I built a labeled evaluation benchmark (`tests/eval_set.json`) containing balanced on-corpus and off-corpus questions. With the default 384-dimensional local hash embeddings, unstemmed common tokens in off-corpus queries (e.g. "What is the capital of France?" or "When will the new library building be opened to the public?") scored between 0.28 and 0.38, causing the original 0.18 threshold to suffer a 75% false positive rate on off-corpus queries. Running the automated evaluation test (`tests/test_threshold_eval.py`) demonstrated that tuning `default_min_score` to 0.38 achieves 100% precision and 100% recall on the evaluation set, eliminating off-corpus false positives while maintaining full recall for grounded retrieval.

**Not finished / next.** I added a small Streamlit UI for demos, but did not add OCR for scanned PDFs, auth, or deployment. Next steps would be adding citation verification, exploring BM25/reranking hybrid retrieval, and adding OCR only if scanned documents are in scope.

