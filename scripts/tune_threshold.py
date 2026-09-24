import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answering import ExtractiveAnswerGenerator, UNKNOWN_ANSWER
from app.embeddings import LocalHashEmbeddingClient
from app.rag import RAGService
from app.vector_store import LocalVectorStore


def run_evaluation():
    eval_set_path = Path("tests/eval_set.json")
    eval_set = json.loads(eval_set_path.read_text(encoding="utf-8"))

    doc_path = Path("sample_documents/garden_guide.txt")
    doc_bytes = doc_path.read_bytes()

    embedder = LocalHashEmbeddingClient(dimensions=384)
    answerer = ExtractiveAnswerGenerator()
    store_path = Path("data/vector_store/eval_index.json")
    store = LocalVectorStore(store_path)
    store.reset()

    service = RAGService(
        store=store,
        embedder=embedder,
        answerer=answerer,
        chunk_size=900,
        chunk_overlap=150,
    )

    service.ingest_files([("garden_guide.txt", doc_bytes)])

    results = []
    for item in eval_set:
        q = item["question"]
        q_emb = embedder.embed_texts([q])[0]
        hits = store.search(q_emb, top_k=4)
        top_score = hits[0].score if hits else 0.0
        results.append({
            "question": q,
            "label": item["label"],
            "expected_grounded": item["expected_grounded"],
            "expected_answer_contains": item.get("expected_answer_contains"),
            "top_score": top_score,
        })

    print(f"\n{'Label':<12} | {'Score':<8} | Question")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<12} | {r['top_score']:<8.4f} | {r['question']}")

    # Let's check false positives at 0.18
    fp_018 = [r for r in results if not r["expected_grounded"] and r["top_score"] >= 0.18]
    print(f"\nOff-corpus questions that score >= 0.18 (False Positives at 0.18): {len(fp_018)}")
    for r in fp_018:
        print(f"  Score: {r['top_score']:.4f} -> {r['question']}")

    on_scores = [r["top_score"] for r in results if r["expected_grounded"]]
    off_scores = [r["top_score"] for r in results if not r["expected_grounded"]]
    print(f"\nOn-corpus min score: {min(on_scores):.4f}, max: {max(on_scores):.4f}, avg: {sum(on_scores)/len(on_scores):.4f}")
    print(f"Off-corpus min score: {min(off_scores):.4f}, max: {max(off_scores):.4f}, avg: {sum(off_scores)/len(off_scores):.4f}")


if __name__ == "__main__":
    run_evaluation()
