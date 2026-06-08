"""Task 5 - Semantic search over the local Task 4 index."""

from __future__ import annotations

from .task4_chunking_indexing import ensure_index, hash_embedding


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks ranked by cosine similarity to a hashed query vector."""
    if not query.strip() or top_k <= 0:
        return []

    chunks = ensure_index()
    query_embedding = hash_embedding(query)
    results: list[dict] = []

    for chunk in chunks:
        embedding = chunk.get("embedding") or hash_embedding(chunk.get("content", ""))
        score = _dot(query_embedding, embedding)
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(score),
                "metadata": dict(chunk.get("metadata", {})),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
