"""Task 9 - Complete retrieval pipeline."""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _safe_search(search_fn, query: str, top_k: int, label: str) -> list[dict]:
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"{label} search failed; skipping ({type(exc).__name__})")
        return []


def _normalize(item: dict, source: str) -> dict:
    metadata = dict(item.get("metadata", {}) or {})
    metadata.setdefault("source", metadata.get("title") or metadata.get("source_path") or "unknown")
    metadata.setdefault("source_path", "")
    metadata.setdefault("type", "unknown")
    return {
        "content": str(item.get("content", "")),
        "score": float(item.get("score", 0.0)),
        "metadata": metadata,
        "source": source,
    }


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Run semantic + lexical retrieval, RRF merge, rerank, and fallback."""
    if not query.strip() or top_k <= 0:
        return []

    expanded_top_k = max(top_k * 2, top_k)
    dense_results = _safe_search(semantic_search, query, expanded_top_k, "Semantic")
    sparse_results = _safe_search(lexical_search, query, expanded_top_k, "Lexical")

    merged = rerank_rrf([dense_results, sparse_results], top_k=expanded_top_k)
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    final_results = [_normalize(item, "hybrid") for item in final_results]
    best_score = final_results[0]["score"] if final_results else 0.0

    if not final_results or best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return [_normalize(item, "pageindex") for item in fallback[:top_k]]

    return final_results[:top_k]


if __name__ == "__main__":
    for query in [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy",
    ]:
        print(f"\nQuery: {query}")
        for index, result in enumerate(retrieve(query, top_k=3), 1):
            print(f"{index}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
