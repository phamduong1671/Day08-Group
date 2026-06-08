"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

    Query
      ├→ Semantic Search (dense)  ─┐
      ├→ Lexical Search (BM25)    ─┴→ Merge RRF → Rerank (cross-encoder) → results[source=hybrid]
      └→ Nếu rỗng hoặc top1.score < threshold → Fallback PageIndex [source=pageindex]

RRF dùng để merge vì thang điểm dense (cosine) và BM25 khác hẳn nhau — fuse theo
thứ hạng an toàn hơn weighted-sum. Cross-encoder rerank cho điểm (0,1) qua sigmoid
→ threshold so sánh được.
"""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _normalize_result(item: dict, retrieval_source: str) -> dict:
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("source", metadata.get("title", ""))
    metadata.setdefault("source_path", "")
    metadata.setdefault("type", "unknown")
    return {
        **item,
        "content": str(item.get("content", "")),
        "score": float(item.get("score", 0.0)),
        "metadata": metadata,
        "source": retrieval_source,
    }


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback.

    Returns:
        List of {'content', 'score', 'metadata', 'source' ∈ {'hybrid','pageindex'}}.
    """
    # Step 1: dense + sparse (lấy dư để rerank chọn lọc).
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: merge bằng RRF (fuse theo rank, không phụ thuộc thang điểm).
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)

    # Step 3: rerank cross-encoder để có relevance score (0,1).
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    final_results = [_normalize_result(item, "hybrid") for item in final_results]

    # Step 4: fallback PageIndex nếu hybrid rỗng hoặc điểm tốt nhất dưới ngưỡng.
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid yếu (best={best_score:.3f} < {score_threshold}). "
            f"Fallback → PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:  # chỉ thay khi PageIndex thực sự có kết quả
            return [_normalize_result(item, "pageindex") for item in fallback[:top_k]]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}\n{'-' * 60}")
        for i, r in enumerate(retrieve(q, top_k=3), 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
