"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _safe_search(search_fn, query: str, top_k: int, label: str) -> list[dict]:
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"! {label} search lỗi; bỏ qua nhánh này ({type(exc).__name__})")
        return []


def _mark_source(results: list[dict], source: str) -> list[dict]:
    marked: list[dict] = []
    for result in results:
        item = result.copy()
        item["source"] = source
        marked.append(item)
    return marked


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if top_k <= 0 or not query.strip():
        return []

    # Lấy rộng hơn top_k để reranker có cơ hội kéo đúng điều luật lên trên.
    expanded_top_k = max(top_k * 6, 20)

    dense_results = _safe_search(semantic_search, query, expanded_top_k, "Semantic")
    sparse_results = _safe_search(lexical_search, query, expanded_top_k, "Lexical")

    if dense_results and sparse_results:
        merged = rerank_rrf([dense_results, sparse_results], top_k=expanded_top_k)
    else:
        merged = sorted(
            dense_results + sparse_results,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )[:expanded_top_k]

    merged = _mark_source(merged, "hybrid")

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _mark_source(final_results, "hybrid")
    else:
        final_results = merged[:top_k]

    best_score = float(final_results[0].get("score", 0.0)) if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ! Hybrid score ({best_score:.3f}) < threshold "
            f"({score_threshold}). Fallback -> PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
