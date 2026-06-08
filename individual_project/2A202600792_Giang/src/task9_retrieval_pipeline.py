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
    from concurrent.futures import ThreadPoolExecutor

    # Step 1: Run semantic + lexical in parallel.
    # Both are I/O-bound (Weaviate socket / disk reads), so threading helps.
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, query, top_k * 2)
        sparse_future = executor.submit(lexical_search, query, top_k * 2)
        try:
            dense_results = dense_future.result()
        except Exception:
            dense_results = []
        try:
            sparse_results = sparse_future.result()
        except Exception:
            sparse_results = []

    # Step 2: Merge with Reciprocal Rank Fusion.
    # RRF is chosen because it is model-free and handles score-scale
    # incompatibility between cosine similarity and BM25 naturally.
    candidate_lists = [lst for lst in (dense_results, sparse_results) if lst]
    if candidate_lists:
        merged = rerank_rrf(candidate_lists, top_k=top_k * 2)
    else:
        merged = []
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank with cross-encoder; fall back to RRF order on API error.
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            for item in final_results:
                item.setdefault("source", "hybrid")
        except Exception:
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Step 4: Fallback to PageIndex when the best hybrid score is too low.
    best_score = final_results[0]["score"] if final_results else 0.0
    if best_score < score_threshold:
        print(
            f"  ⚠ Hybrid best score {best_score:.3f} < threshold {score_threshold}. "
            "Falling back to PageIndex."
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            pass

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
