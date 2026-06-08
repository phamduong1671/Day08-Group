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

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"

# Hybrid fusion dùng weighted score fusion:
#   - semantic 0.55: ưu tiên BAAI/bge-m3 để bắt nghĩa tiếng Việt/đa ngôn ngữ
#   - lexical 0.45: BM25 rất quan trọng cho tên luật, điều khoản, cụm pháp lý
# Điểm từng retriever được chuẩn hóa trước khi cộng để BM25 không lấn át dense.
SEMANTIC_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.45
HYBRID_FETCH_MULTIPLIER = 4
MIN_HYBRID_CANDIDATES = 10
MAX_RERANK_CANDIDATES = 20
FUSION_METHOD = "weighted_score_fusion"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe_key(item: dict) -> str:
    metadata = item.get("metadata") or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"

    source_path = metadata.get("source_path") or metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if source_path and chunk_index is not None:
        return f"{source_path}#{chunk_index}"

    return str(item.get("content", "")).strip()


def _copy_result(item: dict) -> dict:
    return {
        "content": str(item.get("content", "")),
        "score": _safe_float(item.get("score")),
        "metadata": dict(item.get("metadata") or {}),
    }


def _safe_search(name: str, query: str, top_k: int) -> list[dict]:
    search_fn = semantic_search if name == "semantic" else lexical_search
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"! {name}_search failed; bỏ qua retriever này ({type(exc).__name__})")
        return []


def _normalise_dense_score(score: float) -> float:
    # Dense score từ cosine/Weaviate thường nằm trong khoảng 0..1 sau khi clamp.
    return max(0.0, min(1.0, score))


def _normalise_lexical_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, score / max_score))


def _add_weighted_results(
    merged: dict[str, dict],
    results: list[dict],
    retriever: str,
    weight: float,
    max_score: float = 1.0,
) -> None:
    for rank, original in enumerate(results, start=1):
        item = _copy_result(original)
        if not item["content"]:
            continue

        raw_score = item["score"]
        if retriever == "semantic":
            normalised_score = _normalise_dense_score(raw_score)
        else:
            normalised_score = _normalise_lexical_score(raw_score, max_score)

        # Rank signal giúp kết quả top của từng retriever vẫn có tiếng nói khi
        # score gốc của hai hệ khác thang đo.
        rank_signal = 1.0 / rank
        contribution = weight * (0.85 * normalised_score + 0.15 * rank_signal)

        key = _dedupe_key(item)
        if key not in merged:
            item["score"] = 0.0
            item["source"] = "hybrid"
            item["metadata"]["retrievers"] = []
            item["metadata"]["fusion_method"] = FUSION_METHOD
            item["metadata"]["fusion_weights"] = {
                "semantic": SEMANTIC_WEIGHT,
                "lexical": LEXICAL_WEIGHT,
            }
            merged[key] = item

        current = merged[key]
        current["score"] += contribution
        current["metadata"][f"{retriever}_score"] = raw_score
        current["metadata"][f"{retriever}_rank"] = rank
        if retriever not in current["metadata"]["retrievers"]:
            current["metadata"]["retrievers"].append(retriever)


def _merge_weighted_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
) -> list[dict]:
    merged: dict[str, dict] = {}
    max_lexical_score = max(
        [_safe_float(item.get("score")) for item in sparse_results] or [0.0]
    )

    _add_weighted_results(
        merged=merged,
        results=dense_results,
        retriever="semantic",
        weight=SEMANTIC_WEIGHT,
    )
    _add_weighted_results(
        merged=merged,
        results=sparse_results,
        retriever="lexical",
        weight=LEXICAL_WEIGHT,
        max_score=max_lexical_score,
    )

    results = list(merged.values())
    for item in results:
        item["score"] = min(1.0, _safe_float(item.get("score")))
        item["source"] = "hybrid"

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _ensure_source(results: list[dict], source: str) -> list[dict]:
    normalised: list[dict] = []
    for result in results:
        item = result.copy()
        item["metadata"] = dict(result.get("metadata") or {})
        item["score"] = _safe_float(result.get("score"))
        item["source"] = source
        normalised.append(item)
    return normalised


def _rerank_hybrid(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []

    rerank_candidates = candidates[: max(top_k, MAX_RERANK_CANDIDATES)]
    try:
        reranked = rerank(
            query=query,
            candidates=rerank_candidates,
            top_k=top_k,
            method=RERANK_METHOD,
        )
    except Exception as exc:
        print(f"! Rerank failed; dùng fusion ranking ({type(exc).__name__})")
        reranked = candidates[:top_k]

    return _ensure_source(reranked, "hybrid")


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
          ├→ Merge (weighted fusion) → merged_results
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

    query = query.strip()
    search_k = max(top_k * HYBRID_FETCH_MULTIPLIER, MIN_HYBRID_CANDIDATES)

    # Step 1: chạy semantic_search + lexical_search. Dùng ThreadPoolExecutor vì
    # semantic có thể gọi Weaviate còn lexical chạy local BM25.
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_safe_search, "semantic", query, search_k)
        sparse_future = executor.submit(_safe_search, "lexical", query, search_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # Step 2: merge bằng weighted fusion (semantic 0.55 + BM25 0.45).
    merged_results = _merge_weighted_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=max(search_k, MAX_RERANK_CANDIDATES),
    )

    # Step 3: rerank để re-score và re-order theo độ liên quan với query.
    if use_reranking:
        final_results = _rerank_hybrid(query, merged_results, top_k)
    else:
        final_results = _ensure_source(merged_results[:top_k], "hybrid")

    # Step 4: nếu hybrid yếu/rỗng thì fallback sang PageIndex vectorless.
    best_score = _safe_float(final_results[0].get("score")) if final_results else 0.0
    if best_score < score_threshold:
        print(
            f"! Hybrid score ({best_score:.3f}) < threshold "
            f"({score_threshold:.3f}); fallback PageIndex"
        )
        fallback_results = pageindex_search(query, top_k=top_k)
        return _ensure_source(fallback_results[:top_k], "pageindex")

    # Step 5: return top_k results.
    return _ensure_source(final_results[:top_k], "hybrid")


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
