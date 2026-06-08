"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

==============================================================================
THIẾT KẾ — ráp pipeline 2 tầng (recall -> precision) + fallback vectorless.
==============================================================================
   Query
     ├→ semantic_search (dense, Task 5) ─┐
     ├→ lexical_search  (BM25,  Task 6) ─┤→ RRF fuse (Task 7) ─→ cross-encoder
     │                                                              rerank (Task 7)
     │                                                                   │
     └→ nếu best_score < threshold ─→ pageindex_search (vectorless, Task 8)

Vì sao RRF để hợp nhất dense & sparse: hai bên có THANG ĐIỂM khác nhau (cosine
∈[0,1] vs BM25 ∈[0,~30]) -> không cộng trực tiếp được; RRF gộp theo THỨ HẠNG
nên công bằng. Sau đó cross-encoder đọc lại cặp (query, chunk) để xếp precision.

Chuẩn hoá điểm: cross-encoder trả LOGIT (không bị chặn) -> ta bọc sigmoid về
[0,1] để ngưỡng fallback (score_threshold) có ý nghĩa nhất quán.
"""

import math

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================
SCORE_THRESHOLD = 0.3            # best score (đã sigmoid) < ngưỡng -> fallback
DEFAULT_TOP_K = 5
RECALL_MULTIPLIER = 4            # lấy rộng ở tầng recall trước khi rerank
RRF_K = 60


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _safe_search(name: str, fn, query: str, top_k: int) -> list[dict]:
    try:
        return fn(query, top_k=top_k)
    except Exception as exc:
        print(f"  ⚠ {name} search failed: {exc}")
        return []


def _normalize_rrf_score(score: float) -> float:
    # Max with two rankers at rank 1: 2/(k+1). Keep threshold comparable.
    return min(1.0, score / (2.0 / (RRF_K + 1)))


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieval pipeline hoàn chỉnh với fallback vectorless.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'hybrid'|'pageindex'}.
    """
    recall_k = max(top_k * RECALL_MULTIPLIER, top_k)

    # --- Tầng recall: dense + sparse độc lập để một bên lỗi không làm sập pipeline ---
    dense = _safe_search("semantic", semantic_search, query, recall_k)
    sparse = _safe_search("lexical", lexical_search, query, recall_k)

    # --- Hợp nhất theo thứ hạng (RRF) ---
    ranked_lists = [r for r in (dense, sparse) if r]
    fused = rerank_rrf(ranked_lists, top_k=recall_k, k=RRF_K) if ranked_lists else []

    # --- Tầng precision: cross-encoder rerank + chuẩn hoá sigmoid ---
    if use_reranking and fused:
        try:
            ranked = rerank(query, fused, top_k=top_k, method="cross_encoder")
            for r in ranked:
                r["score"] = _sigmoid(r["score"])
                r["source"] = "hybrid"
        except Exception as exc:
            print(f"  ⚠ Rerank failed: {exc}; using normalized RRF.")
            ranked = fused[:top_k]
            for r in ranked:
                r["score"] = _normalize_rrf_score(r["score"])
                r["source"] = "hybrid"
    else:
        ranked = fused[:top_k]
        for r in ranked:
            r["score"] = _normalize_rrf_score(r["score"])
            r["source"] = "hybrid"

    best = ranked[0]["score"] if ranked else 0.0

    # --- Fallback: hybrid yếu -> vectorless PageIndex theo cấu trúc cây ---
    if not ranked or best < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]
        return []

    return ranked[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}\n" + "-" * 60)
        for i, r in enumerate(retrieve(q, top_k=3), 1):
            m = r.get("metadata", {})
            tag = f"Điều {m.get('dieu')}" if m.get("dieu") else m.get("source", "")
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] ({tag}) {r['content'][:70]}...")
