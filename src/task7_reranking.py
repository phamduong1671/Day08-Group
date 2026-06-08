"""
Task 7 — Reranking Module.

Cung cấp 3 chiến lược (đều dùng được, mỗi cái một vai trò):
    - cross_encoder: BAAI/bge-reranker-v2-m3 — chấm lại relevance (query, doc) sâu hơn
      bi-encoder. Mặc định cho rerank() vì chất lượng cao nhất cho tiếng Việt.
    - mmr: Maximal Marginal Relevance — giảm trùng lặp, tăng đa dạng.
    - rrf: Reciprocal Rank Fusion — gộp nhiều ranked list (dùng ở Task 9 merge dense+sparse).
"""

from __future__ import annotations

import math


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng cross-encoder. Score chuẩn hoá về (0,1) bằng sigmoid để ngưỡng
    ở Task 9 có ý nghĩa nhất quán.
    """
    if not candidates:
        return []
    try:
        from .config import get_reranker

        reranker = get_reranker()
        pairs = [(query, c["content"]) for c in candidates]
        raw_scores = reranker.predict(pairs)
    except Exception:
        # Reranker không tải được → giữ thứ tự cũ, không crash demo.
        return candidates[:top_k]

    reranked = []
    for cand, score in zip(candidates, raw_scores):
        item = {**cand, "score": 1.0 / (1.0 + math.exp(-float(score)))}  # sigmoid
        reranked.append(item)
    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    MMR = λ·sim(query, doc) - (1-λ)·max sim(doc, đã_chọn).
    candidates cần có key 'embedding'.
    """
    if not candidates:
        return []
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            relevance = _cosine(query_embedding, emb)
            max_sim = max(
                (_cosine(emb, candidates[s].get("embedding", [])) for s in selected),
                default=0.0,
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion: RRF(d) = Σ 1/(k + rank_r(d)).
    Gộp dựa trên thứ hạng (không phụ thuộc thang điểm) → lý tưởng để trộn
    dense (cosine) với BM25 (thang khác hẳn). k=60 theo Cormack et al. 2009.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        results.append({**content_map[content], "score": score})
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified interface. Mặc định cross-encoder (chất lượng cao nhất)."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        raise ValueError("rerank_mmr cần query_embedding — gọi rerank_mmr() trực tiếp")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
