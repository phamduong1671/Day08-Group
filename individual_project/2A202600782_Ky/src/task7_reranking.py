"""
Task 7 — Reranking Module.

==============================================================================
THIẾT KẾ — tầng PRECISION của pipeline 2 tầng. Dùng đủ 3 kỹ thuật, đúng vai:
==============================================================================
  • cross_encoder (VAI CHÍNH): tinh chỉnh top kết quả. Bi-encoder ở Task 5 nén
    query và chunk ĐỘC LẬP -> mất tương tác chi tiết; cross-encoder đọc ĐỒNG
    THỜI cặp (query, chunk) qua attention chéo -> phân biệt được "tàng trữ"
    (Điều 249) vs "mua bán" (Điều 251) mà dense dễ nhầm vì ngữ nghĩa rất gần.
    Model: BAAI/bge-reranker-v2-m3 — local, multilingual mạnh tiếng Việt, cùng
    họ bge-m3, KHÔNG cần API key.
  • mmr: đa dạng hoá -> tránh top-k toàn mảnh của CÙNG một Điều (vd bảng Danh
    mục nd57 bị cắt nhiều part). MMR = λ·rel(q,d) − (1−λ)·max sim(d, đã chọn).
  • rrf: hợp nhất nhiều ranked list (dense+sparse) bằng thứ HẠNG, không cần
    cùng thang điểm. Đây là vai thật của RRF -> Task 9 gọi để fuse Task 5 & 6.
"""

import math
from typing import Optional

from .model_runtime import load_with_cpu_fallback

# Cross-encoder nạp 1 lần rồi cache (lazy) — model nặng, tránh nạp lại mỗi query.
_RERANKER = None
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# =============================================================================
# Cross-encoder (vai chính)
# =============================================================================
def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = load_with_cpu_fallback(
            lambda device: CrossEncoder(RERANKER_MODEL, device=device),
            "RERANKER_DEVICE",
        )
    return _RERANKER


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank bằng cross-encoder bge-reranker-v2-m3 (chấm điểm cặp query-chunk)."""
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)   # logit relevance, càng cao càng liên quan

    ranked = sorted(
        ({**c, "score": float(s)} for c, s in zip(candidates, scores)),
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_k]


# =============================================================================
# MMR (đa dạng hoá)
# =============================================================================
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _ensure_embeddings(query: str, candidates: list[dict]) -> list[float]:
    """Đảm bảo mỗi candidate có 'embedding'; trả về luôn embedding của query."""
    from .task4_chunking_indexing import _get_embedder

    model = _get_embedder()
    missing = [c for c in candidates if "embedding" not in c]
    if missing:
        embs = model.encode([c["content"] for c in missing], normalize_embeddings=True)
        for c, e in zip(missing, embs):
            c["embedding"] = e.tolist()
    return model.encode([query], normalize_embeddings=True)[0].tolist()


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance — cân bằng liên quan và đa dạng."""
    if not candidates:
        return []
    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < top_k:
        best_idx, best = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx]["embedding"]
            rel = _cosine(query_embedding, emb)
            div = max((_cosine(emb, candidates[s]["embedding"]) for s in selected),
                      default=0.0)
            mmr = lambda_param * rel - (1 - lambda_param) * div
            if mmr > best:
                best, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[i], "score": _cosine(query_embedding, candidates[i]["embedding"])}
            for i in selected]


# =============================================================================
# RRF (hợp nhất nhiều ranker — dùng ở Task 9)
# =============================================================================
def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: RRF(d) = Σ 1/(k + rank_r(d)).

    Gộp theo THỨ HẠNG nên không cần dense/sparse cùng thang điểm. Khử trùng lặp
    theo chunk_id (nếu có) rồi tới content.
    """
    rrf: dict[str, float] = {}
    item_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = item.get("metadata", {}).get("chunk_id") or item["content"]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank)
            item_map.setdefault(key, item)

    out = []
    for key, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        out.append({**item_map[key], "score": score})
    return out


# =============================================================================
# Giao diện thống nhất
# =============================================================================
def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",   # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """Rerank thống nhất. Mặc định cross_encoder (precision cao nhất cho luật)."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        q_emb = _ensure_embeddings(query, candidates)
        return rerank_mmr(q_emb, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhiều list; ở đây coi mỗi candidate là 1 list 1-phần tử là vô
        # nghĩa -> yêu cầu gọi rerank_rrf trực tiếp với các ranked_lists.
        raise ValueError("RRF cần nhiều ranked_lists -> gọi rerank_rrf() trực tiếp.")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 249. Tội tàng trữ trái phép chất ma tuý", "score": 0.5, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Điều 251. Tội mua bán trái phép chất ma tuý", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
