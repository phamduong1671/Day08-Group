"""Task 7 - Reranking helpers: token-overlap, MMR, and RRF."""

from __future__ import annotations

import math

from .task4_chunking_indexing import hash_embedding, tokenize


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


def _overlap_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    if not query_terms or not text_terms:
        return 0.0
    precision = len(query_terms & text_terms) / len(query_terms)
    density = len(query_terms & text_terms) / len(text_terms)
    return 0.85 * precision + 0.15 * density


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Lightweight cross-encoder substitute based on query/document token interaction."""
    scored: list[dict] = []
    for candidate in candidates:
        lexical = _overlap_score(query, candidate.get("content", ""))
        prior = float(candidate.get("score", 0.0))
        score = 0.75 * lexical + 0.25 * prior
        item = dict(candidate)
        item["metadata"] = dict(candidate.get("metadata", {}))
        item["score"] = float(score)
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance selection."""
    if not candidates:
        return []
    enriched = []
    for candidate in candidates:
        item = dict(candidate)
        item.setdefault("embedding", hash_embedding(candidate.get("content", "")))
        enriched.append(item)

    selected: list[int] = []
    selected_scores: dict[int, float] = {}
    remaining = list(range(len(enriched)))
    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            relevance = _cosine(query_embedding, enriched[idx]["embedding"])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine(enriched[idx]["embedding"], enriched[sel]["embedding"])
                    for sel in selected
                )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        item = dict(enriched[best_idx])
        item["score"] = float(best_score)
        enriched[best_idx] = item
        selected_scores[best_idx] = float(best_score)
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [enriched[i] for i in selected]


def _result_key(item: dict) -> str:
    metadata = item.get("metadata", {}) or {}
    return str(metadata.get("chunk_id") or metadata.get("source_path") or item.get("content", ""))


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion for merging multiple ranked result lists."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = _result_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items or item.get("score", 0) > items[key].get("score", 0):
                items[key] = dict(item)

    fused = []
    for key, score in scores.items():
        item = dict(items[key])
        item["metadata"] = dict(item.get("metadata", {}))
        item["score"] = float(score)
        fused.append(item)
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(hash_embedding(query), candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma túy", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ bị bắt vì sử dụng ma túy", "score": 0.7, "metadata": {}},
    ]
    for result in rerank("hình phạt tàng trữ ma túy", dummy, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
