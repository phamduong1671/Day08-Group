"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests

PROJECT_DIR = Path(__file__).parent.parent
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass


def _clean_env_value(name: str) -> str:
    return os.getenv(name, "").strip().strip("\"'")


def _get_jina_api_key() -> str:
    api_key = _clean_env_value("JINA_API_KEY") or _clean_env_value("JINAAI_API_KEY")
    placeholders = {"your-api-key", "your_api_key", "YOUR_API_KEY", "api_key_cua_ban"}
    return "" if api_key in placeholders else api_key


def _strip_vietnamese_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _tokenize(text: str) -> list[str]:
    text = _strip_vietnamese_accents(text.lower())
    return re.findall(r"[a-z0-9]+", text)


def _local_relevance_score(query: str, content: str, original_score: float = 0.0) -> float:
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return float(original_score)

    content_token_set = set(content_tokens)
    overlap = sum(1 for token in query_tokens if token in content_token_set) / len(query_tokens)
    phrase_bonus = 0.15 if _strip_vietnamese_accents(query.lower()) in _strip_vietnamese_accents(content.lower()) else 0.0
    return float(overlap + phrase_bonus + 0.05 * original_score)


def _fallback_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    reranked: list[dict] = []

    for candidate in candidates:
        item = candidate.copy()
        item["metadata"] = dict(candidate.get("metadata", {}))
        item["score"] = _local_relevance_score(
            query=query,
            content=str(candidate.get("content", "")),
            original_score=float(candidate.get("score", 0.0) or 0.0),
        )
        item["metadata"]["reranker"] = "local_overlap_fallback"
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0

    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if top_k <= 0 or not candidates or not query.strip():
        return []

    api_key = _get_jina_api_key()
    if not api_key:
        return _fallback_rerank(query, candidates, top_k)

    documents = [str(candidate.get("content", "")) for candidate in candidates]
    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(candidates)),
            },
            timeout=(5, 20),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"! Không gọi được Jina Reranker; dùng local fallback ({type(exc).__name__})")
        return _fallback_rerank(query, candidates, top_k)

    reranked: list[dict] = []
    for result in payload.get("results", []):
        index = result.get("index")
        if index is None or index >= len(candidates):
            continue

        item = candidates[index].copy()
        item["metadata"] = dict(candidates[index].get("metadata", {}))
        item["score"] = float(result.get("relevance_score", 0.0))
        item["metadata"]["reranker"] = JINA_RERANK_MODEL
        item["metadata"]["original_score"] = candidates[index].get("score", 0.0)
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = remaining[0]
        best_score = float("-inf")

        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = _cosine_similarity(query_embedding, embedding)
            diversity_penalty = 0.0

            for selected_idx in selected:
                selected_embedding = candidates[selected_idx].get("embedding", [])
                diversity_penalty = max(
                    diversity_penalty,
                    _cosine_similarity(embedding, selected_embedding),
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["metadata"] = dict(candidates[idx].get("metadata", {}))
        item["metadata"]["reranker"] = "mmr"
        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            content = item.get("content", "")
            if not content:
                continue
            scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank)
            content_map[content] = item

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results: list[dict] = []
    for content, score in ranked[:top_k]:
        item = content_map[content].copy()
        item["metadata"] = dict(content_map[content].get("metadata", {}))
        item["score"] = float(score)
        item["metadata"]["reranker"] = "rrf"
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
