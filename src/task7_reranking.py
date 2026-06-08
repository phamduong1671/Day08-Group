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

TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 2]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _local_relevance_score(query: str, document: str, original_score: float = 0.0) -> float:
    """
    Reranker offline cho demo lớp học.

    Nó ưu tiên độ phủ từ khóa query trong document, cộng một phần nhỏ score từ
    retriever ban đầu. Khi có Jina/Qwen thật, có thể thay bằng cross-encoder.
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    if not query_tokens or not doc_tokens:
        return max(0.0, min(original_score, 1.0))

    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    overlap = len(query_set & doc_set)

    coverage = overlap / max(len(query_set), 1)
    precision = overlap / max(len(doc_set), 1)
    phrase_bonus = 0.1 if query.lower() in document.lower() else 0.0
    base = 0.75 * coverage + 0.20 * min(precision * 8, 1.0) + phrase_bonus

    # Original scores can be BM25, cosine, or RRF, so squash them gently.
    retriever_bonus = 0.05 * (original_score / (abs(original_score) + 1.0))
    return max(0.0, min(base + retriever_bonus, 1.0))


def _source_type_bonus(query: str, document: str, metadata: dict) -> float:
    query_lower = query.lower()
    document_lower = document.lower()
    doc_type = metadata.get("type") or metadata.get("doc_type") or ""

    legal_markers = [
        "luật",
        "điều",
        "khoản",
        "hình phạt",
        "mức phạt",
        "xử lý",
        "trách nhiệm hình sự",
        "quy định",
        "cai nghiện",
    ]
    news_markers = [
        "nghệ sĩ",
        "ca sĩ",
        "diễn viên",
        "người mẫu",
        "bị bắt",
        "tin tức",
        "vụ án",
    ]

    wants_legal = any(marker in query_lower for marker in legal_markers)
    wants_news = any(marker in query_lower for marker in news_markers)

    bonus = 0.0

    if doc_type == "legal" and wants_legal:
        bonus += 0.12
    if doc_type == "news" and wants_news:
        bonus += 0.12
    if doc_type == "news" and wants_legal and not wants_news:
        bonus -= 0.06
    if doc_type == "legal" and wants_news and not wants_legal:
        bonus -= 0.04

    if doc_type == "legal":
        legal_anchor_bonus = 0.0
        anchors = [
            ("tàng trữ", "điều 249"),
            ("vận chuyển", "điều 250"),
            ("mua bán", "điều 251"),
            ("chiếm đoạt", "điều 252"),
            ("tổ chức sử dụng", "điều 255"),
            ("cai nghiện", "cai nghiện ma túy"),
        ]
        for query_marker, doc_marker in anchors:
            if query_marker in query_lower and doc_marker in document_lower:
                legal_anchor_bonus += 0.18

        if ("hình phạt" in query_lower or "mức phạt" in query_lower or "xử lý" in query_lower) and "phạt tù" in document_lower:
            legal_anchor_bonus += 0.08

        bonus += min(legal_anchor_bonus, 0.28)

    return bonus


def _rerank_local(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    rescored: list[dict] = []
    for candidate in candidates:
        item = candidate.copy()
        item["original_score"] = _safe_float(candidate.get("score"))
        score = _local_relevance_score(
            query=query,
            document=str(candidate.get("content", "")),
            original_score=item["original_score"],
        )
        score += _source_type_bonus(
            query,
            str(candidate.get("content", "")),
            candidate.get("metadata", {}) or {},
        )
        item["score"] = max(0.0, min(score, 1.0))
        item["rerank_method"] = "local_token_overlap"
        rescored.append(item)

    rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored[:top_k]


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
    if top_k <= 0 or not candidates:
        return []

    use_jina = os.getenv("USE_JINA_RERANKER", "0") == "1"
    jina_api_key = os.getenv("JINA_API_KEY", "").strip()

    if use_jina and jina_api_key:
        try:
            import requests

            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [candidate.get("content", "") for candidate in candidates],
                    "top_n": top_k,
                },
                timeout=20,
            )
            response.raise_for_status()
            reranked = response.json().get("results", [])
            results: list[dict] = []
            for ranked in reranked:
                item = candidates[int(ranked["index"])].copy()
                item["original_score"] = _safe_float(item.get("score"))
                item["score"] = _safe_float(ranked.get("relevance_score"))
                item["rerank_method"] = "jina-reranker-v2-base-multilingual"
                results.append(item)
            if results:
                return results[:top_k]
        except Exception as exc:
            print(f"! Không gọi được Jina reranker; dùng local reranker ({type(exc).__name__})")

    return _rerank_local(query, candidates, top_k)


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
        best_idx: int | None = None
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding") or []
            relevance = _cosine_similarity(query_embedding, candidate_embedding)
            if not candidate_embedding:
                relevance = _safe_float(candidates[idx].get("score"))

            max_sim_to_selected = 0.0
            for selected_idx in selected:
                selected_embedding = candidates[selected_idx].get("embedding") or []
                max_sim_to_selected = max(
                    max_sim_to_selected,
                    _cosine_similarity(candidate_embedding, selected_embedding),
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    results: list[dict] = []
    for idx in selected:
        item = candidates[idx].copy()
        item["rerank_method"] = "mmr"
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
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    best_items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            content = str(item.get("content", ""))
            if not content:
                continue

            metadata = item.get("metadata", {})
            key = metadata.get("chunk_id") or metadata.get("source_path") or content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)

            previous = best_items.get(key)
            if previous is None or _safe_float(item.get("score")) > _safe_float(previous.get("score")):
                best_items[key] = item

    sorted_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)

    results: list[dict] = []
    for key in sorted_keys[:top_k]:
        item = best_items[key].copy()
        item["original_score"] = _safe_float(item.get("score"))
        item["score"] = rrf_scores[key]
        item["rerank_method"] = "rrf"
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
    if top_k <= 0 or not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return _rerank_local(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
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
