"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import math
import re
from functools import lru_cache

from .task4_chunking_indexing import chunk_documents, load_documents

TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 2]


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[dict, ...]:
    """
    Load corpus từ markdown đã chuẩn hóa.

    Dùng lại chunking của Task 4 để lexical search và semantic search truy hồi
    trên cùng đơn vị tài liệu. Nếu chưa index vector store, BM25 vẫn chạy trực
    tiếp trên `data/standardized`.
    """
    documents = load_documents()
    chunks = chunk_documents(documents)
    return tuple(
        {
            "content": chunk.get("content", ""),
            "metadata": dict(chunk.get("metadata", {})),
        }
        for chunk in chunks
        if chunk.get("content")
    )


# Kept for notebooks/demo code that imports CORPUS directly.
CORPUS: list[dict] = list(_load_corpus())


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]

    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _get_bm25_index():
    return build_bm25_index(list(_load_corpus()))


def _fallback_keyword_scores(query_tokens: list[str], corpus: tuple[dict, ...]) -> list[float]:
    """TF-IDF nhẹ dùng khi môi trường thiếu `rank_bm25`."""
    if not query_tokens:
        return [0.0] * len(corpus)

    tokenized_docs = [_tokenize(doc.get("content", "")) for doc in corpus]
    total_docs = max(len(tokenized_docs), 1)
    document_frequency: dict[str, int] = {}

    for tokens in tokenized_docs:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    scores: list[float] = []
    for tokens in tokenized_docs:
        if not tokens:
            scores.append(0.0)
            continue

        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            tf = token_counts.get(token, 0)
            if tf == 0:
                continue
            idf = math.log((total_docs + 1) / (document_frequency.get(token, 0) + 1)) + 1
            score += (tf / len(tokens)) * idf

        scores.append(score)

    return scores


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    bm25 = _get_bm25_index()
    if bm25 is not None:
        raw_scores = [float(score) for score in bm25.get_scores(tokenized_query)]
    else:
        raw_scores = _fallback_keyword_scores(tokenized_query, corpus)

    ranked_indices = sorted(
        range(len(raw_scores)),
        key=lambda index: raw_scores[index],
        reverse=True,
    )

    results: list[dict] = []
    for index in ranked_indices[:top_k]:
        score = float(raw_scores[index])
        if score <= 0:
            continue

        doc = corpus[index]
        results.append(
            {
                "content": doc["content"],
                "score": score,
                "metadata": dict(doc.get("metadata", {})),
            }
        )

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
