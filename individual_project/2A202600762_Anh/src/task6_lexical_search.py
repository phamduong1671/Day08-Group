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

import json
import re
import unicodedata
from pathlib import Path

from src.task4_chunking_indexing import LOCAL_INDEX_PATH, chunk_documents, load_documents

CORPUS: list[dict] = []
BM25_INDEX = None


def _strip_vietnamese_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def tokenize(text: str) -> list[str]:
    """
    Tokenizer đơn giản cho BM25.

    Bỏ dấu để các biến thể như "ma túy" và "ma tuý" cùng match thành "ma tuy".
    """
    text = _strip_vietnamese_accents(text.lower())
    return re.findall(r"[a-z0-9]+", text)


def load_corpus() -> list[dict]:
    """
    Load chunks đã index ở Task 4. Nếu file index local chưa có, tự chunk lại từ
    data/standardized để module vẫn chạy được trong môi trường mới.
    """
    if LOCAL_INDEX_PATH.exists():
        corpus: list[dict] = []
        with LOCAL_INDEX_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                corpus.append(
                    {
                        "content": chunk.get("content", ""),
                        "metadata": chunk.get("metadata", {}),
                    }
                )
        return corpus

    return [
        {"content": chunk["content"], "metadata": chunk.get("metadata", {})}
        for chunk in chunk_documents(load_documents())
    ]


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_corpus_and_index() -> tuple[list[dict], object]:
    global CORPUS, BM25_INDEX

    if not CORPUS:
        CORPUS = load_corpus()

    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(CORPUS)

    return CORPUS, BM25_INDEX


def _overlap_score(query_tokens: list[str], content: str) -> float:
    content_tokens = set(tokenize(content))
    if not query_tokens or not content_tokens:
        return 0.0
    return sum(1 for token in query_tokens if token in content_tokens) / len(query_tokens)


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

    corpus, bm25 = _get_corpus_and_index()
    if not corpus:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)

    results: list[dict] = []
    for idx, score in ranked:
        score = float(score)
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": score,
                "metadata": corpus[idx].get("metadata", {}),
            }
        )
        if len(results) >= top_k:
            return results

    # Trường hợp BM25 trả 0 do query quá ngắn/phổ biến, dùng overlap nhẹ để
    # vẫn trả được keyword matches có score dương.
    fallback: list[dict] = []
    for doc in corpus:
        score = _overlap_score(tokenized_query, doc["content"])
        if score <= 0:
            continue
        fallback.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": doc.get("metadata", {}),
            }
        )

    fallback.sort(key=lambda item: item["score"], reverse=True)
    return fallback[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
