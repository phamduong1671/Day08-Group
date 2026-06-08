"""
Task 6 — Lexical Search Module (BM25).

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao.
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn.
    - Length normalization: document dài không bị ưu tiên quá mức.
    - score(q,d) = Σ IDF(qi) * (tf(qi,d)*(k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization).

ĐIỂM MẤU CHỐT TIẾNG VIỆT: `str.split()` tách sai từ ghép ("tàng trữ", "ma tuý").
→ Dùng pyvi.ViTokenizer để word-segment, cùng tokenizer cho cả corpus lẫn query.

Corpus = chính tập chunk của Task 4 (chunk_documents(load_documents())) → nội dung
trùng khớp với những gì đã index vào Weaviate, để RRF ở Task 9 merge hợp lệ.
"""

from __future__ import annotations

from functools import lru_cache

from .task4_chunking_indexing import chunk_documents, load_documents


def _tokenize(text: str) -> list[str]:
    """Word-segment tiếng Việt rồi lowercase. Fallback split() nếu pyvi lỗi."""
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text.lower()).split()
    except Exception:
        return text.lower().split()


@lru_cache(maxsize=1)
def _get_index():
    """Build (corpus, bm25) một lần. Trả (None, None) nếu chưa có data."""
    corpus = chunk_documents(load_documents())
    if not corpus:
        return None, None
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    return corpus, BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khoá bằng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted desc.
    """
    import numpy as np

    corpus, bm25 = _get_index()
    if not corpus:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict] = []
    for idx in top_indices:
        if scores[idx] <= 0:  # bỏ chunk không khớp keyword nào
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            }
        )
    return results


if __name__ == "__main__":
    for r in lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
