"""Task 6 - Lexical search using BM25 over the local index."""

from __future__ import annotations

import math
from collections import Counter

from .task4_chunking_indexing import ensure_index, tokenize


CORPUS: list[dict] = []


class SimpleBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        document_count = max(1, len(tokenized_corpus))
        df: Counter[str] = Counter()
        for doc in tokenized_corpus:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for freqs, doc_len in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = [
            {
                "content": chunk.get("content", ""),
                "metadata": dict(chunk.get("metadata", {})),
            }
            for chunk in ensure_index()
        ]
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index; prefer rank-bm25 when installed."""
    tokenized_corpus = [tokenize(doc.get("content", "")) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:
        return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks ranked by BM25 keyword score."""
    if not query.strip() or top_k <= 0:
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    scores = list(bm25.get_scores(tokenize(query)))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results: list[dict] = []
    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": score,
                "metadata": dict(corpus[idx].get("metadata", {})),
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
