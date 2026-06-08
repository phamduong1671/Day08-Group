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

from pathlib import Path

_STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Chunk size mirrors Task 4 so both retrievers work on identical granularity.
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

# Module-level caches so repeated calls don't rebuild the index.
_bm25 = None


def _load_corpus() -> list[dict]:
    """Load and chunk all markdown files from data/standardized/."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for md_file in sorted(_STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = md_file.parent.name
        for i, chunk_text in enumerate(splitter.split_text(content)):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "chunk_index": i,
                },
            })
    return chunks


def _get_corpus_and_bm25():
    global CORPUS, _bm25
    if _bm25 is None:
        CORPUS = _load_corpus()
        _bm25 = build_bm25_index(CORPUS)
    return CORPUS, _bm25


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    # Split by whitespace — sufficient for Vietnamese because word boundaries
    # are space-delimited; underthesea would improve recall for compound words
    # but adds a heavy dependency not required by the task spec.
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


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
    import numpy as np

    corpus, bm25 = _get_corpus_and_bm25()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
