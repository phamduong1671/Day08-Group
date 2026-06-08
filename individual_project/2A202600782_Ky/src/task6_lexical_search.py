"""
Task 6 — Lexical Search Module (BM25).

==============================================================================
THIẾT KẾ — tầng RECALL từ-vựng, MẢNH QUAN TRỌNG NHẤT cho corpus pháp luật.
==============================================================================
Query pháp lý thường chứa ĐỊNH DANH CHÍNH XÁC mà dense (Task 5) hay trượt:
số Điều ("Điều 248"), mã văn bản ("57/2022/NĐ-CP"), tên hoá chất
("Methamphetamine"). BM25 khớp đúng token nên bù chính xác điểm yếu đó.

Quyết định:
  1. Corpus = data/index/chunks.jsonl (cùng tập với vector store ở Task 4)
     -> dense & sparse xếp trên CÙNG không gian -> RRF ở Task 9 mới hợp lệ.
  2. Tokenize tiếng Việt bằng pyvi (word-segmentation): "tàng trữ", "ma túy"
     là TỪ GHÉP 2 âm tiết; split thô tách rời -> nhiễu IDF. Dùng CÙNG tokenizer
     cho cả corpus lẫn query. Fallback split() nếu chưa cài pyvi.
  3. GIỮ định danh: lowercase + bỏ markdown emphasis, NHƯNG không bỏ số/dấu
     gạch chéo/dấu tiếng Việt -> "248", "57/2022/nđ-cp" còn nguyên làm token.

Cơ chế BM25 (BM25Okapi, k1=1.5, b=0.75):
    score(q,d) = Σ IDF(qi) · tf(qi,d)·(k1+1) / (tf(qi,d) + k1·(1−b + b·|d|/avgdl))
  - TF: từ xuất hiện nhiều trong doc -> điểm cao (bão hoà bởi k1).
  - IDF: từ hiếm trong corpus -> trọng số lớn hơn.
  - |d|/avgdl: chuẩn hoá độ dài, doc dài không bị ưu tiên quá mức (điều chỉnh b).
"""

import re

from .task4_chunking_indexing import load_chunks

# Index dựng 1 lần rồi cache (lazy) — tránh tokenize lại corpus mỗi truy vấn.
_BM25 = None
_CORPUS: list[dict] = []

# Token hợp lệ: chữ (có dấu tiếng Việt), chữ số, và '/.' để giữ mã văn bản như
# 57/2022/nđ-cp. \w trong Python (re.UNICODE) đã bao gồm ký tự có dấu.
_TOKEN_RE = re.compile(r"[\w/]+", re.UNICODE)
_DIEU_RE = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
_DOCCODE_RE = re.compile(r"\d+\s*/\s*\d+\s*/\s*[\w\-]+", re.IGNORECASE)


def _normalize_code(text: str) -> str:
    return re.sub(r"[^0-9a-zA-ZàáâãèéêìíòóôõùúýđĐ/-]+", "", text).lower()


def _tokenize(text: str) -> list[str]:
    """Chuẩn hoá + tách token tiếng Việt (pyvi word-segmentation, fallback split)."""
    text = text.lower().replace("_", " ")          # bỏ '_' của markdown emphasis
    try:
        from pyvi import ViTokenizer
        # ViTokenizer nối các âm tiết cùng từ bằng '_'; ta tách lại thành token từ.
        text = ViTokenizer.tokenize(text)
        tokens = []
        for w in text.split():
            tokens.extend(_TOKEN_RE.findall(w.replace("_", "")))
        return [t for t in tokens if t]
    except Exception:
        # Fallback: tách theo regex giữ định danh (đủ tốt nếu chưa cài pyvi).
        return _TOKEN_RE.findall(text)


def build_bm25_index(corpus: list[dict]):
    """Dựng BM25 index từ corpus [{'content', 'metadata'}].

    Returns:
        (bm25, corpus) — giữ corpus song song để map ngược chỉ số -> chunk.
    """
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized)   # k1=1.5, b=0.75 mặc định
    return bm25, corpus


def _get_index():
    """Lazy-load: đọc chunks.jsonl (source-of-truth Task 4) và dựng index 1 lần."""
    global _BM25, _CORPUS
    if _BM25 is None:
        _CORPUS = load_chunks()
        _BM25, _ = build_bm25_index(_CORPUS)
    return _BM25, _CORPUS


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm từ khoá bằng BM25.

    Args:
        query: Câu truy vấn.
        top_k: Số kết quả tối đa.

    Returns:
        List of {'content', 'score', 'metadata'} sorted by score giảm dần,
        chỉ giữ chunk có score > 0 (có khớp token).
    """
    import numpy as np

    bm25, corpus = _get_index()
    scores = bm25.get_scores(_tokenize(query))
    boosted = scores.copy()

    query_lower = query.lower()
    query_dieu = {int(m.group(1)) for m in _DIEU_RE.finditer(query)}
    query_codes = {_normalize_code(c) for c in _DOCCODE_RE.findall(query)}
    phrase_boosts = [
        phrase for phrase in (
            "tàng trữ trái phép",
            "vận chuyển trái phép",
            "mua bán trái phép",
            "tổ chức sử dụng",
            "cai nghiện bắt buộc",
            "cai nghiện tự nguyện",
        )
        if phrase in query_lower
    ]

    for idx, doc in enumerate(corpus):
        meta = doc.get("metadata", {})
        title_text = (meta.get("doc_title", "") + " " + meta.get("dieu_title", "")).lower()
        text = (doc.get("content", "") + " " + meta.get("source", "") + " " +
                title_text).lower()
        if query_dieu and meta.get("dieu") in query_dieu:
            boosted[idx] += 20.0
        for code in query_codes:
            if code and code in _normalize_code(text):
                boosted[idx] += 15.0
        for phrase in phrase_boosts:
            if phrase in title_text:
                boosted[idx] += 12.0
            if phrase in text:
                boosted[idx] += 5.0

    top_idx = np.argsort(boosted)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if boosted[idx] <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": float(boosted[idx]),
                "metadata": corpus[idx]["metadata"],
            }
        )
    return results


if __name__ == "__main__":
    for q in ["Điều 248 tàng trữ trái phép chất ma tuý",
              "57/2022/NĐ-CP"]:
        print(f"\nQuery: {q}\n" + "-" * 60)
        for r in lexical_search(q, top_k=5):
            m = r["metadata"]
            tag = f"Điều {m.get('dieu')}" if m.get("dieu") else m.get("source", "")
            print(f"[{r['score']:.2f}] ({tag}) {r['content'][:90]}...")
