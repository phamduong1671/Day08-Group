"""
Task 8 — PageIndex Vectorless RAG.

PageIndex retrieve trên CÂY CẤU TRÚC tài liệu (Điều/Khoản/Chương) thay vì embedding
→ rất hợp văn bản luật. Dùng làm fallback khi hybrid search yếu.

API thật (KHÁC ví dụ cũ trong scaffold):
    from pageindex import PageIndexClient
    pi = PageIndexClient(api_key=...)
    doc = pi.submit_document("file.pdf")            # async → trả doc_id
    pi.is_retrieval_ready(doc_id)                   # poll tới khi xử lý xong
    pi.submit_retrieval_query(doc_id, query)        # truy vấn

Quy trình: upload_documents() chạy MỘT LẦN offline, lưu {filename: doc_id} ra
data/pageindex_docs.json. Lúc query chỉ đọc lại doc_id (không upload lại).

Upload PDF gốc ở data/landing/legal/ (PageIndex mạnh nhất trên PDF có cấu trúc).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import LANDING_DIR, PAGEINDEX_API_KEY, STANDARDIZED_DIR

DOC_MAP_PATH = Path(__file__).parent.parent / "data" / "pageindex_docs.json"


def _get_client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env — đăng ký tại pageindex.ai")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_map() -> dict:
    if DOC_MAP_PATH.exists():
        return json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def upload_documents() -> dict:
    """
    Upload tài liệu lên PageIndex (PDF luật ưu tiên, fallback markdown chuẩn hoá).
    Lưu mapping {filename: doc_id} ra disk. Trả mapping.
    """
    client = _get_client()

    sources: list[Path] = sorted((LANDING_DIR / "legal").glob("*.pdf"))
    if not sources:  # chưa có PDF → thử markdown đã chuẩn hoá
        sources = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not sources:
        print("⚠ Không có tài liệu để upload (data/landing/legal/*.pdf hoặc standardized/*.md)")
        return {}

    doc_map = _load_doc_map()
    for path in sources:
        if path.name in doc_map:
            continue  # đã upload trước đó
        result = client.submit_document(str(path))
        doc_id = result.get("doc_id") if isinstance(result, dict) else getattr(result, "doc_id", None)
        if doc_id:
            doc_map[path.name] = doc_id
            print(f"  ✓ Uploaded: {path.name} → {doc_id}")

    DOC_MAP_PATH.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc_map


def _extract_nodes(retrieval_result, top_k: int) -> list[dict]:
    """Chuẩn hoá kết quả retrieval của PageIndex về format chung (defensive)."""
    # PageIndex trả các node cây kèm relevance; cấu trúc có thể là dict/obj.
    nodes = []
    if isinstance(retrieval_result, dict):
        nodes = retrieval_result.get("results") or retrieval_result.get("nodes") or []
    else:
        nodes = getattr(retrieval_result, "results", None) or getattr(retrieval_result, "nodes", []) or []

    out: list[dict] = []
    for n in nodes[:top_k]:
        get = (lambda k, d=None: n.get(k, d)) if isinstance(n, dict) else (lambda k, d=None: getattr(n, k, d))
        out.append(
            {
                "content": get("text") or get("content") or "",
                "score": float(get("score", 0.0) or 0.0),
                "metadata": {"title": get("title", ""), "node_id": get("node_id", "")},
                "source": "pageindex",
            }
        )
    return out


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex. Trả [] nếu chưa cấu hình (test sẽ skip).

    Returns:
        List of {'content', 'score', 'metadata', 'source':'pageindex'}.
    """
    try:
        client = _get_client()
    except Exception:
        return []

    doc_map = _load_doc_map()
    if not doc_map:
        return []

    results: list[dict] = []
    for doc_id in doc_map.values():
        try:
            # Đợi document sẵn sàng (đã xử lý xong cây cấu trúc).
            for _ in range(10):
                if client.is_retrieval_ready(doc_id):
                    break
                time.sleep(1)
            retrieval = client.submit_retrieval_query(doc_id, query)
            results.extend(_extract_nodes(retrieval, top_k))
        except Exception:
            continue

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Set PAGEINDEX_API_KEY trong .env — đăng ký tại https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        for r in pageindex_search("hình phạt sử dụng ma tuý", top_k=3):
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
