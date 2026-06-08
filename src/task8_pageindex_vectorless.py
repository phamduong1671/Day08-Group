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
import os
import time
from pathlib import Path
from typing import Any

from .config import LANDING_DIR, PAGEINDEX_API_KEY

DOC_MAP_PATH = Path(__file__).parent.parent / "data" / "pageindex_docs.json"
PROJECT_DIR = Path(__file__).parent.parent
READY_TIMEOUT = int(os.getenv("PAGEINDEX_READY_TIMEOUT", "600"))
QUERY_TIMEOUT = int(os.getenv("PAGEINDEX_QUERY_TIMEOUT", "90"))
POLL_INTERVAL = float(os.getenv("PAGEINDEX_POLL_INTERVAL", "2"))


def _get_client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env — đăng ký tại pageindex.ai")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_map() -> dict:
    if DOC_MAP_PATH.exists():
        return json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def _document_sources() -> list[Path]:
    """PageIndex SDK hiện chỉ nhận PDF; không gửi markdown với MIME sai."""
    return sorted((LANDING_DIR / "legal").glob("*.pdf"))


def _doc_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("doc_id") or value.get("id") or "")
    return str(getattr(value, "doc_id", "") or getattr(value, "id", ""))


def upload_documents(wait_until_ready: bool = True) -> dict:
    """
    Upload PDF luật lên PageIndex đúng một lần.
    Lưu mapping giàu metadata nhưng vẫn đọc được mapping legacy {filename: doc_id}.
    """
    client = _get_client()

    sources = _document_sources()
    if not sources:
        print("! Không có PDF để upload trong data/landing/legal/")
        return {}

    doc_map = _load_doc_map()
    for path in sources:
        if path.name in doc_map:
            continue
        result = client.submit_document(str(path))
        doc_id = _doc_id(result)
        if doc_id:
            doc_map[path.name] = {
                "doc_id": doc_id,
                "source": path.name,
                "source_path": path.relative_to(PROJECT_DIR).as_posix(),
                "type": "legal",
            }
            print(f"  Uploaded: {path.name} -> {doc_id}")

    DOC_MAP_PATH.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")
    if wait_until_ready:
        for value in doc_map.values():
            doc_id = _doc_id(value)
            deadline = time.monotonic() + READY_TIMEOUT
            while doc_id and time.monotonic() < deadline:
                if client.is_retrieval_ready(doc_id):
                    break
                time.sleep(POLL_INTERVAL)
    return doc_map


def _first(data: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = data.get(key) if isinstance(data, dict) else getattr(data, key, None)
        if value not in (None, "", []):
            return value
    return default


def _extract_nodes(retrieval_result, top_k: int, document: dict) -> list[dict]:
    """Chuẩn hoá kết quả retrieval của PageIndex về format chung (defensive)."""
    # PageIndex trả các node cây kèm relevance; cấu trúc có thể là dict/obj.
    payload = _first(retrieval_result, "result", "data", default=retrieval_result)
    nodes = _first(
        payload,
        "results",
        "nodes",
        "retrieval_results",
        "retrieved_nodes",
        "relevant_nodes",
        "references",
        default=[],
    )
    if isinstance(nodes, dict):
        nodes = list(nodes.values())

    out: list[dict] = []
    for rank, n in enumerate(nodes[:top_k], 1):
        content = _first(n, "text", "content", "node_text", "summary", "snippet", default="")
        section_title = _first(n, "title", "section_title", default="")
        physical_index = None
        if not content:
            relevant_groups = _first(n, "relevant_contents", default=[]) or []
            relevant_parts: list[str] = []
            for group in relevant_groups:
                entries = group if isinstance(group, list) else [group]
                for entry in entries:
                    relevant = _first(entry, "relevant_content", "content", "text", default="")
                    if relevant:
                        relevant_parts.append(str(relevant))
                    section_title = _first(entry, "section_title", default=section_title)
                    physical_index = _first(entry, "physical_index", default=physical_index)
            content = "\n\n".join(relevant_parts)

        # Retrieval API ranks nodes but currently does not expose numeric relevance.
        # Reciprocal rank preserves that ordering and gives the UI a stable [0, 1] score.
        score = _first(n, "score", "relevance_score", "similarity", default=1.0 / rank)
        out.append(
            {
                "content": str(content),
                "score": float(score or 0.0),
                "metadata": {
                    "source": document.get("source", ""),
                    "source_path": document.get("source_path", ""),
                    "type": document.get("type", "legal"),
                    "title": section_title,
                    "node_id": _first(n, "node_id", "id", default=""),
                    "page": _first(n, "page", "page_number", default=physical_index),
                },
                "source": "pageindex",
            }
        )
    return [item for item in out if item["content"]]


def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    deadline = time.monotonic() + QUERY_TIMEOUT
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = client.get_retrieval(retrieval_id)
        status = str(_first(latest, "status", default="")).lower()
        if status in {"completed", "complete", "ready", "success", "succeeded"}:
            return latest
        if status in {"failed", "error", "cancelled"}:
            return latest
        if _first(latest, "results", "nodes", "retrieved_nodes", "result", default=None):
            return latest
        time.sleep(POLL_INTERVAL)
    return latest


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
    for filename, value in doc_map.items():
        document = value if isinstance(value, dict) else {"doc_id": value}
        document = {
            "source": filename,
            "source_path": f"data/landing/legal/{filename}",
            "type": "legal",
            **document,
        }
        doc_id = _doc_id(document)
        try:
            if not client.is_retrieval_ready(doc_id):
                continue
            submitted = client.submit_query(doc_id, query, thinking=True)
            retrieval_id = _first(submitted, "retrieval_id", "id", default="")
            if not retrieval_id:
                continue
            retrieval = _wait_for_retrieval(client, str(retrieval_id))
            results.extend(_extract_nodes(retrieval, top_k, document))
        except Exception as exc:
            print(f"! PageIndex query failed for {filename}: {type(exc).__name__}")
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
