"""
Task 8 — PageIndex Vectorless RAG.

PageIndex retrieve trên cây cấu trúc tài liệu (Điều/Khoản/Chương) thay vì
embedding, rất hợp văn bản luật. Dùng làm fallback khi hybrid search yếu.

Nếu PageIndex chưa có API key hoặc SDK không sẵn sàng, module fallback về tìm
kiếm vectorless local trên các block markdown để demo/test không crash.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from .config import LANDING_DIR, PAGEINDEX_API_KEY, PROJECT_DIR, STANDARDIZED_DIR
except Exception:
    PROJECT_DIR = Path(__file__).parent.parent
    LANDING_DIR = PROJECT_DIR / "data" / "landing"
    STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
    PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

DOC_MAP_PATH = PROJECT_DIR / "data" / "pageindex_docs.json"
READY_TIMEOUT = int(os.getenv("PAGEINDEX_READY_TIMEOUT", "600"))
QUERY_TIMEOUT = int(os.getenv("PAGEINDEX_QUERY_TIMEOUT", "90"))
POLL_INTERVAL = float(os.getenv("PAGEINDEX_POLL_INTERVAL", "2"))
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 2]


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
    """PageIndex SDK hiện mạnh nhất với PDF; ưu tiên PDF luật gốc."""
    return sorted((LANDING_DIR / "legal").glob("*.pdf"))


def _doc_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("doc_id") or value.get("id") or "")
    return str(getattr(value, "doc_id", "") or getattr(value, "id", ""))


def _first(data: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = data.get(key) if isinstance(data, dict) else getattr(data, key, None)
        if value not in (None, "", []):
            return value
    return default


def upload_documents(wait_until_ready: bool = True) -> dict:
    """
    Upload PDF luật lên PageIndex một lần.

    Lưu mapping giàu metadata nhưng vẫn đọc được mapping legacy {filename: doc_id}.
    Nếu thiếu API key, trả local backend metadata thay vì raise.
    """
    try:
        client = _get_client()
    except Exception as exc:
        print(f"! PageIndex chưa cấu hình; bỏ qua upload ({type(exc).__name__})")
        return {"uploaded": 0, "backend": "local_markdown_blocks"}

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

    DOC_MAP_PATH.write_text(
        json.dumps(doc_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if wait_until_ready:
        for value in doc_map.values():
            doc_id = _doc_id(value)
            deadline = time.monotonic() + READY_TIMEOUT
            while doc_id and time.monotonic() < deadline:
                if client.is_retrieval_ready(doc_id):
                    break
                time.sleep(POLL_INTERVAL)
    return doc_map


def _extract_nodes(retrieval_result, top_k: int, document: dict) -> list[dict]:
    """Chuẩn hóa kết quả retrieval của PageIndex về format chung."""
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
    for rank, node in enumerate(nodes[:top_k], 1):
        content = _first(node, "text", "content", "node_text", "summary", "snippet", default="")
        section_title = _first(node, "title", "section_title", default="")
        physical_index = None

        if not content:
            relevant_groups = _first(node, "relevant_contents", default=[]) or []
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

        # Retrieval API ranks nodes but may not expose numeric relevance.
        # Reciprocal rank preserves ordering and gives UI a stable [0,1] score.
        score = _first(node, "score", "relevance_score", "similarity", default=1.0 / rank)
        out.append(
            {
                "content": str(content),
                "score": float(score or 0.0),
                "metadata": {
                    "source": document.get("source", ""),
                    "source_path": document.get("source_path", ""),
                    "type": document.get("type", "legal"),
                    "title": section_title,
                    "node_id": _first(node, "node_id", "id", default=""),
                    "page": _first(node, "page", "page_number", default=physical_index),
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


@lru_cache(maxsize=1)
def _load_markdown_blocks() -> tuple[dict, ...]:
    blocks: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return tuple()

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        current = ""
        block_index = 0
        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= 1200:
                current = f"{current}\n\n{paragraph}".strip()
                continue

            if current:
                blocks.append(_make_block(md_file, current, block_index))
                block_index += 1
            current = paragraph

        if current:
            blocks.append(_make_block(md_file, current, block_index))

    return tuple(blocks)


def _make_block(md_file: Path, content: str, block_index: int) -> dict:
    relative_path = md_file.relative_to(PROJECT_DIR).as_posix()
    doc_type = "legal" if "/legal/" in f"/{relative_path}" else "news" if "/news/" in f"/{relative_path}" else "unknown"
    return {
        "content": content,
        "metadata": {
            "source": md_file.name,
            "source_path": relative_path,
            "type": doc_type,
            "block_index": block_index,
            "vectorless_backend": "local_markdown_blocks",
        },
    }


def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: list[dict] = []
    for block in _load_markdown_blocks():
        content = block.get("content", "")
        content_tokens = set(_tokenize(content))
        if not content_tokens:
            continue

        overlap = len(query_tokens & content_tokens)
        if overlap == 0:
            continue

        coverage = overlap / max(len(query_tokens), 1)
        precision = overlap / max(len(content_tokens), 1)
        score = min(coverage + precision * 4, 1.0)
        scored.append(
            {
                "content": content,
                "score": score,
                "metadata": dict(block.get("metadata", {})),
                "source": "pageindex",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _submit_retrieval_query(client, doc_id: str, query: str):
    if hasattr(client, "submit_query"):
        return client.submit_query(doc_id, query, thinking=True)
    if hasattr(client, "submit_retrieval_query"):
        return client.submit_retrieval_query(doc_id, query)
    raise AttributeError("PageIndex client has no supported retrieval query method")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex.

    Returns:
        List of {'content', 'score', 'metadata', 'source':'pageindex'}.
    """
    if top_k <= 0 or not query.strip():
        return []

    use_api = os.getenv("USE_PAGEINDEX_API", "1") != "0"
    if use_api and PAGEINDEX_API_KEY:
        try:
            client = _get_client()
            doc_map = _load_doc_map()
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
                    submitted = _submit_retrieval_query(client, doc_id, query)
                    retrieval_id = _first(submitted, "retrieval_id", "id", default="")
                    if not retrieval_id:
                        results.extend(_extract_nodes(submitted, top_k, document))
                        continue
                    retrieval = _wait_for_retrieval(client, str(retrieval_id))
                    results.extend(_extract_nodes(retrieval, top_k, document))
                except Exception as exc:
                    print(f"! PageIndex query failed for {filename}: {type(exc).__name__}")
                    continue

            results.sort(key=lambda item: item["score"], reverse=True)
            if results:
                return results[:top_k]
        except Exception as exc:
            print(f"! Không query được PageIndex; dùng local fallback ({type(exc).__name__})")

    return _local_vectorless_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Set PAGEINDEX_API_KEY trong .env — đăng ký tại https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    for r in pageindex_search("hình phạt sử dụng ma tuý", top_k=3):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
