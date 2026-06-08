"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pageindex import PageIndexClient, PageIndexAPIError

PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(PROJECT_DIR / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "indexes"
PAGEINDEX_MANIFEST_PATH = INDEX_DIR / "pageindex_documents.json"
PAGEINDEX_FOLDER_ID = os.getenv("PAGEINDEX_FOLDER_ID", "").strip() or None
PAGEINDEX_BASE_URL = "https://api.pageindex.ai"
SUPPORTED_UPLOAD_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".doc"}
PAGEINDEX_UPLOAD_DIR = INDEX_DIR / "pageindex_uploads"


def _clean_api_key() -> str:
    api_key = PAGEINDEX_API_KEY.strip().strip("\"'")
    placeholders = {"your-api-key", "your_api_key", "YOUR_API_KEY", "api_key_cua_ban"}
    return "" if api_key in placeholders else api_key


def _read_manifest() -> dict[str, Any]:
    if not PAGEINDEX_MANIFEST_PATH.exists():
        return {"documents": []}
    return json.loads(PAGEINDEX_MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(manifest: dict[str, Any]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _iter_document_files() -> list[Path]:
    return [
        file
        for file in sorted(STANDARDIZED_DIR.rglob("*"))
        if file.is_file() and file.suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS
    ]


def _strip_to_pdf_text(text: str) -> str:
    import unicodedata

    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    text = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return text.encode("ascii", "ignore").decode("ascii")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_simple_pdf(text: str, output_path: Path, title: str) -> None:
    """
    Minimal text-only PDF writer for PageIndex upload when no PDF converter is
    installed. It uses built-in Helvetica and ASCII text.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = _strip_to_pdf_text(text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            lines.append("")
            continue
        while len(raw_line) > 95:
            split_at = raw_line.rfind(" ", 0, 95)
            if split_at <= 0:
                split_at = 95
            lines.append(raw_line[:split_at].strip())
            raw_line = raw_line[split_at:].strip()
        lines.append(raw_line)

    lines = [title, ""] + lines[:4000]
    lines_per_page = 48
    pages = [lines[index : index + lines_per_page] for index in range(0, len(lines), lines_per_page)]
    if not pages:
        pages = [[title]]

    objects: list[bytes] = []
    catalog_id = 1
    pages_id = 2
    font_id = 3
    next_id = 4
    page_ids: list[int] = []
    content_ids: list[int] = []

    for page_lines in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content_ids.append(content_id)

        commands = ["BT", "/F1 10 Tf", "50 790 Td", "12 TL"]
        for line in page_lines:
            commands.append(f"({_escape_pdf_text(line)}) Tj")
            commands.append("T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", "ignore")
        objects.append(
            f"{page_id} 0 obj\n<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>\nendobj\n".encode()
        )
        objects.append(
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream\nendobj\n"
        )

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    base_objects = {
        catalog_id: f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {pages_id} 0 R >>\nendobj\n".encode(),
        pages_id: f"{pages_id} 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>\nendobj\n".encode(),
        font_id: f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode(),
    }

    ordered: dict[int, bytes] = dict(base_objects)
    for page_id, content_id, page_obj, content_obj in zip(page_ids, content_ids, objects[0::2], objects[1::2]):
        ordered[page_id] = page_obj
        ordered[content_id] = content_obj

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id in sorted(ordered):
        offsets.append(len(pdf))
        pdf.extend(ordered[object_id])

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root {catalog_id} 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    )
    output_path.write_bytes(bytes(pdf))


def _prepare_pageindex_upload_files() -> list[tuple[Path, Path]]:
    prepared: list[tuple[Path, Path]] = []
    for source in _iter_document_files():
        if source.suffix.lower() == ".pdf":
            prepared.append((source, source))
            continue

        relative = source.relative_to(STANDARDIZED_DIR)
        output_pdf = PAGEINDEX_UPLOAD_DIR / relative.with_suffix(".pdf")
        if not output_pdf.exists() or output_pdf.stat().st_mtime < source.stat().st_mtime:
            content = source.read_text(encoding="utf-8", errors="ignore")
            _write_simple_pdf(content, output_pdf, title=source.name)
        prepared.append((source, output_pdf))

    return prepared


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_DIR).as_posix()


def _submit_document_with_timeout(client: PageIndexClient, file_path: Path) -> dict[str, Any]:
    data = {"if_retrieval": True}
    if PAGEINDEX_FOLDER_ID:
        data["folder_id"] = PAGEINDEX_FOLDER_ID

    with file_path.open("rb") as file:
        response = requests.post(
            f"{client.BASE_URL}/doc/",
            headers=client._headers(),
            files={"file": file},
            data=data,
            timeout=(10, 60),
        )

    if response.status_code != 200:
        raise PageIndexAPIError(f"Failed to submit document: {response.text}")
    return response.json()


def _remote_documents_from_manifest() -> list[dict[str, Any]]:
    manifest = _read_manifest()
    return [
        doc
        for doc in manifest.get("documents", [])
        if doc.get("doc_id") and doc.get("status") in {"submitted", "ready", "uploaded"}
    ]


def _list_existing_pageindex_documents(client: PageIndexClient) -> dict[str, dict[str, Any]]:
    try:
        response = client.list_documents(limit=100, folder_id=PAGEINDEX_FOLDER_ID)
    except TypeError:
        response = client.list_documents(limit=100)

    documents = response.get("documents", [])
    by_name: dict[str, dict[str, Any]] = {}
    for document in documents:
        name = document.get("name") or document.get("filename")
        if name:
            by_name[name] = document
    return by_name


def _submit_retrieval(api_key: str, doc_id: str, query: str) -> str:
    response = requests.post(
        f"{PAGEINDEX_BASE_URL}/retrieval/",
        headers={"api_key": api_key},
        json={"doc_id": doc_id, "query": query, "thinking": False},
        timeout=(5, 30),
    )
    if response.status_code != 200:
        raise PageIndexAPIError(f"Failed to submit retrieval: {response.text}")
    payload = response.json()
    return payload.get("retrieval_id") or payload.get("id") or ""


def _get_retrieval(api_key: str, retrieval_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{PAGEINDEX_BASE_URL}/retrieval/{retrieval_id}/",
        headers={"api_key": api_key},
        timeout=(5, 30),
    )
    if response.status_code != 200:
        raise PageIndexAPIError(f"Failed to get retrieval result: {response.text}")
    return response.json()


def _find_text_items(value: Any) -> list[dict[str, Any]]:
    """
    PageIndex response shapes can vary by API version; this walks the payload and
    extracts objects that look like retrieval hits.
    """
    results: list[dict[str, Any]] = []
    if isinstance(value, dict):
        text = (
            value.get("text")
            or value.get("content")
            or value.get("markdown")
            or value.get("answer")
            or value.get("snippet")
        )
        if isinstance(text, str) and text.strip():
            results.append(
                {
                    "content": text.strip(),
                    "score": float(value.get("score") or value.get("relevance_score") or 1.0),
                    "metadata": {
                        key: item
                        for key, item in value.items()
                        if key not in {"text", "content", "markdown", "answer", "snippet"}
                    },
                }
            )

        for item in value.values():
            results.extend(_find_text_items(item))
    elif isinstance(value, list):
        for item in value:
            results.extend(_find_text_items(item))

    return results


def _query_remote_pageindex(query: str, top_k: int) -> list[dict]:
    api_key = _clean_api_key()
    remote_docs = _remote_documents_from_manifest()
    if not api_key or not remote_docs:
        return []

    results: list[dict] = []
    for doc in remote_docs:
        try:
            retrieval_id = _submit_retrieval(api_key, doc["doc_id"], query)
            if not retrieval_id:
                continue

            payload: dict[str, Any] = {}
            for _ in range(6):
                payload = _get_retrieval(api_key, retrieval_id)
                status = str(payload.get("status", "")).lower()
                if status in {"completed", "complete", "done", "ready", "success"} or payload.get("result"):
                    break
                time.sleep(1)

            for item in _find_text_items(payload):
                metadata = item.get("metadata", {})
                metadata.update(
                    {
                        "doc_id": doc.get("doc_id"),
                        "source_path": doc.get("source_path"),
                        "filename": doc.get("filename"),
                        "retrieval_id": retrieval_id,
                        "retriever": "pageindex_remote",
                    }
                )
                results.append(
                    {
                        "content": item["content"],
                        "score": item["score"],
                        "metadata": metadata,
                        "source": "pageindex",
                    }
                )
        except Exception:
            continue

        if len(results) >= top_k:
            break

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    from src.task6_lexical_search import lexical_search

    results = []
    for item in lexical_search(query, top_k=top_k):
        metadata = dict(item.get("metadata", {}))
        metadata["retriever"] = "pageindex_local_vectorless_fallback"
        results.append(
            {
                "content": item.get("content", ""),
                "score": float(item.get("score", 0.0)),
                "metadata": metadata,
                "source": "pageindex",
            }
        )
    return results


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    api_key = _clean_api_key()
    files = _prepare_pageindex_upload_files()
    manifest = {
        "updated_at": datetime.now().isoformat(),
        "documents": [],
    }

    if not api_key:
        manifest["mode"] = "local_only_no_pageindex_api_key"
        manifest["documents"] = [
            {
                "filename": file.name,
                "source_path": _relative_path(file),
                "doc_id": None,
                "status": "local_only",
            }
            for source, _upload_file in files
            for file in [source]
        ]
        _write_manifest(manifest)
        return manifest

    client = PageIndexClient(api_key=api_key)
    existing_docs = _list_existing_pageindex_documents(client)
    limit_reached = False

    for source_file, upload_file in files:
        record = {
            "filename": source_file.name,
            "source_path": _relative_path(source_file),
            "upload_path": _relative_path(upload_file),
            "doc_id": None,
            "status": "failed",
        }

        existing = existing_docs.get(upload_file.name) or existing_docs.get(source_file.name)
        if existing:
            record["doc_id"] = existing.get("id") or existing.get("doc_id")
            record["status"] = "ready" if existing.get("status") == "completed" else "submitted"
            record["response"] = existing
            record["note"] = "reused_existing_pageindex_document"
            manifest["documents"].append(record)
            print(f"  ✓ Reused: {source_file.name}")
            continue

        if limit_reached:
            record["status"] = "skipped_limit_reached"
            record["error"] = "PageIndex upload limit already reached; reused existing documents where possible"
            manifest["documents"].append(record)
            print(f"  ! Skipped: {source_file.name} (PageIndex limit reached)")
            continue

        try:
            response = _submit_document_with_timeout(client, upload_file)
            record["doc_id"] = response.get("doc_id") or response.get("id")
            record["status"] = "submitted" if record["doc_id"] else "uploaded"
            record["response"] = response
            print(f"  ✓ Uploaded: {source_file.name}")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if "LimitReached" in str(exc):
                limit_reached = True
                record["status"] = "skipped_limit_reached"
                print(f"  ! Skipped: {source_file.name} (PageIndex limit reached)")
            else:
                print(f"  ! PageIndex upload failed for {source_file.name}: {type(exc).__name__}")

        manifest["documents"].append(record)

    _write_manifest(manifest)
    return manifest


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0 or not query.strip():
        return []

    try:
        results = _query_remote_pageindex(query.strip(), top_k)
        if results:
            return results
    except Exception as exc:
        print(f"! PageIndex remote search failed; dùng local vectorless fallback ({type(exc).__name__})")

    return _local_vectorless_search(query.strip(), top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
