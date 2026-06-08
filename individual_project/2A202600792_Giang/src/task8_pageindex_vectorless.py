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
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Handle malformed key of the form "PAGEINDEX_API_KEY=actual_key"
_raw_key = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_API_KEY = _raw_key.split("=")[-1] if "=" in _raw_key else _raw_key

READY_TIMEOUT = int(os.getenv("PAGEINDEX_READY_TIMEOUT", "600"))
QUERY_TIMEOUT = int(os.getenv("PAGEINDEX_QUERY_TIMEOUT", "90"))

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"


def _get_client():
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _wait_for_ready(pi, doc_id: str, timeout: int) -> bool:
    """Poll until document is ready for retrieval or timeout expires."""
    deadline = time.time() + timeout
    interval = 5
    while time.time() < deadline:
        if pi.is_retrieval_ready(doc_id):
            return True
        time.sleep(interval)
    return False


def _poll_retrieval(pi, retrieval_id: str, timeout: int) -> dict | None:
    """Poll get_retrieval until status == 'completed' or timeout."""
    deadline = time.time() + timeout
    interval = 2
    while time.time() < deadline:
        result = pi.get_retrieval(retrieval_id)
        if result.get("status") == "completed":
            return result
        time.sleep(interval)
    return None


def upload_documents():
    """
    Upload toàn bộ legal documents (DOCX) lên PageIndex.

    Sử dụng landing directory vì PageIndex cần file gốc (không phải markdown).
    Sau khi upload, PageIndex sẽ tự động xử lý structural understanding.
    """
    pi = _get_client()

    # Check existing documents to avoid duplicate uploads
    existing = pi.list_documents()
    uploaded_names = {doc.get("name", "") for doc in existing.get("documents", [])}

    source_files = list(LANDING_DIR.rglob("*.docx")) + list(LANDING_DIR.rglob("*.pdf"))
    if not source_files:
        print("  ⚠ Không tìm thấy file DOCX/PDF trong landing directory")
        return []

    doc_ids = []
    for file_path in source_files:
        if file_path.name in uploaded_names:
            print(f"  ↷ Đã tồn tại, bỏ qua: {file_path.name}")
            continue

        print(f"  ↑ Uploading: {file_path.name} ...", end=" ", flush=True)
        response = pi.submit_document(str(file_path))
        doc_id = response.get("doc_id")
        if not doc_id:
            print(f"FAILED (no doc_id returned)")
            continue
        print(f"doc_id={doc_id}")
        doc_ids.append((doc_id, file_path.name))

    # Wait for all newly uploaded docs to become retrieval-ready
    print("\nWaiting for documents to be processed...")
    for doc_id, name in doc_ids:
        print(f"  Processing {name} ...", end=" ", flush=True)
        ready = _wait_for_ready(pi, doc_id, READY_TIMEOUT)
        print("ready ✓" if ready else "timeout ✗")

    return doc_ids


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
    pi = _get_client()

    # List all available documents
    docs_response = pi.list_documents()
    documents = docs_response.get("documents", [])
    if not documents:
        return []

    all_results: list[dict] = []

    for doc in documents:
        doc_id = doc.get("id")
        if not doc_id:
            continue

        # Only query documents that are ready
        if not pi.is_retrieval_ready(doc_id):
            continue

        # Submit async query
        try:
            query_response = pi.submit_query(doc_id, query)
        except Exception:
            continue

        retrieval_id = query_response.get("retrieval_id")
        if not retrieval_id:
            continue

        # Poll for results
        result_data = _poll_retrieval(pi, retrieval_id, QUERY_TIMEOUT)
        if not result_data:
            continue

        # Parse result — PageIndex may return nodes under different keys
        nodes = (
            result_data.get("results")
            or result_data.get("nodes")
            or result_data.get("retrieval_nodes")
            or []
        )

        for node in nodes:
            content = node.get("text") or node.get("content") or ""
            if not content:
                continue
            score = float(node.get("score") or node.get("relevance_score") or 0.0)
            metadata = {
                "doc_id": doc_id,
                "doc_name": doc.get("name", ""),
            }
            metadata.update(node.get("metadata") or {})

            all_results.append({
                "content": content,
                "score": score,
                "metadata": metadata,
                "source": "pageindex",
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


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
