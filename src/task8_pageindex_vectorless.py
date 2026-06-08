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
import re
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 2]


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
    relative_path = md_file.relative_to(Path(__file__).parent.parent).as_posix()
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


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("! PAGEINDEX_API_KEY chưa có; bỏ qua upload và dùng local vectorless fallback")
        return {"uploaded": 0, "backend": "local_markdown_blocks"}

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        uploaded = 0
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name},
            )
            uploaded += 1
            print(f"  ✓ Uploaded: {md_file.name}")

        return {"uploaded": uploaded, "backend": "pageindex"}
    except Exception as exc:
        print(f"! Không upload được PageIndex; dùng local fallback ({type(exc).__name__})")
        return {"uploaded": 0, "backend": "local_markdown_blocks", "error": str(exc)}


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

    if PAGEINDEX_API_KEY and os.getenv("USE_PAGEINDEX_API", "0") == "1":
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)

            return [
                {
                    "content": getattr(result, "text", ""),
                    "score": float(getattr(result, "score", 0.0)),
                    "metadata": getattr(result, "metadata", {}) or {},
                    "source": "pageindex",
                }
                for result in results
            ]
        except Exception as exc:
            print(f"! Không query được PageIndex; dùng local vectorless fallback ({type(exc).__name__})")

    return _local_vectorless_search(query, top_k)


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
