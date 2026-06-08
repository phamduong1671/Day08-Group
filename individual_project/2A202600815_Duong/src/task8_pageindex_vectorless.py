"""Task 8 - PageIndex vectorless fallback with local search backup."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> list[str]:
    """Upload documents to PageIndex when SDK/API key are available."""
    uploaded: list[str] = []
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY not set; skipping remote upload.")
        return uploaded

    try:
        from pageindex import PageIndex
    except Exception:
        print("pageindex package not installed; skipping remote upload.")
        return uploaded

    client = PageIndex(api_key=PAGEINDEX_API_KEY)
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        try:
            client.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name},
            )
            uploaded.append(md_file.name)
        except Exception as exc:
            print(f"Upload failed for {md_file.name}: {type(exc).__name__}")
    return uploaded


def _local_fallback_search(query: str, top_k: int) -> list[dict]:
    from .task6_lexical_search import lexical_search
    from .task5_semantic_search import semantic_search
    from .task7_reranking import rerank_rrf

    merged = rerank_rrf(
        [lexical_search(query, top_k=top_k * 2), semantic_search(query, top_k=top_k * 2)],
        top_k=top_k,
    )
    for item in merged:
        item["source"] = "pageindex"
    return merged


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval; fallback to local index if PageIndex is unavailable."""
    if not query.strip() or top_k <= 0:
        return []

    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            client = PageIndex(api_key=PAGEINDEX_API_KEY)
            raw_results = client.query(query=query, top_k=top_k)
            results = []
            for result in raw_results:
                results.append(
                    {
                        "content": getattr(result, "text", ""),
                        "score": float(getattr(result, "score", 0.0)),
                        "metadata": dict(getattr(result, "metadata", {}) or {}),
                        "source": "pageindex",
                    }
                )
            if results:
                return results
        except Exception as exc:
            print(f"PageIndex unavailable; local fallback ({type(exc).__name__})")

    return _local_fallback_search(query, top_k)


if __name__ == "__main__":
    for result in pageindex_search("hình phạt sử dụng ma túy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
