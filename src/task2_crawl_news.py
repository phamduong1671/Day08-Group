"""
Task 2 - Crawl news articles about Vietnamese artists related to drug cases.

The script prefers Crawl4AI when it is installed. If Crawl4AI or Playwright is
unavailable, it falls back to requests/BeautifulSoup.

Run:
    python src/task2_crawl_news.py
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://vietnamnet.vn/nha-thiet-ke-nguyen-cong-tri-bi-bat-lien-quan-duong-day-ma-tuy-2424939.html",
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://vietnamnet.vn/rapper-binh-gold-duong-tinh-voi-ma-tuy-va-bi-csgt-truy-bat-tren-cao-toc-la-ai-2425199.html",
    "https://vietnamnet.vn/truy-to-ca-sy-chau-viet-cuong-sau-dem-thac-loan-kinh-hoang-500843.html",
    "https://vietnamnet.vn/ca-si-hoi-cho-thuong-xuyen-ru-ban-gai-moi-quen-dap-da-153974.html",
]


def setup_directory() -> None:
    """Create data/landing/news/ when missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slugify(text: str, fallback: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return (text or fallback)[:90]


def normalize_article(url: str, title: str, content: str, crawler: str) -> dict[str, Any]:
    return {
        "url": url,
        "source_domain": urlparse(url).netloc,
        "title": title.strip() or "Unknown title",
        "date_crawled": now_iso(),
        "crawler": crawler,
        "content_markdown": content.strip(),
    }


async def crawl_with_crawl4ai(url: str) -> dict[str, Any] | None:
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception:
        return None

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    content = getattr(result, "markdown", "") or ""
    metadata = getattr(result, "metadata", {}) or {}
    title = metadata.get("title") or metadata.get("og:title") or "Unknown title"
    if len(content.strip()) < 300:
        return None
    return normalize_article(url, title, content, "crawl4ai")


def crawl_with_requests(url: str) -> dict[str, Any] | None:
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return None

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Day08RAGCrawler/1.0)"},
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string
    meta_title = soup.find("meta", property="og:title")
    if meta_title and meta_title.get("content"):
        title = meta_title["content"]

    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all(["h1", "h2", "p"])
        if len(p.get_text(" ", strip=True)) > 40
    ]
    content = "\n\n".join(paragraphs)
    if len(content) < 300:
        return None
    markdown = f"# {title or 'Unknown title'}\n\n{content}"
    return normalize_article(url, title, markdown, "requests_bs4")


async def crawl_article(url: str) -> dict[str, Any]:
    """
    Crawl one article and return metadata + content.

    Returns:
        {
            "url": str,
            "source_domain": str,
            "title": str,
            "date_crawled": str,
            "crawler": str,
            "content_markdown": str,
        }
    """
    try:
        article = await crawl_with_crawl4ai(url)
        if article:
            return article
    except Exception as exc:
        print(f"  ! Crawl4AI failed: {exc}")

    try:
        article = crawl_with_requests(url)
        if article:
            return article
    except Exception as exc:
        print(f"  ! requests fallback failed: {exc}")

    raise RuntimeError(f"Could not crawl article: {url}")


def save_article(article: dict[str, Any], index: int) -> Path:
    title_slug = slugify(article.get("title", ""), f"article-{index:02d}")
    filepath = DATA_DIR / f"{index:02d}-{title_slug}.json"
    filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
    return filepath


async def crawl_all() -> list[Path]:
    """Crawl every URL in ARTICLE_URLS and save one JSON file per article."""
    setup_directory()
    saved_files: list[Path] = []

    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filepath = save_article(article, index)
        saved_files.append(filepath)
        print(f"  Saved: {filepath}")

    return saved_files


if __name__ == "__main__":
    asyncio.run(crawl_all())
