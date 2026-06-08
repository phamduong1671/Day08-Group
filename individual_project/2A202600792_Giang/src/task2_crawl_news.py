"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://nld.com.vn/clip-ca-si-long-nhat-su-dung-ma-tuy-tu-khi-nao-voi-ai-196260520123533512.htm",
    "https://nld.com.vn/ca-si-miu-le-bi-khoi-to-tam-giam-ve-toi-to-chuc-su-dung-trai-phep-chat-ma-tuy-196260516215034895.htm",
    "https://nld.com.vn/nguoi-mau-an-tay-em-da-tu-huy-hoai-tuong-lai-cua-minh-196241115133651159.htm",
    "https://tuoitre.vn/bat-tam-giam-rapper-mr-nhan-lien-quan-chuyen-an-pha-duong-day-ma-tuy-lien-tinh-2026052819173896.htm",
    "https://tuoitre.vn/soc-ngoi-sao-bong-chuyen-bi-bat-vi-tang-tru-ma-tuy-khi-len-tuyen-20260528133032596.htm"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    # Import inside function to keep the module importable even if crawl4ai is
    # not installed — callers that don't actually crawl won't pay the import cost.
    # BrowserConfig and CrawlerRunConfig are exported from the top-level package
    # in all crawl4ai versions >= 0.3; importing from async_configs is fragile.
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    # Use a realistic user-agent so Vietnamese news sites don't block headless requests.
    browser_cfg = BrowserConfig(
        headless=True,
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )

    # word_count_threshold=10 filters out navigation/footer fragments that are
    # shorter than a real sentence, keeping only meaningful body paragraphs.
    run_cfg = CrawlerRunConfig(word_count_threshold=10)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

        if not result.success:
            # Raise so crawl_all() can log the failure and skip rather than
            # silently saving an empty file that would corrupt the dataset.
            raise RuntimeError(f"Crawl failed for {url}: {result.error_message}")

        # result.metadata is populated from <meta> tags by crawl4ai; fall back
        # to the URL slug when the page has no <title>.
        title = (result.metadata or {}).get("title") or url.split("/")[-1]

        return {
            "url": url,
            "title": title,
            # ISO-8601 timestamp lets downstream code parse dates unambiguously.
            "date_crawled": datetime.now().isoformat(),
            # Markdown preserves structure (headings, bold) better than raw HTML
            # and is easier to feed into an LLM pipeline later.
            "content_markdown": result.markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    success_count = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            # Log and continue — a single broken URL should not abort the whole
            # batch; the task requires at least 5 successfully saved articles.
            print(f"  ✗ Failed: {exc}")
            continue

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")
        success_count += 1

        # Polite delay between requests to avoid hammering the same origin
        # (nld.com.vn / tuoitre.vn) and getting rate-limited or IP-banned.
        await asyncio.sleep(1)

    print(f"\nDone: {success_count}/{len(ARTICLE_URLS)} articles saved to {DATA_DIR}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
