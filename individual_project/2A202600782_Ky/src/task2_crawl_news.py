"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Đọc danh sách URL từ data/landing/news/link.txt
    2. Khử trùng lặp (bỏ query tracking như ?utm_source=...)
    3. Crawl bằng Crawl4AI (Playwright headless) -> lấy markdown + title
    4. Lưu mỗi bài thành 1 file JSON kèm metadata:
       url, title, date_crawled, content_markdown

Chạy:
    python src/task2_crawl_news.py
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
LINK_FILE = DATA_DIR / "link.txt"

TITLE_OVERRIDES = {
    "nghe-si-viet-keu-goi-gioi-tre-khong-thu-ma-tuy-du-chi-mot-lan":
        "Nghệ sĩ Việt kêu gọi giới trẻ không thử ma túy dù chỉ một lần",
    "ca-si-long-nhat-bi-bat-showbiz-viet-lien-tiep-chan-dong-vi-ma-tuy":
        "Ca sĩ Long Nhật bị bắt: Showbiz Việt liên tiếp chấn động vì ma túy",
    "chi-dan-huu-tin-va-loat-sao-viet-gay-on-ao-vi-dinh-toi-ma-tuy":
        "Chi Dân, Hữu Tín và loạt sao Việt gây ồn ào vì dính tới ma túy",
    "nghe-si-dinh-ma-tuy-can-mot-lan-ranh-do":
        "Nghệ sĩ dính ma túy cần một lằn ranh đỏ",
    "ca-si-son-ngoc-minh-vua-bi-bat-vi-lien-quan-den-ma-tuy-la-ai":
        "Ca sĩ Sơn Ngọc Minh vừa bị bắt vì liên quan đến ma túy là ai?",
    "ntk-nguyen-cong-tri-bi-bat-vi-ma-tuy-dung-khoa-lap-cho-sai-pham-bang-tai-nang":
        "NTK Nguyễn Công Trí bị bắt vì ma túy: Đừng khỏa lấp sai phạm bằng tài năng",
}


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url: str) -> str:
    """Bỏ query string (vd ?utm_source=chatgpt.com) để khử trùng lặp."""
    parsed = urlparse(url.strip())
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def load_urls() -> list[str]:
    """Đọc URL từ link.txt, normalize và khử trùng lặp (giữ thứ tự)."""
    if not LINK_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy {LINK_FILE}. Hãy tạo file link.txt.")

    seen: set[str] = set()
    urls: list[str] = []
    for line in LINK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        norm = normalize_url(line)
        if norm not in seen:
            seen.add(norm)
            urls.append(norm)
    return urls


def slug_from_url(url: str, max_length: int | None = 80) -> str:
    """Lấy slug từ path URL để đặt tên file dễ đọc."""
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    slug = re.sub(r"\.html?$", "", path)
    slug = re.sub(r"-\d{6,}$", "", slug)  # bỏ đuôi id số dài của báo VN
    slug = re.sub(r"[^a-zA-Z0-9-]", "", slug)
    if max_length is not None:
        slug = slug[:max_length]
    return slug or "article"


def is_unknown_title(title: str | None) -> bool:
    """Crawl4AI đôi khi trả title rỗng/Unknown khi selector chỉ lấy thân bài."""
    if not title:
        return True
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    return normalized in {"unknown", "n/a", "none", "null", "untitled"}


def title_from_slug(url: str) -> str:
    """Fallback deterministic từ URL slug khi site metadata không có title."""
    slug = slug_from_url(url, max_length=None)
    if slug in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[slug]
    words = slug.replace("-", " ").strip()
    return words[:1].upper() + words[1:] if words else "Unknown"


def resolve_title(raw_title: str | None, url: str, content: str = "") -> str:
    """Chọn title tốt nhất: metadata -> curated URL title -> content fallback."""
    slug = slug_from_url(url, max_length=None)
    if slug in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[slug]

    if not is_unknown_title(raw_title):
        return re.sub(r"\s+", " ", raw_title).strip()

    slug_title = title_from_slug(url)
    if not is_unknown_title(slug_title):
        return slug_title

    for line in content.splitlines():
        line = re.sub(r"^#+\s*", "", line).strip()
        if 20 <= len(line) <= 180 and not line.lower().startswith(("ảnh:", "source:")):
            if line.endswith("?") or any(token in line.lower() for token in ("ma túy", "ma tuý", "bị bắt")):
                return re.sub(r"\s+", " ", line).strip()

    return "Unknown"


def _drop_repeated_tail(lines: list[str]) -> list[str]:
    """Remove exact article duplication where crawler returns body twice."""
    compact = [line.strip() for line in lines if line.strip()]
    if len(compact) < 8:
        return lines

    for start in range(1, len(compact)):
        first = compact[:start]
        tail = compact[start: start + len(first)]
        if tail == first and start + len(first) >= len(compact):
            return first + compact[start + len(first):]
    return lines


def clean_article_content(content: str) -> str:
    """Normalize crawled markdown and remove duplicated long paragraphs."""
    lines = _drop_repeated_tail(content.splitlines())
    out: list[str] = []
    seen_long_lines: set[str] = set()

    for line in lines:
        stripped = re.sub(r"[ \t]+", " ", line).strip()
        if not stripped:
            if out and out[-1]:
                out.append("")
            continue

        key = re.sub(r"\s+", " ", stripped).casefold()
        if len(key) >= 80:
            if key in seen_long_lines:
                continue
            seen_long_lines.add(key)

        out.append(stripped)

    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


async def crawl_article(crawler, run_cfg, url: str) -> dict:
    """Crawl một bài báo và trả về dict chứa metadata + content.

    Ưu tiên fit_markdown (đã lọc boilerplate qua PruningContentFilter +
    CSS selector vào thân bài) để giảm nhiễu nav/menu cho RAG downstream.
    """
    result = await crawler.arun(url=url, config=run_cfg)
    if not result.success:
        raise RuntimeError(f"Crawl thất bại: {url} — {result.error_message}")

    fit = getattr(result.markdown, "fit_markdown", "") or ""
    raw = getattr(result.markdown, "raw_markdown", "") or str(result.markdown)
    content = clean_article_content(fit if len(fit) > 300 else raw)

    metadata = result.metadata or {}
    raw_title = metadata.get("title") or metadata.get("og:title")
    title = resolve_title(raw_title, url, content)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": content,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong link.txt."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    setup_directory()
    urls = load_urls()
    print(f"Tìm thấy {len(urls)} URL (sau khi khử trùng lặp).")

    # CSS selector nhắm vào thân bài của thanhnien.vn + lọc boilerplate.
    md_gen = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed")
    )
    run_cfg = CrawlerRunConfig(
        markdown_generator=md_gen,
        css_selector="div.detail-content, div[data-role=content], .detail__content, article",
        word_count_threshold=10,
    )

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    saved = 0
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Crawling: {url}")
            try:
                article = await crawl_article(crawler, run_cfg, url)
            except Exception as e:
                print(f"  ✗ Lỗi: {e}")
                continue

            filename = f"article_{i:02d}_{slug_from_url(url)}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved += 1
            print(f"  ✓ Saved: {filepath.name} ({len(article['content_markdown'])} chars)")

    print(f"\n✓ Hoàn tất: {saved}/{len(urls)} bài đã lưu vào {DATA_DIR}")
    if saved < 5:
        print("⚠ Chưa đủ 5 bài — kiểm tra lỗi crawl phía trên.")


if __name__ == "__main__":
    asyncio.run(crawl_all())
