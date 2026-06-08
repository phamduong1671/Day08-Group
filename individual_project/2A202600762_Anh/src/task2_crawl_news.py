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
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo cần crawl (tối thiểu 5 bài báo về các nghệ sĩ Việt Nam liên quan tới ma tuý)
ARTICLE_URLS = [
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    "https://tuoitre.vn/vu-ma-tuy-lien-quan-tiep-vien-vietnam-airlines-truy-to-nguoi-mau-an-tay-chi-dan-va-225-bi-can-20260402112720784.htm",
    "https://thanhnien.vn/dien-vien-hai-tran-huu-tin-lanh-7-nam-6-thang-tu-185230428134549434.htm",
    "https://dantri.com.vn/phap-luat/ca-si-chu-bin-bi-tam-giu-vi-lien-quan-den-ma-tuy-20240606183158183.htm",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060855.html"
]


def crawl_fallback(url: str) -> dict:
    """
    Fallback crawling method using requests and BeautifulSoup.
    Extremely robust and does not require headless browser execution.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 1. Extract title
    title = ""
    meta_title = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "title"})
    if meta_title:
        title = meta_title.get("content", "").strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.text.strip()
            
    # 2. Extract publish date
    publish_date = "Unknown"
    meta_date = (
        soup.find("meta", attrs={"property": "article:published_time"}) or 
        soup.find("meta", attrs={"itemprop": "datePublished"}) or
        soup.find("meta", attrs={"name": "pubdate"}) or
        soup.find("meta", attrs={"property": "pubdate"})
    )
    if meta_date:
        publish_date = meta_date.get("content", "").strip()
    else:
        time_tag = soup.find("time")
        if time_tag:
            publish_date = time_tag.text.strip()
        else:
            for selector in [".date", ".date-time", ".time", ".date-time-release", "span.time", ".detail-time", ".author-time"]:
                date_el = soup.select_one(selector)
                if date_el:
                    publish_date = date_el.text.strip()
                    break
                
    # 3. Extract content
    # Remove script, style, header, footer, etc.
    for element in soup(["script", "style", "header", "footer", "nav", "iframe"]):
        element.decompose()
        
    content_div = None
    if "vnexpress.net" in url:
        content_div = soup.select_one("article.fck_detail") or soup.select_one(".sidebar-1")
    elif "tuoitre.vn" in url:
        content_div = soup.select_one(".detail-content") or soup.select_one(".detail-cmain") or soup.select_one("#main-detail-body") or soup.select_one(".content-detail") or soup.select_one(".fck")
    elif "thanhnien.vn" in url:
        content_div = soup.select_one(".detail-content") or soup.select_one(".detail-cmain") or soup.select_one("[itemprop='articleBody']")
    elif "dantri.com.vn" in url:
        content_div = soup.select_one(".singular-content") or soup.select_one(".detail-content")
        
    if not content_div:
        content_div = soup.find("article") or soup.find("main") or soup.find("body")
        
    paragraphs = []
    if content_div:
        p_tags = content_div.find_all("p")
        if p_tags:
            for p in p_tags:
                text = p.text.strip()
                if text:
                    paragraphs.append(text)
        else:
            paragraphs.append(content_div.text.strip())
            
    content_markdown = "\n\n".join(paragraphs)
    
    if not title:
        title = "Unknown Title"
        
    return {
        "url": url,
        "title": title,
        "publish_date": publish_date,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
        "crawler": "requests_bs4"
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "publish_date": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    print(f"  Attempting to crawl with crawl4ai: {url}")
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.success:
                soup = BeautifulSoup(result.html, "html.parser")
                
                # Title
                title = result.metadata.get("title", "") if result.metadata else ""
                if not title:
                    h1 = soup.find("h1")
                    title = h1.text.strip() if h1 else "Unknown Title"
                
                # Publish date
                publish_date = "Unknown"
                meta_date = (
                    soup.find("meta", attrs={"property": "article:published_time"}) or 
                    soup.find("meta", attrs={"itemprop": "datePublished"}) or
                    soup.find("meta", attrs={"name": "pubdate"}) or
                    soup.find("meta", attrs={"property": "pubdate"})
                )
                if meta_date:
                    publish_date = meta_date.get("content", "").strip()
                else:
                    time_tag = soup.find("time")
                    if time_tag:
                        publish_date = time_tag.text.strip()
                    else:
                        for selector in [".date", ".date-time", ".time", ".date-time-release", "span.time", ".detail-time", ".author-time"]:
                            date_el = soup.select_one(selector)
                            if date_el:
                                publish_date = date_el.text.strip()
                                break
                            
                return {
                    "url": url,
                    "title": title,
                    "publish_date": publish_date,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown or "",
                    "crawler": "crawl4ai"
                }
            else:
                print("    -> crawl4ai success is False. Falling back to requests+BeautifulSoup...")
    except Exception as e:
        print(f"    -> crawl4ai error: {e}. Falling back to requests+BeautifulSoup...")
        
    # Fallback to requests + BeautifulSoup
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, crawl_fallback, url)


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())

