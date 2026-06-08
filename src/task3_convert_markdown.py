"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

try:
    from markitdown import MarkItDown
except ImportError:  # pragma: no cover - requirements.txt installs this.
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _normalize_blank_lines(text: str) -> str:
    """Keep converted markdown readable and stable across converter versions."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank

    return "\n".join(normalized).strip() + "\n"


def _format_metadata_value(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


def _read_docx_with_ooxml(filepath: Path) -> str:
    """
    Fallback for environments that installed markitdown without the [docx] extra.

    DOCX files are ZIP archives containing WordprocessingML. This extracts text
    from document paragraphs and table cells without adding another dependency.
    MarkItDown is still the preferred converter when its DOCX plugin is present.
    """
    paragraph_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
    text_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
    tab_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tab"
    break_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"

    with zipfile.ZipFile(filepath) as archive:
        xml_content = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_content)
    paragraphs: list[str] = []

    for paragraph in root.iter(paragraph_tag):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == text_tag and node.text:
                parts.append(node.text)
            elif node.tag == tab_tag:
                parts.append("\t")
            elif node.tag == break_tag:
                parts.append("\n")

        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def _convert_file_with_markitdown(filepath: Path) -> str:
    if MarkItDown is None:
        raise RuntimeError("MarkItDown chưa được cài. Hãy chạy: pip install markitdown")

    result = MarkItDown().convert(str(filepath))
    return result.text_content or ""


def _convert_legal_file(filepath: Path) -> str:
    try:
        return _convert_file_with_markitdown(filepath)
    except Exception as exc:
        if filepath.suffix.lower() == ".docx":
            print(f"  ! MarkItDown DOCX conversion failed, using OOXML fallback ({type(exc).__name__})")
            return _read_docx_with_ooxml(filepath)
        raise


def _article_json_to_markdown(data: dict[str, Any]) -> str:
    title = _format_metadata_value(data.get("title") or data.get("headline") or "Unknown Title")
    url = _format_metadata_value(data.get("url") or data.get("source_url"))
    publish_date = _format_metadata_value(data.get("publish_date") or data.get("published_at"))
    date_crawled = _format_metadata_value(data.get("date_crawled") or data.get("crawled_at"))
    crawler = _format_metadata_value(data.get("crawler"))

    content = (
        data.get("content_markdown")
        or data.get("markdown")
        or data.get("content")
        or data.get("text")
        or ""
    )

    if isinstance(content, list):
        content = "\n\n".join(str(item).strip() for item in content if str(item).strip())
    else:
        content = str(content).strip()

    metadata = [
        f"# {title}",
        "",
        f"**Source:** {url}",
        f"**Published:** {publish_date}",
        f"**Crawled:** {date_crawled}",
        f"**Crawler:** {crawler}",
        "",
        "---",
        "",
    ]
    return "\n".join(metadata) + content


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() in LEGAL_EXTENSIONS:
            print(f"Converting: {filepath.name}")
            markdown = _convert_legal_file(filepath)
            markdown = _normalize_blank_lines(markdown)

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(markdown, encoding="utf-8")
            converted += 1
            print(f"  ✓ Saved: {output_path}")

    if converted == 0:
        print(f"  ! Không tìm thấy PDF/DOC/DOCX trong {legal_dir}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            markdown = _normalize_blank_lines(_article_json_to_markdown(data))

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(markdown, encoding="utf-8")
            converted += 1
            print(f"  ✓ Saved: {output_path}")

    if converted == 0:
        print(f"  ! Không tìm thấy JSON trong {news_dir}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
