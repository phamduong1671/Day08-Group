"""Task 3 - Convert landing files to Markdown."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

try:
    from markitdown import MarkItDown
except Exception:  # MarkItDown is optional at runtime; DOCX fallback below is enough for tests.
    MarkItDown = None


PROJECT_DIR = Path(__file__).parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _docx_to_text(path: Path) -> str:
    """Small DOCX text extractor used when MarkItDown is unavailable."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n\n".join(paragraphs)


def _convert_document(path: Path) -> str:
    if MarkItDown is not None:
        try:
            result = MarkItDown().convert(str(path))
            content = getattr(result, "text_content", "") or getattr(result, "markdown", "")
            if content and len(content.strip()) > 50:
                return _clean_text(content)
        except Exception as exc:
            print(f"  ! MarkItDown failed for {path.name}: {type(exc).__name__}")

    if path.suffix.lower() == ".docx":
        return _clean_text(_docx_to_text(path))
    return ""


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if not legal_dir.exists():
        return written

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        print(f"Converting legal: {filepath.name}")
        body = _convert_document(filepath)
        if not body:
            body = f"Không thể trích xuất nội dung từ {filepath.name}."
        output_path = output_dir / f"{filepath.stem}.md"
        markdown = (
            f"# {filepath.stem.replace('-', ' ').title()}\n\n"
            f"**Source:** {filepath.name}\n"
            f"**Type:** legal\n\n---\n\n"
            f"{body}\n"
        )
        output_path.write_text(markdown, encoding="utf-8")
        written.append(output_path)
        print(f"  Saved: {output_path}")
    return written


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles in data/landing/news/ to markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if not news_dir.exists():
        return written

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting news: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        title = data.get("title") or filepath.stem.replace("-", " ").title()
        content = data.get("content_markdown") or data.get("markdown") or data.get("content") or ""
        header = (
            f"# {title}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n"
            f"**Domain:** {data.get('source_domain', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
            f"**Type:** news\n\n---\n\n"
        )
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(_clean_text(header + content) + "\n", encoding="utf-8")
        written.append(output_path)
        print(f"  Saved: {output_path}")
    return written


def convert_all() -> list[Path]:
    """Convert all landing files and return written markdown paths."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)
    written = convert_legal_docs() + convert_news_articles()
    print(f"Done. Wrote {len(written)} markdown files to {OUTPUT_DIR}")
    return written


if __name__ == "__main__":
    convert_all()
