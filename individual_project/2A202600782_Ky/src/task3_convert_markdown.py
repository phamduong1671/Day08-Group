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
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from markitdown import MarkItDown

try:
    from .task2_crawl_news import clean_article_content, resolve_title
except ImportError:  # Khi chạy trực tiếp: python src/task3_convert_markdown.py
    from task2_crawl_news import clean_article_content, resolve_title

PROJECT_DIR = Path(__file__).parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"

# tessdata local (chứa vie.traineddata) — không cần cài tesseract-data-vie hệ thống.
TESSDATA_DIR = PROJECT_DIR / "tessdata"

# Nếu markitdown trích được ít hơn ngưỡng này -> coi như PDF scan, dùng OCR.
OCR_TEXT_THRESHOLD = 200


def doc_to_docx(filepath: Path) -> Path:
    """Convert .doc cũ (OLE2 binary) sang .docx bằng LibreOffice headless.

    MarkItDown chỉ đọc được .docx, không đọc .doc cũ. Các nghị định tải từ
    thuvienphapluat thường là .doc nên cần bước này. Trả về đường dẫn .docx tạm.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError(
            "Cần LibreOffice (soffice) để convert .doc -> .docx. "
            "Cài: sudo pacman -S libreoffice-still (hoặc bản distro tương ứng)."
        )
    tmp_dir = Path(tempfile.mkdtemp(prefix="doc2docx_"))
    subprocess.run(
        [soffice, "--headless", "--convert-to", "docx", "--outdir", str(tmp_dir), str(filepath)],
        check=True,
        capture_output=True,
    )
    out = tmp_dir / f"{filepath.stem}.docx"
    if not out.exists():
        raise RuntimeError(f"Convert .doc thất bại: {filepath.name}")
    return out


def ocr_pdf(filepath: Path, dpi: int = 200, lang: str = "vie") -> str:
    """OCR một PDF scan (ảnh) sang text bằng Tesseract tiếng Việt.

    Dùng cho các nghị định scan không có text layer (vd nd105, nd57, nd282).
    """
    import pytesseract
    from pdf2image import convert_from_path

    if TESSDATA_DIR.exists():
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)

    pages = convert_from_path(str(filepath), dpi=dpi)
    parts = []
    for i, img in enumerate(pages, 1):
        text = pytesseract.image_to_string(img, lang=lang)
        parts.append(f"<!-- page {i} -->\n{text.strip()}")
    return "\n\n".join(parts)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")

            # .doc cũ -> convert sang .docx tạm trước khi đưa vào MarkItDown.
            convert_target = filepath
            method = "markitdown"
            if filepath.suffix.lower() == ".doc":
                print("  → .doc cũ, convert sang .docx bằng LibreOffice...")
                convert_target = doc_to_docx(filepath)
                method = "libreoffice-docx+markitdown"

            result = md.convert(str(convert_target))
            text = result.text_content

            # PDF scan -> markitdown ra rỗng -> fallback OCR tiếng Việt.
            if filepath.suffix.lower() == ".pdf" and len(text.strip()) < OCR_TEXT_THRESHOLD:
                print("  ⚠ Ít text (PDF scan?) → chạy OCR tiếng Việt...")
                text = ocr_pdf(filepath)
                method = "ocr-tesseract-vie"

            output_path = output_dir / f"{filepath.stem}.md"
            header = (
                f"# {filepath.stem}\n\n"
                f"**Source file:** {filepath.name}\n\n"
                f"**Convert method:** {method}\n\n---\n\n"
            )
            output_path.write_text(header + text, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(text)} chars, {method})")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm metadata header để giữ nguồn cho citation ở Task 10.
            url = data.get("url", "N/A")
            raw_content = data.get("content_markdown", "")
            title = resolve_title(data.get("title"), url, raw_content)
            content_markdown = clean_article_content(raw_content)

            header = f"# {title}\n\n"
            header += f"**Source:** {url}\n\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + content_markdown
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(content)} chars)")


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
