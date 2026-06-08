"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(PROJECT_DIR / ".env")

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context.
# Chọn 5 vì mỗi chunk ở Task 4 dài khoảng 900 ký tự; 5 chunks thường đủ phủ
# legal + news evidence mà vẫn giữ prompt gọn, giảm nguy cơ lost in the middle.
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích lũy cho token generation.
# Chọn 0.85 vì RAG cần câu trả lời bám nguồn, nhưng vẫn cho model đủ linh hoạt
# để diễn đạt tiếng Việt tự nhiên khi nhiều chunks nói cùng một ý.
TOP_P = 0.85

# temperature: Độ ngẫu nhiên của output
# Chọn 0.2 vì generation có citation cần factual, hạn chế suy đoán.
TEMPERATURE = 0.2

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_OUTPUT_TOKENS = 700
INSUFFICIENT_EVIDENCE_ANSWER = "I cannot verify this information"
CITATION_RE = re.compile(r"\[[^\[\]\n]+,\s*[^\[\]\n]+\]")

KNOWN_SOURCE_NAMES = {
    "bo-luat-hinh-su-2017": "Bộ luật Hình sự",
    "luat-phong-chong-ma-tuy-2021": "Luật Phòng chống ma túy",
    "nghi-dinh-105-2021": "Nghị định 105",
    "thong-tu-danh-muc-ma-tuy": "Thông tư danh mục ma túy",
}

DOMAIN_SOURCE_NAMES = {
    "vnexpress.net": "VnExpress",
    "tuoitre.vn": "Tuổi Trẻ",
    "thanhnien.vn": "Thanh Niên",
    "dantri.com.vn": "Dân Trí",
    "zingnews.vn": "Zing News",
    "vietnamnet.vn": "VietnamNet",
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
using the source label and year from the provided context
(e.g., [Luật Phòng chống ma túy, 2021] or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state exactly 'I cannot verify this information' rather than guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, return exactly: I cannot verify this information
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    # Chunks đầu vào đã được rerank theo score giảm dần. Pattern này giữ chunk
    # quan trọng nhất ở đầu, chunk quan trọng thứ hai ở cuối, các chunk còn lại
    # dàn vào giữa: [1, 3, 5, 4, 2] cho 5 chunks.
    front = list(chunks[::2])
    back = list(reversed(chunks[1::2]))
    return front + back


def _strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "la", "va", "cua", "co", "cho", "the", "theo", "trong", "ve", "voi",
        "nhung", "cac", "mot", "duoc", "bi", "da", "nao", "gi", "hoi",
    }
    text = _strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+", text)
    return {token for token in tokens if len(token) >= 2 and token not in stopwords}


def _normalise_path(path_value: str) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path if path.exists() else None


def _prettify_source_name(source: str) -> str:
    stem = Path(source).stem
    if stem in KNOWN_SOURCE_NAMES:
        return KNOWN_SOURCE_NAMES[stem]

    cleaned = re.sub(r"[-_]+", " ", stem).strip()
    cleaned = re.sub(r"\b20\d{2}\b|\b19\d{2}\b", "", cleaned).strip()
    return cleaned.title() or "Nguồn không rõ"


def _extract_year(*values: str) -> str:
    for value in values:
        if not value:
            continue
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return match.group(0)
    return "Không rõ năm"


def _source_from_url(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    for key, label in DOMAIN_SOURCE_NAMES.items():
        if key in domain:
            return label
    if domain:
        return domain.split(".")[0].title()
    return ""


@lru_cache(maxsize=256)
def _read_markdown_header(path_text: str) -> dict[str, str]:
    path = _normalise_path(path_text)
    if not path:
        return {}

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    source_match = re.search(r"^\*\*Source:\*\*\s*(.+)$", text, flags=re.MULTILINE)
    published_match = re.search(r"^\*\*Published:\*\*\s*(.+)$", text, flags=re.MULTILINE)

    return {
        "title": title_match.group(1).strip() if title_match else "",
        "url": source_match.group(1).strip() if source_match else "",
        "published": published_match.group(1).strip() if published_match else "",
        "text_head": text[:1200],
    }


def _citation_parts(chunk: dict) -> tuple[str, str]:
    metadata = chunk.get("metadata") or {}
    source = (
        metadata.get("source")
        or metadata.get("source_path")
        or metadata.get("filename")
        or f"Source {metadata.get('chunk_index', '')}".strip()
    )
    source_path = metadata.get("source_path") or source
    header = _read_markdown_header(str(source_path))
    content = str(chunk.get("content", ""))

    url_source = _source_from_url(header.get("url", ""))
    if metadata.get("type") == "news" and url_source:
        source_name = url_source
    else:
        source_name = _prettify_source_name(str(source))

    year = _extract_year(
        str(metadata.get("year", "")),
        header.get("published", ""),
        header.get("text_head", ""),
        str(source),
        content[:500],
    )
    return source_name, year


def citation_label(chunk: dict) -> str:
    source_name, year = _citation_parts(chunk)
    return f"[{source_name}, {year}]"


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = (
            metadata.get("source")
            or metadata.get("source_path")
            or metadata.get("filename")
            or f"Source {i}"
        )
        source_name, year = _citation_parts(chunk)
        doc_type = metadata.get("type", "unknown")
        retrieval_source = chunk.get("source", "unknown")
        score = chunk.get("score", 0.0)
        content = str(chunk.get("content", "")).strip()

        context_parts.append(
            f"[Document {i}]\n"
            f"Citation label: [{source_name}, {year}]\n"
            f"Source: {source}\n"
            f"Year: {year}\n"
            f"Type: {doc_type}\n"
            f"Retriever: {retrieval_source}\n"
            f"Score: {float(score):.4f}\n"
            f"Content:\n{content}\n"
        )

    return "\n---\n".join(context_parts)


def _get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip("\"'")
    placeholders = {
        "sk-xxx",
        "your-api-key",
        "your_api_key",
        "YOUR_API_KEY",
        "api_key_cua_ban",
    }
    if not api_key or api_key in placeholders or api_key.endswith("xxx"):
        return ""
    return api_key


def _build_user_message(query: str, context: str) -> str:
    return f"""Context:
{context}

---

Question: {query}

Answer in Vietnamese. Use only the context above. Every factual claim must use
one of the provided Citation labels. If evidence is insufficient, return exactly:
{INSUFFICIENT_EVIDENCE_ANSWER}"""


def _call_openai_llm(query: str, context: str) -> str:
    api_key = _get_openai_api_key()
    if not api_key:
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=30)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, context)},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"! Không gọi được OpenAI; dùng local fallback ({type(exc).__name__})")
        return ""


def _has_citation(answer: str) -> bool:
    return bool(CITATION_RE.search(answer))


def _has_evidence(query: str, chunks: list[dict]) -> bool:
    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return False

    for chunk in chunks:
        content_tokens = _tokenize(str(chunk.get("content", "")))
        if query_tokens & content_tokens:
            return True

    return False


def _split_sentences(content: str) -> list[str]:
    content = re.sub(r"\s+", " ", content).strip()
    if not content:
        return []

    # Legal docs often contain numbered clauses split by newlines/semicolons,
    # so we split on strong punctuation first and then keep long fragments.
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", content)
    sentences = []
    for part in parts:
        sentence = part.strip(" -•\t")
        if len(sentence) >= 30:
            sentences.append(sentence)
    return sentences or [content]


def _truncate(text: str, max_chars: int = 430) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0].rstrip() + "..."


def _best_snippet_with_score(query: str, chunk: dict) -> tuple[str, float]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return "", 0.0

    best_sentence = ""
    best_score = 0.0
    for sentence in _split_sentences(str(chunk.get("content", ""))):
        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            continue
        overlap = len(query_tokens & sentence_tokens) / len(query_tokens)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence

    if best_score <= 0:
        return "", 0.0

    chunk_score = 0.0
    try:
        chunk_score = float(chunk.get("score", 0.0))
    except (TypeError, ValueError):
        pass
    return _truncate(best_sentence), best_score + 0.03 * chunk_score


def _best_snippet(query: str, chunk: dict) -> str:
    return _best_snippet_with_score(query, chunk)[0]


def _local_generate_with_citations(query: str, chunks: list[dict]) -> str:
    if not _has_evidence(query, chunks):
        return INSUFFICIENT_EVIDENCE_ANSWER

    candidates: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        snippet, relevance = _best_snippet_with_score(query, chunk)
        if not snippet or snippet in seen:
            continue

        seen.add(snippet)
        candidates.append((relevance, snippet, citation_label(chunk)))

    if not candidates:
        return INSUFFICIENT_EVIDENCE_ANSWER

    candidates.sort(key=lambda item: item[0], reverse=True)
    lines = [f"- {snippet} {citation}" for _score, snippet, citation in candidates[:4]]
    return "\n".join(lines)


def _retrieval_source(chunks: list[dict]) -> str:
    sources = sorted({str(chunk.get("source", "unknown")) for chunk in chunks if chunk})
    if not sources:
        return "none"
    return sources[0] if len(sources) == 1 else "+".join(sources)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks hoặc dùng context_chunks truyền sẵn
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM; nếu thiếu API key/lỗi mạng thì dùng local fallback
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        context_chunks: Optional chunks đã rerank từ Task 9
        top_k: Số chunks retrieve nếu context_chunks chưa được truyền

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if isinstance(context_chunks, int):
        # Backward compatibility nếu gọi generate_with_citation(query, 3).
        top_k = context_chunks
        context_chunks = None

    if top_k <= 0 or not query.strip():
        return {
            "answer": INSUFFICIENT_EVIDENCE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
            "model": "none",
        }

    # Step 1: Retrieve hoặc dùng chunks đã có.
    chunks = list(context_chunks) if context_chunks is not None else retrieve(query, top_k=top_k)

    # Step 2: Reorder để tránh lost in the middle.
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với source/citation metadata.
    context = format_context(reordered)

    # Step 4-5: Gọi LLM nếu có OpenAI API key hợp lệ.
    if not _has_evidence(query, reordered):
        answer = INSUFFICIENT_EVIDENCE_ANSWER
        model_used = "none"
    else:
        llm_answer = _call_openai_llm(query, context)
        if llm_answer and (
            llm_answer == INSUFFICIENT_EVIDENCE_ANSWER or _has_citation(llm_answer)
        ):
            answer = llm_answer
            model_used = LLM_MODEL
        else:
            answer = _local_generate_with_citations(query, reordered)
            model_used = "local_extractive_fallback"

    # Step 6: Return answer + sources để các task/evaluation sau dùng tiếp.
    return {
        "answer": answer,
        "sources": chunks,
        "reordered_sources": reordered,
        "context": context,
        "retrieval_source": _retrieval_source(chunks),
        "model": model_used,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
