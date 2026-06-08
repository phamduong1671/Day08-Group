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
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)

SOURCE_LABELS = {
    "bo-luat-hinh-su-2017": "Bộ luật Hình sự 2015 sửa đổi 2017",
    "luat-phong-chong-ma-tuy-2021": "Luật Phòng, chống ma túy 2021",
    "nghi-dinh-105-2021": "Nghị định 105/2021/NĐ-CP",
    "thong-tu-danh-muc-ma-tuy": "Thông tư danh mục chất ma túy",
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
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

    front = list(chunks[::2])
    back = list(chunks[1::2])
    back.reverse()
    return front + back


def _source_stem(source: str) -> str:
    return source.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def citation_label(chunk: dict, fallback_index: int = 1) -> str:
    """Tạo nhãn citation ngắn, ổn định cho source document."""
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source") or metadata.get("source_path") or f"Source {fallback_index}"
    stem = _source_stem(str(source))

    if stem in SOURCE_LABELS:
        return SOURCE_LABELS[stem]

    content = chunk.get("content", "")
    title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()

    return str(source)


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
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source", f"Source {i}")
        source_path = metadata.get("source_path", "")
        doc_type = metadata.get("type", "unknown")
        citation = citation_label(chunk, fallback_index=i)
        score = float(chunk.get("score", 0.0))

        context_parts.append(
            f"[Document {i} | Citation: {citation} | Source: {source} | "
            f"Type: {doc_type} | Score: {score:.3f}]\n"
            f"Path: {source_path}\n"
            f"{chunk.get('content', '')}\n"
        )

    return "\n---\n".join(context_parts)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 2}


def _split_candidate_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: list[str] = []

    for line in lines:
        if line.startswith(("Source:", "Published:", "Crawled:", "Crawler:", "---")):
            continue
        if line.startswith("#"):
            continue

        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            part = part.strip(" -")
            if len(part) >= 40:
                candidates.append(part)

    return candidates


def _best_snippet(query: str, chunk: dict, max_chars: int = 360) -> str:
    query_tokens = _tokenize(query)
    candidates = _split_candidate_sentences(chunk.get("content", ""))
    if not candidates:
        text = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
        return text[:max_chars].rstrip()

    def score(sentence: str) -> tuple[int, int]:
        sentence_tokens = _tokenize(sentence)
        return (len(query_tokens & sentence_tokens), min(len(sentence), max_chars))

    best = max(candidates, key=score)
    if len(best) <= max_chars:
        return best
    return best[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def _fallback_generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    lines = []
    used_labels: set[str] = set()
    for index, chunk in enumerate(chunks, 1):
        label = citation_label(chunk, fallback_index=index)
        snippet = _best_snippet(query, chunk)
        if not snippet or label in used_labels and len(lines) >= 2:
            continue

        lines.append(f"- {snippet} [{label}]")
        used_labels.add(label)
        if len(lines) >= 3:
            break

    if not lines:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return "Dựa trên các tài liệu tìm được:\n" + "\n".join(lines)


def _has_openai_key() -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    placeholders = {"", "your-api-key", "your_api_key", "OPENAI_API_KEY", "api_key_cua_ban"}
    return api_key not in placeholders


def _format_conversation_history(conversation_history: list[dict] | None) -> str:
    if not conversation_history:
        return ""

    formatted = []
    for turn in conversation_history[-6:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        if content:
            formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    conversation_history: list[dict] | None = None,
    retrieval_query: str | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    search_query = retrieval_query or query
    chunks = retrieve(search_query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    history = _format_conversation_history(conversation_history)

    user_message = f"""Conversation history:
{history if history else '(none)'}

Context:
{context}

---

Question: {query}"""

    answer = ""
    generation_backend = "local_context_extractive"

    if _has_openai_key():
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
            generation_backend = OPENAI_MODEL
        except Exception as exc:
            print(f"! Không gọi được OpenAI; dùng local generation fallback ({type(exc).__name__})")

    if not answer.strip():
        answer = _fallback_generate_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "reordered_sources": reordered,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "retrieval_query": search_query,
        "generation_backend": generation_backend,
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
