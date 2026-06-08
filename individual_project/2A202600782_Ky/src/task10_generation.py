"""
Task 10 — Generation Có Citation.

==============================================================================
THIẾT KẾ — sinh câu trả lời grounded, citation phân biệt theo LOẠI văn bản.
==============================================================================
  1. retrieve (Task 9) -> chunks đã rerank.
  2. reorder_for_llm: chống "lost in the middle" — chunk tốt nhất ở ĐẦU & CUỐI.
  3. format_context: gắn nhãn nguồn để LLM trích dẫn — LUẬT cite theo Điều
     ([<văn bản>, Điều N]), BÁO cite theo nguồn ([<tiêu đề/nguồn>]).
  4. LLM sinh trả lời, bắt buộc citation; thiếu bằng chứng -> nói rõ không xác minh.

Tham số sinh (giải thích): RAG cần factual -> temperature thấp (0.3) giảm bịa;
top_p=0.9 đủ tự nhiên; top_k context=5 đủ bằng chứng mà không loãng/lost-in-middle.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
TOP_K = 5            # đủ bằng chứng, tránh context loãng gây lost-in-the-middle
TOP_P = 0.9          # nucleus sampling: tự nhiên nhưng không quá ngẫu nhiên
TEMPERATURE = 0.3    # RAG cần factual -> ít sáng tạo
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

INSUFFICIENT = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý. Trả lời câu hỏi bằng tiếng Việt, đầy đủ.
Với MỌI câu hoặc bullet có khẳng định/sự kiện, chèn citation ở cuối chính câu
hoặc bullet đó, ví dụ [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VnExpress].

Nếu thông tin KHÔNG có trong context được cung cấp, hãy nói rõ
'Tôi không thể xác minh thông tin này từ nguồn hiện có' thay vì đoán.

Quy tắc:
- Chỉ dùng thông tin trong context.
- Mỗi câu/bullet factual PHẢI có ít nhất một citation dạng [nguồn].
- Chỉ dùng đúng nhãn citation được cung cấp sau dòng `Citation:`.
- Context không đủ -> nói rõ.
- Trình bày mạch lạc theo đoạn."""


# =============================================================================
# DOCUMENT REORDERING (chống lost-in-the-middle)
# =============================================================================
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đặt chunk quan trọng ở ĐẦU và CUỐI, kém quan trọng vào GIỮA.

    Input (sorted desc):  [0, 1, 2, 3, 4]
    Output:               [0, 2, 4, 3, 1]  (best đứng đầu, second-best đứng cuối)
    """
    if len(chunks) <= 2:
        return list(chunks)
    evens = chunks[0::2]        # 0, 2, 4, ... (gồm chunk tốt nhất ở đầu)
    odds = chunks[1::2]         # 1, 3, 5, ...
    return evens + odds[::-1]   # ...4, rồi 3, 1 -> second-best về cuối


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================
def _citation_label(meta: dict) -> str:
    """Nhãn citation theo loại: luật -> Điều; báo -> tiêu đề/nguồn."""
    title = meta.get("doc_title") or meta.get("source", "")
    doc_type = meta.get("doc_type") or meta.get("type")
    if doc_type == "legal" and meta.get("dieu"):
        return f"{title}, Điều {meta['dieu']}"
    return title


def format_context(chunks: list[dict]) -> str:
    """Ghép chunks thành context có nhãn nguồn để LLM trích dẫn chính xác."""
    parts = []
    for i, ch in enumerate(chunks, 1):
        meta = ch.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        label = _citation_label(meta)
        parts.append(
            f"[Tài liệu {i}]\nSource: {source}\nCitation: [{label}]\n{ch['content']}\n"
        )
    return "\n---\n".join(parts)


def _needs_more_context(query: str) -> bool:
    q = query.lower()
    return any(token in q for token in ("những", "nào", "ai", "liệt kê", "danh sách"))


def _has_sufficient_evidence(chunks: list[dict]) -> bool:
    if not chunks:
        return False
    top = chunks[0].get("score", 0.0)
    source = chunks[0].get("source")
    if source == "pageindex":
        return top > 0
    return top >= 0.25


def _validate_citations(answer: str) -> bool:
    if INSUFFICIENT in answer:
        return True
    factual_lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip() and len(line.strip()) > 25
    ]
    if not factual_lines:
        return False
    return all("[" in line and "]" in line for line in factual_lines)


def _build_extractive_fallback(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return INSUFFICIENT
    lines = []
    for chunk in chunks[:5]:
        meta = chunk.get("metadata", {})
        label = _citation_label(meta)
        snippet = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        lines.append(f"- {snippet} [{label}]")
    return "\n".join(lines) if lines else INSUFFICIENT


# =============================================================================
# GENERATION
# =============================================================================
def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """RAG end-to-end có citation.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    retrieval_k = max(top_k, 8) if _needs_more_context(query) else top_k
    chunks = retrieve(query, top_k=retrieval_k)

    # Không có bằng chứng -> trả lời không xác minh, không gọi LLM.
    if not _has_sufficient_evidence(chunks):
        return {"answer": INSUFFICIENT, "sources": [], "retrieval_source": "none"}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content
    if not _validate_citations(answer):
        retry_messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content":
             "Hãy viết lại. Mỗi câu hoặc bullet có thông tin factual phải kết thúc "
             "bằng citation dạng [nguồn]. Chỉ dùng nhãn sau dòng Citation trong context."},
        ]
        retry = client.chat.completions.create(
            model=LLM_MODEL,
            messages=retry_messages,
            temperature=0.0,
            top_p=TOP_P,
        )
        answer = retry.choices[0].message.content
    if not _validate_citations(answer):
        answer = _build_extractive_fallback(query, chunks)

    return {
        "answer": answer,
        "sources": chunks[:retrieval_k],
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]
    for q in test_queries:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
