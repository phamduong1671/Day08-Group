"""
Task 10 — Generation Có Citation.

Pipeline: retrieve → reorder (tránh lost-in-the-middle) → format context có source
→ inject vào prompt → gọi LLM (Ollama/OpenAI-compatible) → trả answer có citation.

Tham số sinh:
    - TEMPERATURE=0.3: RAG cần factual, ít sáng tạo.
    - TOP_P=0.9: đủ tự nhiên, không quá ngẫu nhiên.
    - TOP_K (chunks)=5: đủ evidence mà không loãng / không lost-in-the-middle.
"""

from __future__ import annotations

from .config import LLM_MODEL, get_llm_client
from .task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

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
# DOCUMENT REORDERING (tránh lost in the middle — Liu et al. 2023)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    LLM chú ý tốt nhất ở ĐẦU và CUỐI prompt. Đặt chunk quan trọng ở hai biên,
    kém quan trọng vào giữa.

    Input (sorted desc):  [0, 1, 2, 3, 4]
    Output:               [0, 2, 4, 3, 1]  (best ở đầu, second-best ở cuối)
    Không trùng lặp, giữ nguyên số lượng.
    """
    if len(chunks) <= 2:
        return list(chunks)
    firsts = chunks[0::2]        # 0, 2, 4 — ở nửa đầu
    lasts = chunks[1::2][::-1]   # 3, 1   — ở nửa cuối (đảo để best-odd về cuối)
    return firsts + lasts


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context có nhãn source để LLM cite được."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        md = chunk.get("metadata", {})
        source = md.get("source") or md.get("title") or f"Source {i}"
        doc_type = md.get("type", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n{chunk['content']}\n"
        )
    return "\n---\n".join(parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]
    for q in test_queries:
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
