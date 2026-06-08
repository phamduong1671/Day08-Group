"""
Task 10 — Generation Có Citation.

Pipeline: retrieve -> reorder (tránh lost-in-the-middle) -> format context có
source/citation -> inject vào prompt -> gọi LLM -> trả answer có citation.

Nếu LLM local/Ollama/OpenAI chưa sẵn sàng, module fallback sang câu trả lời
extractive từ context để demo backend vẫn có citation và source documents.
"""

from __future__ import annotations

import os
import re

try:
    from .config import LLM_MODEL, get_llm_client
except Exception:
    LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    get_llm_client = None

from .task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)

SOURCE_LABELS = {
    "bo-luat-hinh-su-2017": "Bộ luật Hình sự 2015 sửa đổi 2017",
    "luat-phong-chong-ma-tuy-2021": "Luật Phòng, chống ma túy 2021",
    "nghi-dinh-105-2021": "Nghị định 105/2021/NĐ-CP",
    "thong-tu-danh-muc-ma-tuy": "Thông tư danh mục chất ma túy",
    "luat-phong-chong-ma-tuy-2021.pdf": "Luật Phòng, chống ma túy 2021",
}

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
    Output:               [0, 2, 4, 3, 1]
    """
    if len(chunks) <= 2:
        return list(chunks)
    firsts = list(chunks[0::2])
    lasts = list(chunks[1::2][::-1])
    return firsts + lasts


def _source_stem(source: str) -> str:
    return source.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def citation_label(chunk: dict, fallback_index: int = 1) -> str:
    """Tạo nhãn citation ngắn, ổn định cho source document."""
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source") or metadata.get("source_path") or f"Source {fallback_index}"
    source = str(source)
    stem = _source_stem(source)

    if source in SOURCE_LABELS:
        return SOURCE_LABELS[source]
    if stem in SOURCE_LABELS:
        return SOURCE_LABELS[stem]

    title = metadata.get("title")
    if title:
        return str(title)

    content = chunk.get("content", "")
    title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()

    return source


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context có nhãn source để LLM cite được."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source") or metadata.get("title") or f"Source {i}"
        source_path = metadata.get("source_path", "")
        doc_type = metadata.get("type", "unknown")
        citation = citation_label(chunk, fallback_index=i)
        score = float(chunk.get("score", 0.0))
        parts.append(
            f"[Document {i} | Citation: {citation} | Source: {source} | "
            f"Type: {doc_type} | Score: {score:.3f}]\n"
            f"Path: {source_path}\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(parts)


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
        if not snippet or (label in used_labels and len(lines) >= 2):
            continue

        lines.append(f"- {snippet} [{label}]")
        used_labels.add(label)
        if len(lines) >= 3:
            break

    if not lines:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return "Dựa trên các tài liệu tìm được:\n" + "\n".join(lines)


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


def _call_llm(system_prompt: str, user_message: str) -> tuple[str, str]:
    """
    Gọi LLM qua src.config. Nếu không được, thử OpenAI mặc định rồi trả rỗng
    để caller dùng extractive fallback.
    """
    try:
        if get_llm_client is None:
            raise RuntimeError("get_llm_client unavailable")
        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or "", LLM_MODEL
    except Exception as exc:
        print(f"! Không gọi được configured LLM; thử OpenAI/default fallback ({type(exc).__name__})")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    placeholders = {"", "your-api-key", "your_api_key", "OPENAI_API_KEY", "api_key_cua_ban", "ollama"}
    if api_key in placeholders:
        return "", "local_context_extractive"

    try:
        from openai import OpenAI

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or "", model
    except Exception as exc:
        print(f"! Không gọi được OpenAI; dùng local generation fallback ({type(exc).__name__})")
        return "", "local_context_extractive"


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    use_reranking: bool = True,
    score_threshold: float | None = None,
    conversation_history: list[dict] | None = None,
    retrieval_query: str | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Args:
        use_reranking: bật/tắt cross-encoder rerank ở tầng retrieval (dùng cho A/B eval).
        score_threshold: override ngưỡng fallback ở tầng retrieval.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'reordered_sources': list[dict],
            'retrieval_source': str,
            'retrieval_query': str,
            'generation_backend': str,
        }
    """
    retrieve_kwargs = {"top_k": top_k, "use_reranking": use_reranking}
    if score_threshold is not None:
        retrieve_kwargs["score_threshold"] = score_threshold
    chunks = retrieve(query, **retrieve_kwargs)
    search_query = retrieval_query or query
    chunks = retrieve(search_query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "reordered_sources": [],
            "retrieval_source": "none",
            "retrieval_query": search_query,
            "generation_backend": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    history = _format_conversation_history(conversation_history)
    user_message = f"""Conversation history:
{history if history else '(none)'}

Context:
{context}

---

Question: {query}"""

    answer, generation_backend = _call_llm(SYSTEM_PROMPT, user_message)
    if not answer.strip():
        answer = _fallback_generate_answer(query, reordered)
        generation_backend = "local_context_extractive"

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
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
