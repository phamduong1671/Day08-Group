"""Task 10 - RAG generation with citations."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import tokenize
from .task9_retrieval_pipeline import retrieve

load_dotenv()


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source. If the information is not explicitly stated in
the provided context, say that the information cannot be verified from current
sources rather than guessing."""


def citation_label(chunk: dict, fallback_index: int = 1) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = str(metadata.get("source") or metadata.get("source_path") or f"Source {fallback_index}")
    stem = Path(source).stem
    return stem.replace("-", " ").strip().title() or f"Source {fallback_index}"


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Keep strongest chunk first, then alternate so strong chunks sit near edges."""
    if len(chunks) <= 2:
        return list(chunks)
    return list(chunks[0::2]) + list(chunks[1::2][::-1])


def format_context(chunks: list[dict]) -> str:
    """Format chunks into citation-aware context."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source") or f"Source {index}"
        doc_type = metadata.get("type", "unknown")
        score = float(chunk.get("score", 0.0))
        citation = citation_label(chunk, fallback_index=index)
        parts.append(
            f"[Document {index} | Citation: {citation} | Source: {source} | "
            f"Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(parts)


def _candidate_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip(" -")
        if not line or line.startswith(("#", "**Source", "**Type", "---")):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if len(sentence) >= 35:
                sentences.append(sentence)
    return sentences


def _best_snippet(query: str, text: str, max_chars: int = 360) -> str:
    query_terms = set(tokenize(query))
    candidates = _candidate_sentences(text)
    if not candidates:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:max_chars].rstrip()

    def score(sentence: str) -> tuple[int, int]:
        sentence_terms = set(tokenize(sentence))
        return (len(query_terms & sentence_terms), -abs(len(sentence) - 180))

    best = max(candidates, key=score)
    if len(best) <= max_chars:
        return best
    return best[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    lines: list[str] = []
    used: set[str] = set()
    for index, chunk in enumerate(chunks, 1):
        label = citation_label(chunk, fallback_index=index)
        snippet = _best_snippet(query, chunk.get("content", ""))
        if not snippet or (label in used and len(lines) >= 2):
            continue
        lines.append(f"- {snippet} [{label}]")
        used.add(label)
        if len(lines) >= 3:
            break

    if not lines:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return "Dựa trên các tài liệu tìm được:\n" + "\n".join(lines)


def _call_llm(query: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"LLM unavailable; using extractive fallback ({type(exc).__name__})")
        return None


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """End-to-end RAG generation with citation and graceful fallback."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = _call_llm(query, context) or _fallback_answer(query, reordered)
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    result = generate_with_citation("Hình phạt cho tội tàng trữ trái phép chất ma túy?")
    print(result["answer"])
