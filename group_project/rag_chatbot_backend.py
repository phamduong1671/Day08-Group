"""
Backend cho yêu cầu 1 - RAG Chatbot nhóm.

UI team có thể import `chat()` hoặc `answer_question()` từ file này. File này
không tạo giao diện, chỉ xử lý:
    - trả lời có citation qua Task 10,
    - conversation memory cho follow-up questions,
    - chuẩn hóa source documents để UI hiển thị.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.task10_generation import citation_label, generate_with_citation


MAX_SOURCE_PREVIEW_CHARS = 500
MAX_MEMORY_TURNS = 6
SOURCE_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class ConversationTurn:
    user: str
    assistant: str
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationMemory:
    max_turns: int = MAX_MEMORY_TURNS
    turns: list[ConversationTurn] = field(default_factory=list)

    def add_turn(self, user: str, assistant: str, sources: list[dict[str, Any]]) -> None:
        self.turns.append(ConversationTurn(user=user, assistant=assistant, sources=sources))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def as_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def retrieval_context(self, last_turns: int = 2) -> str:
        recent_turns = self.turns[-last_turns:]
        parts: list[str] = []
        for turn in recent_turns:
            parts.append(f"User: {turn.user}")
            parts.append(f"Assistant: {_compact_text(turn.assistant, max_chars=420)}")
        return "\n".join(parts)

    def clear(self) -> None:
        self.turns.clear()


_SESSIONS: dict[str, ConversationMemory] = {}


def get_memory(session_id: str = "default") -> ConversationMemory:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = ConversationMemory()
    return _SESSIONS[session_id]


def reset_session(session_id: str = "default") -> None:
    get_memory(session_id).clear()


def _compact_text(text: str, max_chars: int = 300) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _exact_phrase_candidates(question: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
    phrases = [first or second for first, second in quoted if (first or second).strip()]
    if not phrases:
        phrases = [question]
    return [_normalize_text(phrase) for phrase in phrases if _normalize_text(phrase)]


def _filter_exact_sources(sources: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    phrases = _exact_phrase_candidates(question)
    if not phrases:
        return sources

    filtered: list[dict[str, Any]] = []
    for source in sources:
        haystack = _normalize_text(
            " ".join(
                [
                    source.get("content", ""),
                    source.get("preview", ""),
                    source.get("citation", ""),
                    source.get("source", ""),
                ]
            )
        )
        if any(phrase in haystack for phrase in phrases):
            filtered.append(source)
    return filtered


def build_contextual_query(question: str, memory: ConversationMemory, exact_phrase: bool = False) -> str:
    """
    Ghép câu hỏi mới với các lượt gần nhất để follow-up không bị mất ngữ cảnh.

    Ví dụ user hỏi "mức phạt là gì?", retrieval cần biết câu trước đang nói về
    tàng trữ, tổ chức sử dụng, hay cai nghiện.
    """
    if exact_phrase:
        return f'"{question}"'

    history = memory.retrieval_context()
    if not history:
        return question

    return (
        "Ngữ cảnh hội thoại gần nhất:\n"
        f"{history}\n\n"
        f"Câu hỏi hiện tại: {question}"
    )


def _extract_citations(answer: str, sources: list[dict[str, Any]]) -> list[str]:
    cited = [match.strip() for match in SOURCE_CITATION_PATTERN.findall(answer)]
    if cited:
        return list(dict.fromkeys(cited))

    labels = [source.get("citation") for source in sources if source.get("citation")]
    return list(dict.fromkeys(labels))


def _normalize_sources(raw_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, source in enumerate(raw_sources, 1):
        metadata = source.get("metadata", {}) or {}
        citation = citation_label(source, fallback_index=index)
        source_path = metadata.get("source_path", "")
        source_name = metadata.get("source", f"Source {index}")
        key = metadata.get("chunk_id") or f"{source_path}:{metadata.get('chunk_index', index)}"

        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            {
                "rank": len(normalized) + 1,
                "citation": citation,
                "source": source_name,
                "source_path": source_path,
                "type": metadata.get("type", "unknown"),
                "score": float(source.get("score", 0.0)),
                "retrieval_source": source.get("source", "hybrid"),
                "chunk_index": metadata.get("chunk_index", metadata.get("block_index")),
                "content": source.get("content", ""),
                "preview": _compact_text(
                    source.get("content", ""),
                    max_chars=MAX_SOURCE_PREVIEW_CHARS,
                ),
            }
        )

    return normalized


def answer_question(
    question: str,
    session_id: str = "default",
    top_k: int = 5,
    exact_phrase: bool = False,
) -> dict[str, Any]:
    """
    API chính cho UI.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "source_documents": list[dict],
            "citations": list[str],
            "session_id": str,
            "retrieval_query": str,
            "history": list[dict],
        }
    """
    memory = get_memory(session_id)
    contextual_query = build_contextual_query(question, memory, exact_phrase=exact_phrase)

    result = generate_with_citation(
        query=question,
        top_k=top_k,
        conversation_history=memory.as_messages(),
        retrieval_query=contextual_query,
    )
    sources = _normalize_sources(result.get("sources", []))
    if exact_phrase:
        sources = _filter_exact_sources(sources, question)
        if not sources:
            answer = (
                "Tôi không tìm thấy văn bản nào chứa chính xác cụm từ bạn nhập. "
                "Bạn có thể bỏ chọn 'Tìm chính xác cụm từ' để tìm theo các từ trong câu hỏi."
            )
            memory.add_turn(question, answer, [])
            return {
                "answer": answer,
                "sources": [],
                "source_documents": [],
                "citations": [],
                "session_id": session_id,
                "retrieval_query": contextual_query,
                "retrieval_source": result.get("retrieval_source", "none"),
                "generation_backend": result.get("generation_backend", "unknown"),
                "history": memory.as_messages(),
                "search_mode": "exact_phrase",
            }
    citations = _extract_citations(result.get("answer", ""), sources)

    memory.add_turn(question, result.get("answer", ""), sources)

    return {
        "answer": result.get("answer", ""),
        "sources": sources,
        "source_documents": sources,
        "citations": citations,
        "session_id": session_id,
        "retrieval_query": result.get("retrieval_query", contextual_query),
        "retrieval_source": result.get("retrieval_source", "none"),
        "generation_backend": result.get("generation_backend", "unknown"),
        "history": memory.as_messages(),
        "search_mode": "exact_phrase" if exact_phrase else "keyword",
    }


def chat(
    question: str,
    session_id: str = "default",
    top_k: int = 5,
    exact_phrase: bool = False,
) -> dict[str, Any]:
    """Alias ngắn cho UI Streamlit/Gradio/Chainlit."""
    return answer_question(
        question=question,
        session_id=session_id,
        top_k=top_k,
        exact_phrase=exact_phrase,
    )


if __name__ == "__main__":
    demo = chat("Luật Phòng chống ma túy quy định chất ma túy là gì?", session_id="demo")
    print(demo["answer"])
    print("\nSources:")
    for source in demo["sources"]:
        print(f"- [{source['score']:.3f}] {source['citation']} :: {source['source_path']}")
