"""
Backend cho yêu cầu 1 - RAG Chatbot nhóm.

UI team có thể import `chat()` hoặc `answer_question()` từ file này. File này
không tạo giao diện, chỉ xử lý:
    - trả lời có citation qua Task 10,
    - conversation memory cho follow-up questions,
    - chuẩn hóa source documents để UI hiển thị.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.task10_generation import citation_label, generate_with_citation


MAX_SOURCE_PREVIEW_CHARS = 500
MAX_MEMORY_TURNS = 6
SOURCE_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or "")).casefold()
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", without_marks).strip()


STOPWORDS = {
    "a", "ai", "anh", "ban", "bao", "bi", "bo", "cai", "cac", "cach", "can",
    "cho", "co", "cua", "dang", "de", "den", "duoc", "gi", "hay", "hoi",
    "khong", "la", "lam", "mot", "nao", "nay", "nhung", "nguoi", "nhu",
    "o", "phai", "qua", "ra", "sao", "the", "thi", "toi", "trong", "tu",
    "va", "ve", "voi", "xu", "ly",
}


def _query_terms(question: str) -> list[str]:
    terms = re.findall(r"[\wÀ-ỹ]+", _fold_text(question), flags=re.UNICODE)
    return [term for term in terms if len(term) >= 2 and term not in STOPWORDS]


def _exact_phrase_candidates(question: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
    phrases = [first or second for first, second in quoted if (first or second).strip()]
    if not phrases:
        phrases = [question]
    return [_fold_text(phrase) for phrase in phrases if _fold_text(phrase)]


def _source_haystack(source: dict[str, Any]) -> str:
    return _fold_text(
        " ".join(
            [
                source.get("content", ""),
                source.get("preview", ""),
                source.get("citation", ""),
                source.get("source", ""),
            ]
        )
    )


def _filter_exact_sources(sources: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    phrases = _exact_phrase_candidates(question)
    if not phrases:
        return sources

    filtered: list[dict[str, Any]] = []
    for source in sources:
        haystack = _source_haystack(source)
        if any(phrase in haystack for phrase in phrases):
            filtered.append(source)
    return filtered


def _filter_keyword_sources(sources: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    terms = _query_terms(question)
    if not terms:
        return sources

    scored: list[tuple[int, float, dict[str, Any]]] = []
    for source in sources:
        haystack = _source_haystack(source)
        matched = sum(1 for term in terms if term in haystack)
        if matched:
            scored.append((matched, float(source.get("score", 0.0)), source))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [source for _, _, source in scored]


def _best_source_snippet(question: str, source: dict[str, Any], max_chars: int = 360) -> str:
    terms = set(_query_terms(question))
    content = source.get("content") or source.get("preview") or ""
    candidates: list[str] = []
    for line in str(content).splitlines():
        line = line.strip(" -")
        if not line or line.startswith(("#", "**Source", "**Type", "---")):
            continue
        candidates.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if len(part.strip()) >= 35)

    if not candidates:
        compact = _compact_text(str(content), max_chars=max_chars)
        return compact

    def score(sentence: str) -> tuple[int, int]:
        sentence_terms = set(_query_terms(sentence))
        return (len(terms & sentence_terms), -abs(len(sentence) - 180))

    snippet = max(candidates, key=score)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
    return snippet


def _answer_from_sources(question: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "Tôi không tìm thấy tài liệu phù hợp với các từ trong câu hỏi."

    lines = []
    for source in sources[:3]:
        snippet = _best_source_snippet(question, source)
        citation = source.get("citation") or source.get("source") or "Nguồn"
        if snippet:
            lines.append(f"- {snippet} [{citation}]")

    if not lines:
        return "Tôi không tìm thấy nội dung đủ phù hợp trong các source documents."
    return "Dựa trên các source documents khớp với câu hỏi:\n" + "\n".join(lines)


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for source in sources:
        key = str(source.get("source_path") or source.get("citation") or source.get("source") or source.get("content", "")[:80])
        if key in seen:
            continue
        seen.add(key)
        source = dict(source)
        source["rank"] = len(deduped) + 1
        deduped.append(source)
    return deduped


def _read_source_file(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            title = data.get("title") or path.stem
            content = data.get("content_markdown") or data.get("content") or data.get("markdown") or ""
            return str(title), str(content)
        except Exception:
            return path.stem, path.read_text(encoding="utf-8", errors="ignore")
    return path.stem, path.read_text(encoding="utf-8", errors="ignore")


def _scan_local_sources(question: str, top_k: int, exact_phrase: bool) -> list[dict[str, Any]]:
    candidates = [
        *PROJECT_ROOT.glob("data/standardized/**/*.md"),
        *PROJECT_ROOT.glob("data/landing/**/*.json"),
    ]
    phrases = _exact_phrase_candidates(question)
    terms = _query_terms(question)
    scored: list[tuple[float, dict[str, Any]]] = []

    for path in candidates:
        title, content = _read_source_file(path)
        if not content.strip():
            continue
        haystack = _fold_text(" ".join([title, content]))
        if exact_phrase:
            if not any(phrase in haystack for phrase in phrases):
                continue
            score = 1.0
        else:
            if not terms:
                continue
            matched = sum(1 for term in terms if term in haystack)
            min_match = 1 if len(terms) <= 2 else min(3, len(terms))
            if matched < min_match:
                continue
            score = matched / max(1, len(terms))

        relative = path.relative_to(PROJECT_ROOT).as_posix()
        doc_type = "news" if "/news/" in f"/{relative}" else "legal"
        scored.append(
            (
                score,
                {
                    "rank": 0,
                    "citation": title,
                    "source": title,
                    "source_path": relative,
                    "type": doc_type,
                    "score": round(score, 3),
                    "retrieval_source": "local_exact" if exact_phrase else "local_keyword",
                    "chunk_index": None,
                    "content": content,
                    "preview": _compact_text(content, max_chars=MAX_SOURCE_PREVIEW_CHARS),
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [source for _, source in scored[:top_k]]


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
    score_threshold: float | None = None,
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
        score_threshold=score_threshold,
        conversation_history=memory.as_messages(),
        retrieval_query=contextual_query,
    )
    sources = _normalize_sources(result.get("sources", []))
    if exact_phrase:
        sources = _dedupe_sources(
            _filter_exact_sources(sources, question)
            + _scan_local_sources(question, top_k=top_k, exact_phrase=True)
        )[:top_k]
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
        answer = _answer_from_sources(question, sources)
        citations = _extract_citations(answer, sources)
        memory.add_turn(question, answer, sources)
        return {
            "answer": answer,
            "sources": sources,
            "source_documents": sources,
            "citations": citations,
            "session_id": session_id,
            "retrieval_query": contextual_query,
            "retrieval_source": result.get("retrieval_source", "none"),
            "generation_backend": "source_filtered_extractive",
            "history": memory.as_messages(),
            "search_mode": "exact_phrase",
        }
    keyword_sources = _dedupe_sources(
        _filter_keyword_sources(sources, question)
        + _scan_local_sources(question, top_k=top_k, exact_phrase=False)
    )[:top_k]
    if keyword_sources:
        sources = keyword_sources
        answer = _answer_from_sources(question, sources)
        citations = _extract_citations(answer, sources)
        memory.add_turn(question, answer, sources)
        return {
            "answer": answer,
            "sources": sources,
            "source_documents": sources,
            "citations": citations,
            "session_id": session_id,
            "retrieval_query": result.get("retrieval_query", contextual_query),
            "retrieval_source": result.get("retrieval_source", "none"),
            "generation_backend": "source_filtered_extractive",
            "history": memory.as_messages(),
            "search_mode": "keyword",
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
    score_threshold: float | None = None,
) -> dict[str, Any]:
    """Alias ngắn cho UI Streamlit/Gradio/Chainlit."""
    return answer_question(
        question=question,
        session_id=session_id,
        top_k=top_k,
        exact_phrase=exact_phrase,
        score_threshold=score_threshold,
    )


if __name__ == "__main__":
    demo = chat("Luật Phòng chống ma túy quy định chất ma túy là gì?", session_id="demo")
    print(demo["answer"])
    print("\nSources:")
    for source in demo["sources"]:
        print(f"- [{source['score']:.3f}] {source['citation']} :: {source['source_path']}")
