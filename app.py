"""Streamlit RAG chatbot for Vietnamese drug law and related news."""

from __future__ import annotations

import html
import re

import streamlit as st

from src.task10_generation import generate_with_citation


MAX_MEMORY_TURNS = 3


def highlight_terms(text: str, query: str) -> str:
    escaped = html.escape(text)
    terms = sorted(
        {term for term in re.findall(r"\w+", query, flags=re.UNICODE) if len(term) > 2},
        key=len,
        reverse=True,
    )
    for term in terms:
        escaped = re.sub(
            rf"(?i)(?<!\w)({re.escape(html.escape(term))})(?!\w)",
            r"<mark>\1</mark>",
            escaped,
        )
    return escaped


def build_contextual_query(question: str, messages: list[dict] | None = None) -> str:
    """Append short conversation memory so follow-up questions can be resolved."""
    turns = (messages if messages is not None else st.session_state.messages)[-MAX_MEMORY_TURNS * 2 :]
    if not turns:
        return question

    history = []
    for message in turns:
        role = "Người dùng" if message["role"] == "user" else "Trợ lý"
        history.append(f"{role}: {message['content']}")

    return (
        "Ngữ cảnh hội thoại gần nhất:\n"
        + "\n".join(history)
        + "\n\nCâu hỏi mới cần trả lời dựa trên corpus pháp luật/tin tức:\n"
        + question
    )


def render_sources(sources: list[dict], query: str) -> None:
    if not sources:
        st.info("Không có source document được truy hồi.")
        return

    with st.expander(f"Source documents đã dùng ({len(sources)})", expanded=True):
        for rank, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("source") or metadata.get("title") or "Không rõ nguồn"
            source_path = metadata.get("source_path") or ""
            doc_type = metadata.get("type") or "unknown"
            retrieval_source = source.get("source", "hybrid")
            score = float(source.get("score", 0.0))
            content = source.get("content", "")

            st.markdown(
                f"""
                <article class="result">
                  <h4>{rank}. {html.escape(title)}</h4>
                  <div class="meta">Relevance {score:.3f} · {html.escape(doc_type)}
                  · {html.escape(retrieval_source)} · {html.escape(source_path)}</div>
                  <div>{highlight_terms(content, query)}</div>
                </article>
                """,
                unsafe_allow_html=True,
            )


st.set_page_config(page_title="RAG Chatbot pháp luật ma túy", page_icon="§", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    :root { --ink:#17201d; --paper:#f4f0e7; --accent:#b9442d; --line:#d4c9b7; }
    .stApp { background: radial-gradient(circle at 90% 0%, #e7d6bd 0, transparent 30%), var(--paper); color:var(--ink); }
    h1,h2,h3,h4 { font-family:'DM Serif Display',serif!important; letter-spacing:-.02em; }
    html,body,[class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
    .result { background:#fffdf8; border:1px solid var(--line); border-left:5px solid var(--accent);
      padding:1rem 1.2rem; margin:.8rem 0; box-shadow:0 8px 24px rgba(44,34,20,.06); }
    .meta { color:#6e6256; font-size:.86rem; margin-bottom:.7rem; }
    mark { background:#f3c967; color:var(--ink); padding:0 .12em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Chatbot pháp luật & tin tức ma túy")
st.caption("Task 9 retrieval + Task 10 generation có citation, conversation memory và source display")

with st.sidebar:
    st.header("Cấu hình")
    top_k = st.slider("Số source chunks", 1, 10, 5)
    use_reranking = st.toggle("Dùng cross-encoder reranking", value=True)
    st.caption("Follow-up memory dùng tối đa 3 lượt hội thoại gần nhất.")
    if st.button("Xoá hội thoại"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []), message.get("query", ""))

prompt = st.chat_input("Ví dụ: Khung hình phạt thấp nhất cho tội tàng trữ ma túy là gì?")
if prompt:
    contextual_query = build_contextual_query(prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy hồi nguồn và sinh câu trả lời có citation..."):
            try:
                result = generate_with_citation(
                    contextual_query,
                    top_k=top_k,
                    use_reranking=use_reranking,
                )
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)
                render_sources(sources, prompt)
            except Exception as exc:
                answer = f"Lỗi khi chạy RAG pipeline: `{type(exc).__name__}: {exc}`"
                sources = []
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "query": prompt}
    )
    st.session_state.last_sources = sources
