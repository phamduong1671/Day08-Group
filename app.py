"""Option A: Streamlit search engine for the hybrid retrieval pipeline."""

from __future__ import annotations

import html
import re

import streamlit as st

from src.task9_retrieval_pipeline import retrieve


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


st.set_page_config(page_title="Tra cứu pháp luật ma túy", page_icon="§", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    :root { --ink:#17201d; --paper:#f4f0e7; --accent:#b9442d; --line:#d4c9b7; }
    .stApp { background: radial-gradient(circle at 90% 0%, #e7d6bd 0, transparent 30%), var(--paper); color:var(--ink); }
    h1,h2,h3 { font-family:'DM Serif Display',serif!important; letter-spacing:-.02em; }
    html,body,[class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
    .result { background:#fffdf8; border:1px solid var(--line); border-left:5px solid var(--accent);
      padding:1.2rem 1.4rem; margin:.8rem 0; box-shadow:0 8px 24px rgba(44,34,20,.06); }
    .meta { color:#6e6256; font-size:.86rem; margin-bottom:.7rem; }
    mark { background:#f3c967; color:var(--ink); padding:0 .12em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tra cứu pháp luật & tin tức ma túy")
st.caption("Hybrid semantic + BM25, cross-encoder reranking, PageIndex fallback")

query = st.text_input(
    "Nội dung cần tìm",
    placeholder="Ví dụ: Hình phạt cho tội tàng trữ trái phép chất ma túy",
)
top_k = st.slider("Số kết quả", 1, 10, 5)

if query.strip():
    with st.spinner("Đang tìm và xếp hạng nguồn..."):
        results = retrieve(query.strip(), top_k=top_k)
    st.subheader(f"{len(results)} kết quả phù hợp")
    for rank, result in enumerate(results, 1):
        metadata = result["metadata"]
        source = metadata.get("source") or "Không rõ nguồn"
        source_path = metadata.get("source_path") or ""
        doc_type = metadata.get("type") or "unknown"
        st.markdown(
            f"""
            <article class="result">
              <h3>{rank}. {html.escape(source)}</h3>
              <div class="meta">Relevance {result['score']:.3f} · {html.escape(doc_type)}
              · {html.escape(result['source'])} · {html.escape(source_path)}</div>
              <div>{highlight_terms(result['content'], query)}</div>
            </article>
            """,
            unsafe_allow_html=True,
        )
