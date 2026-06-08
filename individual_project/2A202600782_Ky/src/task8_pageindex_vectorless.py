"""
Task 8 — PageIndex / Vectorless RAG.

==============================================================================
THIẾT KẾ — retrieval KHÔNG dùng vector, khai thác CẤU TRÚC CÂY của văn bản.
==============================================================================
Tinh thần PageIndex (VectifyAI): thay vì so khớp embedding, ta xây "mục lục cây"
của tài liệu rồi truy hồi bằng SUY LUẬN trên cây (chọn nhánh/nút liên quan).
Điều này CỰC HỢP với văn bản pháp luật vốn có cây phân cấp rõ ràng
Văn bản → Chương → Điều — thứ mà Task 4 đã parse sẵn vào metadata.

=> Quyết định: KHÔNG phụ thuộc SaaS pageindex.ai (cần API key + upload + mạng).
   Ta dựng cây NỘI BỘ từ data/index/chunks.jsonl và truy hồi vectorless theo
   cấu trúc. Chạy offline, deterministic, dùng tốt làm FALLBACK ở Task 9 khi
   hybrid (dense+sparse) cho điểm thấp.

Hai chế độ truy hồi (đều vectorless, đều gắn source="pageindex"):
  • structural (mặc định): chấm điểm nút theo cấu trúc — tiêu đề Điều/Chương
    có trọng số cao, cộng boost khi query nêu đích danh "Điều N" hoặc mã văn
    bản ("57/2022/NĐ-CP"). Khác BM25 ở chỗ làm việc trên NÚT-ĐIỀU (gộp các
    part) và ưu tiên khớp TIÊU ĐỀ — đúng kiểu điều hướng mục lục.
  • llm (bật bằng PAGEINDEX_USE_LLM=1): đưa mục lục (toc) cho LLM, để LLM chọn
    node_id liên quan nhất — "tree search" đúng nghĩa PageIndex.

Đăng ký SaaS chính chủ (nếu muốn dùng): https://pageindex.ai/
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import load_chunks
from .task6_lexical_search import _tokenize

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_MODE = os.getenv("PAGEINDEX_MODE", "auto").lower()
USE_LLM = os.getenv("PAGEINDEX_USE_LLM", "0") == "1"
PROJECT_DIR = Path(__file__).parent.parent
PAGEINDEX_MANIFEST = PROJECT_DIR / "data" / "index" / "pageindex_manifest.json"

# Cây dựng 1 lần rồi cache.
_TREE: list[dict] | None = None

_DIEU_IN_QUERY = re.compile(r"điều\s+(\d+)", re.IGNORECASE)
# Mã văn bản kiểu 57/2022/NĐ-CP, 105/2021/NĐ-CP...
_DOCCODE_RE = re.compile(r"\d+\s*/\s*\d+\s*/\s*[a-zàáâãèéêìíòóôõùúýđ\-]+", re.IGNORECASE)


def _normalize_code(text: str) -> str:
    return re.sub(r"[^0-9a-zA-ZàáâãèéêìíòóôõùúýđĐ/-]+", "", text).lower()


def build_tree(chunks: list[dict] | None = None) -> list[dict]:
    """Dựng cây cấu trúc từ chunks: gộp các part của cùng một Điều thành 1 nút.

    Mỗi nút (NÚT-ĐIỀU với luật, NÚT-CHUNK với báo):
        {node_id, source, doc_title, doc_type, chuong, dieu, dieu_title, content}
    """
    if chunks is None:
        chunks = load_chunks()

    nodes: dict[str, dict] = {}
    order: list[str] = []
    for c in chunks:
        m = c["metadata"]
        dieu = m.get("dieu", 0) or 0
        if m.get("type") == "legal" and dieu:
            key = f"{m['source']}#dieu{dieu}"          # gộp mọi part của Điều
        else:
            key = m.get("chunk_id", f"{m['source']}#{len(order)}")  # báo: từng chunk
        if key not in nodes:
            nodes[key] = {
                "node_id": key,
                "source": m.get("source", ""),
                "doc_title": m.get("doc_title", ""),
                "doc_type": m.get("type", ""),
                "chuong": m.get("chuong", ""),
                "dieu": dieu,
                "dieu_title": m.get("dieu_title", ""),
                "content": c["content"],
            }
            order.append(key)
        else:
            nodes[key]["content"] += "\n" + c["content"]   # nối part tiếp theo
    return [nodes[k] for k in order]


def _get_tree() -> list[dict]:
    global _TREE
    if _TREE is None:
        _TREE = build_tree()
    return _TREE


def _node_header(node: dict) -> str:
    """Văn bản 'tiêu đề' của nút để khớp ưu tiên cao (điều hướng mục lục)."""
    parts = [node.get("doc_title", ""), node.get("chuong", "")]
    if node.get("dieu"):
        parts.append(f"Điều {node['dieu']} {node.get('dieu_title','')}")
    return " ".join(p for p in parts if p)


def _structural_score(q_tokens: set[str], query: str, node: dict) -> float:
    """Chấm điểm vectorless theo cấu trúc: tiêu đề > thân, + boost định danh."""
    if not q_tokens:
        return 0.0
    header_tok = set(_tokenize(_node_header(node)))
    body_tok = set(_tokenize(node["content"]))

    # Khớp tiêu đề (Điều/Chương) quan trọng hơn nhiều so với khớp thân bài.
    score = 3.0 * len(q_tokens & header_tok) / len(q_tokens)
    score += 1.0 * len(q_tokens & body_tok) / len(q_tokens)

    # Boost: query nêu đích danh "Điều N" và nút đúng Điều đó.
    for mm in _DIEU_IN_QUERY.finditer(query):
        if node.get("dieu") == int(mm.group(1)):
            score += 5.0

    # Boost: query chứa mã văn bản và nút thuộc đúng văn bản đó.
    hay = _normalize_code(node.get("source", "") + " " + node["content"][:800])
    for code in _DOCCODE_RE.findall(query.lower()):
        if _normalize_code(code) in hay:
            score += 4.0
    return score


def _search_structural(query: str, top_k: int) -> list[dict]:
    q_tokens = set(_tokenize(query))
    scored = []
    for node in _get_tree():
        s = _structural_score(q_tokens, query, node)
        if s > 0:
            scored.append((s, node))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    seen_nodes: set[str] = set()
    news_per_source: dict[str, int] = {}
    for s, node in scored:
        node_id = node["node_id"]
        if node_id in seen_nodes:
            continue
        if node.get("doc_type") == "news":
            src = node.get("source", "")
            if news_per_source.get(src, 0) >= 2:
                continue
            news_per_source[src] = news_per_source.get(src, 0) + 1
        seen_nodes.add(node_id)
        results.append(_to_result(node, s))
        if len(results) >= top_k:
            break
    return results


def _search_llm(query: str, top_k: int) -> list[dict]:
    """Tree search: đưa mục lục cho LLM chọn node_id liên quan (đúng kiểu PageIndex)."""
    from openai import OpenAI

    tree = _get_tree()
    toc = "\n".join(
        f"{n['node_id']} :: {n.get('doc_title','')} | {n.get('chuong','')} | "
        f"Điều {n['dieu']}. {n.get('dieu_title','')}" if n.get("dieu")
        else f"{n['node_id']} :: {n.get('doc_title','')} (bài báo)"
        for n in tree
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rsp = client.chat.completions.create(
        model=os.getenv("PAGEINDEX_LLM_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {"role": "system", "content":
             "Bạn là bộ định tuyến tài liệu pháp luật. Cho một MỤC LỤC gồm các "
             "node_id và tiêu đề, hãy chọn các node_id liên quan nhất tới câu hỏi. "
             f"Trả về JSON: {{\"node_ids\": [...]}} tối đa {top_k} phần tử."},
            {"role": "user", "content": f"MỤC LỤC:\n{toc}\n\nCÂU HỎI: {query}"},
        ],
        response_format={"type": "json_object"},
    )
    ids = json.loads(rsp.choices[0].message.content).get("node_ids", [])[:top_k]
    by_id = {n["node_id"]: n for n in tree}
    out = []
    for rank, nid in enumerate(ids):
        if nid in by_id:
            out.append(_to_result(by_id[nid], float(top_k - rank)))  # score giảm dần
    return out


def _to_result(node: dict, score: float) -> dict:
    return {
        "content": node["content"],
        "score": float(score),
        "metadata": {
            "source": node["source"],
            "doc_type": node["doc_type"],
            "doc_title": node["doc_title"],
            "chuong": node["chuong"],
            "dieu": node["dieu"],
            "dieu_title": node["dieu_title"],
            "node_id": node["node_id"],
        },
        "source": "pageindex",     # đánh dấu nguồn retrieval (cho Task 9)
    }


def upload_documents():
    """Upload to PageIndex SDK when configured; always support local fallback."""
    if PAGEINDEX_MODE in {"auto", "sdk"} and PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            manifest = {}
            for path in sorted((PROJECT_DIR / "data" / "landing" / "legal").glob("*")):
                if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
                    continue
                submitted = client.submit_document(path)
                manifest[path.name] = {
                    "doc_id": getattr(submitted, "doc_id", None) or getattr(submitted, "id", None),
                    "path": str(path),
                }
            PAGEINDEX_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            PAGEINDEX_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
            print(f"  ✓ Uploaded {len(manifest)} documents to PageIndex SDK.")
            return manifest
        except Exception as exc:
            if PAGEINDEX_MODE == "sdk":
                raise
            print(f"  ⚠ PageIndex SDK upload unavailable ({exc}); dùng local tree.")

    """Vectorless nội bộ không cần upload — chỉ dựng lại cây từ chunks.jsonl."""
    tree = build_tree()
    print(f"  ✓ Dựng cây PageIndex nội bộ: {len(tree)} nút (Điều/bài) — không cần upload.")
    return tree


def _search_sdk(query: str, top_k: int) -> list[dict]:
    if not PAGEINDEX_API_KEY or not PAGEINDEX_MANIFEST.exists():
        return []
    from pageindex import PageIndexClient

    manifest = json.loads(PAGEINDEX_MANIFEST.read_text(encoding="utf-8"))
    doc_ids = [v.get("doc_id") for v in manifest.values() if v.get("doc_id")]
    if not doc_ids:
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    out = []
    for doc_id in doc_ids:
        try:
            if hasattr(client, "is_retrieval_ready") and not client.is_retrieval_ready(doc_id):
                continue
            query_job = client.submit_query(doc_id=doc_id, query=query)
            retrieved = client.get_retrieval(query_job)
        except Exception:
            continue
        for item in retrieved[:top_k]:
            text = getattr(item, "text", None) or getattr(item, "content", None) or str(item)
            score = getattr(item, "score", 1.0)
            out.append({
                "content": text,
                "score": float(score),
                "metadata": {"source": f"pageindex:{doc_id}", "doc_type": "pageindex",
                             "doc_title": f"PageIndex {doc_id}", "dieu": 0,
                             "dieu_title": "", "node_id": str(doc_id)},
                "source": "pageindex",
            })
            if len(out) >= top_k:
                return out
    return out


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval theo cấu trúc cây. Dùng làm fallback ở Task 9.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
    """
    if PAGEINDEX_MODE in {"auto", "sdk"}:
        try:
            sdk_results = _search_sdk(query, top_k)
            if sdk_results:
                return sdk_results
        except Exception as exc:
            if PAGEINDEX_MODE == "sdk":
                raise
            print(f"  ⚠ PageIndex SDK search unavailable ({exc}); dùng local structural.")

    if USE_LLM and os.getenv("OPENAI_API_KEY"):
        try:
            return _search_llm(query, top_k)
        except Exception as e:           # lỗi mạng/LLM -> rơi về structural
            print(f"  ⚠ PageIndex LLM lỗi ({e}); dùng structural.")
    return _search_structural(query, top_k)


if __name__ == "__main__":
    upload_documents()
    for q in ["Điều 249 tàng trữ ma túy", "57/2022/NĐ-CP", "cai nghiện ma túy"]:
        print(f"\nQuery: {q}\n" + "-" * 60)
        for r in pageindex_search(q, top_k=3):
            m = r["metadata"]
            print(f"  [{r['score']:.2f}] {m['source']} Điều {m['dieu']}. {m['dieu_title'][:45]}")
