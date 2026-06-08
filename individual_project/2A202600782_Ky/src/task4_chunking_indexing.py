"""
Task 4 — Chunking & Indexing vào Vector Store.

Pipeline: load_documents -> chunk_documents -> embed_chunks -> index_to_vectorstore

==============================================================================
QUYẾT ĐỊNH THIẾT KẾ — tối ưu cho corpus "chủ yếu là LUẬT + ít BÀI BÁO"
==============================================================================
Bài toán đặc thù: phần lớn tài liệu là văn bản quy phạm pháp luật (Luật, Nghị
định) có cấu trúc phân cấp Chương -> Điều -> Khoản; còn lại là vài bài báo
(văn xuôi). Hai loại này cần cách cắt khác nhau.

1. CHUNKING — STRUCTURE-AWARE theo loại tài liệu (không cắt cứng theo ký tự):

   a) LUẬT: cắt theo ranh giới **Điều** — đơn vị pháp lý nguyên tử mà người ta
      trích dẫn ("theo Điều 249 BLHS"). Mỗi Điều:
        - Nếu <= CHUNK_SIZE  -> giữ NGUYÊN 1 chunk (không xé ngang quy định).
        - Nếu dài hơn (vd bảng Danh mục chất ma túy ~72k ký tự) -> cắt nhỏ theo
          Khoản/đoạn, NHƯNG prepend lại header "Điều N. <tiêu đề>" vào MỌI mảnh
          con (contextual header) -> mảnh nào cũng còn số Điều để grounding +
          trích dẫn, và retrieval không lẫn quy định của hai Điều khác nhau.
      Phần mở đầu (căn cứ ban hành, trước Điều 1) tách riêng.

   b) BÁO: recursive splitting cho văn xuôi, prepend tiêu đề bài vào mỗi chunk.

   => Lý do KHÔNG dùng MarkdownHeaderTextSplitter: trong file đã chuẩn hoá,
      "Điều"/"Chương" là **bold** chứ không phải heading `#`, nên splitter theo
      `#` vô dụng; ta tự parse cấu trúc bằng regex.

2. METADATA giàu cho citation/filter: doc_title, chuong, dieu, dieu_title, part.

3. EMBEDDING — BAAI/bge-m3 (1024 chiều, multilingual, mạnh tiếng Việt,
   normalize_embeddings=True, tự dùng CUDA nếu có).

4. VECTOR STORE — Weaviate (vectorizer=none, ta tự cấp vector). Field text được
   Weaviate tự lập chỉ mục BM25 => bật được HYBRID search (dense + lexical).
   Hybrid quan trọng cho luật: query thường chứa định danh chính xác như
   "57/2022/NĐ-CP", "Methamphetamine", "Điều 249" mà dense dễ bỏ sót.

5. SOURCE-OF-TRUTH — data/index/chunks.jsonl: lưu mọi chunk (content + metadata,
   không kèm embedding) để Task 6 (BM25) và các task khác tái sử dụng cùng tập.
==============================================================================
"""

import json
import re
from pathlib import Path

from .model_runtime import load_with_cpu_fallback

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"

# =============================================================================
# CONFIGURATION
# =============================================================================
CHUNK_SIZE = 1500        # ký tự — khớp median 1 Điều (~900-1600), giữ Điều nguyên
CHUNK_OVERLAP = 200      # ~13% — ngữ cảnh bắc cầu khi 1 Điều dài bị cắt nhỏ
MIN_CHUNK_CHARS = 40     # bỏ mảnh quá ngắn (separator lẻ) -> tránh rác trong index
CHUNKING_METHOD = "structure_aware"   # luật: theo Điều | báo: recursive văn xuôi

EMBEDDING_MODEL = "BAAI/bge-m3"   # multilingual, mạnh tiếng Việt
EMBEDDING_DIM = 1024

VECTOR_STORE = "weaviate"
COLLECTION_NAME = "DrugLawDocs"

# Tách thân Điều dài: ưu tiên ranh giới Khoản/đoạn rồi tới câu/từ.
BODY_SEPARATORS = ["\n\n", "\n", "; ", ". ", " ", ""]

# Một dòng bắt đầu bằng "Điều N" (cho phép bọc bởi * _ # markdown).
_ARTICLE_RE = re.compile(r"(?m)^[*_#\s]*Điều\s+(\d+)\s*[.:)]")
# Một dòng tiêu đề Chương.
_CHUONG_RE = re.compile(r"(?m)^[*_#\s]*(Chương\s+[IVXLCDM\d]+[^\n]*)")


# =============================================================================
# LOAD
# =============================================================================
def _doc_title(content: str, fallback: str) -> str:
    """Lấy tiêu đề tài liệu từ dòng `# ...` đầu tiên."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown trong data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source','type','doc_title'}}
    """
    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "doc_title": _doc_title(content, md_file.stem),
                },
            }
        )
    return documents


# =============================================================================
# CHUNKING (structure-aware)
# =============================================================================
def _clean_md(s: str) -> str:
    """Bỏ ký tự nhấn mạnh markdown ở đầu/cuối + khoảng trắng thừa."""
    return s.strip().strip("*_# ").strip()


def _recursive_split(text: str, size: int) -> list[str]:
    """Cắt văn bản dài thành mảnh <= size bằng RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    size = max(size, 200)  # guard nếu header quá dài
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(CHUNK_OVERLAP, size // 4),
        separators=BODY_SEPARATORS,
        length_function=len,
    )
    return [p.strip() for p in splitter.split_text(text) if p.strip()]


def _chuong_at(content: str, pos: int) -> str:
    """Tiêu đề Chương gần nhất đứng TRƯỚC vị trí pos (để gắn metadata Điều)."""
    last = ""
    for m in _CHUONG_RE.finditer(content):
        if m.start() < pos:
            last = _clean_md(m.group(1))
        else:
            break
    return last


def _split_legal(content: str, base: dict) -> list[dict]:
    """Cắt văn bản luật theo Điều (giữ Điều nguyên nếu vừa, prepend header nếu dài)."""
    matches = list(_ARTICLE_RE.finditer(content))
    out: list[dict] = []

    def emit(text: str, dieu: int, dieu_title: str, chuong: str, part: int):
        text = text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            return
        meta = dict(base)
        meta.update({"chuong": chuong, "dieu": dieu, "dieu_title": dieu_title, "part": part})
        out.append({"content": text, "metadata": meta})

    if not matches:  # không nhận ra cấu trúc Điều -> fallback văn xuôi
        for p, piece in enumerate(_recursive_split(content, CHUNK_SIZE)):
            emit(piece, 0, "", "", p)
        return out

    # Phần mở đầu (căn cứ ban hành) trước Điều 1.
    preamble = content[: matches[0].start()].strip()
    for p, piece in enumerate(_recursive_split(preamble, CHUNK_SIZE)):
        emit(piece, 0, "Phần mở đầu", _chuong_at(content, 0), p)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        seg = content[start:end].strip()
        dieu = int(m.group(1))
        header_line = seg.split("\n", 1)[0]
        header = _clean_md(header_line)                      # "Điều 249. Tội ..."
        dieu_title = re.sub(r"^Điều\s+\d+\s*[.:)]?\s*", "", header).strip()
        chuong = _chuong_at(content, start)

        if len(seg) <= CHUNK_SIZE:
            emit(seg, dieu, dieu_title, chuong, 0)           # giữ nguyên cả Điều
        else:
            body = seg[len(header_line):].strip()
            avail = CHUNK_SIZE - len(header) - 1             # chừa chỗ cho header
            for p, piece in enumerate(_recursive_split(body, avail)):
                emit(f"{header}\n{piece}", dieu, dieu_title, chuong, p)
    return out


def _split_news(content: str, base: dict) -> list[dict]:
    """Cắt bài báo (văn xuôi) + prepend tiêu đề bài vào mỗi chunk để grounding."""
    title = base.get("doc_title", "")
    prefix = f"[{title}]\n" if title else ""
    out: list[dict] = []
    for p, piece in enumerate(_recursive_split(content, CHUNK_SIZE - len(prefix))):
        text = (prefix + piece).strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        meta = dict(base)
        meta.update({"chuong": "", "dieu": 0, "dieu_title": "", "part": p})
        out.append({"content": text, "metadata": meta})
    return out


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Dispatch theo loại tài liệu; gán chunk_index toàn cục + chunk_id."""
    chunks: list[dict] = []
    for doc in documents:
        base = doc["metadata"]
        if base["type"] == "legal":
            chunks.extend(_split_legal(doc["content"], base))
        else:
            chunks.extend(_split_news(doc["content"], base))

    for idx, c in enumerate(chunks):
        m = c["metadata"]
        m["chunk_index"] = idx
        m["chunk_id"] = f"{m['source']}::d{m.get('dieu', 0)}::p{m.get('part', 0)}"
    return chunks


# =============================================================================
# PERSIST (source-of-truth cho Task 6)
# =============================================================================
def persist_chunks(chunks: list[dict], path: Path = CHUNKS_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"content": c["content"], "metadata": c["metadata"]},
                               ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy Task 4 trước.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# =============================================================================
# EMBED
# =============================================================================
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = load_with_cpu_fallback(
            lambda device: SentenceTransformer(EMBEDDING_MODEL, device=device),
            "EMBEDDING_DEVICE",
        )
    return _EMBEDDER


def embed_chunks(chunks: list[dict]) -> list[dict]:
    model = _get_embedder()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


# =============================================================================
# INDEX (Weaviate, hybrid-ready)
# =============================================================================
def connect_weaviate():
    import weaviate

    return weaviate.connect_to_local()


def index_to_vectorstore(chunks: list[dict]):
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances

    client = connect_weaviate()
    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)  # idempotent re-index

        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="doc_title", data_type=DataType.TEXT),
                Property(name="chuong", data_type=DataType.TEXT),
                Property(name="dieu", data_type=DataType.INT),
                Property(name="dieu_title", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )
        collection = client.collections.get(COLLECTION_NAME)
        with collection.batch.dynamic() as batch:
            for c in chunks:
                m = c["metadata"]
                batch.add_object(
                    properties={
                        "content": c["content"],
                        "source": m["source"],
                        "doc_type": m["type"],
                        "doc_title": m.get("doc_title", ""),
                        "chuong": m.get("chuong", ""),
                        "dieu": m.get("dieu", 0),
                        "dieu_title": m.get("dieu_title", ""),
                        "chunk_id": m["chunk_id"],
                        "chunk_index": m["chunk_index"],
                    },
                    vector=c["embedding"],
                )
        failed = collection.batch.failed_objects
        if failed:
            print(f"  ⚠ {len(failed)} object insert lỗi (vd: {failed[0].message})")
        total = collection.aggregate.over_all(total_count=True).total_count
        print(f"  ✓ Collection '{COLLECTION_NAME}' hiện có {total} objects")
    finally:
        client.close()


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
    """Hybrid (dense + BM25) search — alpha=1.0 thuần vector, 0.0 thuần BM25.

    Dùng để kiểm chứng index ở Task 4 và làm nền cho Task 5/6/9.
    """
    vec = _get_embedder().encode([query], normalize_embeddings=True)[0].tolist()
    client = connect_weaviate()
    try:
        col = client.collections.get(COLLECTION_NAME)
        res = col.query.hybrid(
            query=query, vector=vec, alpha=alpha, limit=top_k, return_metadata=["score"]
        )
        return [
            {
                "content": o.properties["content"],
                "score": o.metadata.score,
                "metadata": {k: o.properties.get(k) for k in
                             ("source", "doc_title", "chuong", "dieu", "dieu_title")},
            }
            for o in res.objects
        ]
    finally:
        client.close()


# =============================================================================
# PIPELINE
# =============================================================================
def run_pipeline():
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} / collection={COLLECTION_NAME} (hybrid-ready)")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    sizes = [len(c["content"]) for c in chunks]
    whole = sum(1 for c in chunks if c["metadata"].get("dieu") and c["metadata"].get("part") == 0)
    print(f"✓ Created {len(chunks)} chunks "
          f"(min={min(sizes)}, avg={sum(sizes)//len(sizes)}, max={max(sizes)} chars)")

    persist_chunks(chunks)
    print(f"✓ Persisted chunks -> {CHUNKS_FILE.relative_to(PROJECT_DIR)}")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks (dim={len(chunks[0]['embedding'])})")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
