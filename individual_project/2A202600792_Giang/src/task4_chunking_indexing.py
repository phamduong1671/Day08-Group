"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Weaviate embedded persists data here — no Docker required.
WEAVIATE_PERSIST_DIR = Path(__file__).parent.parent / "data" / "weaviate_store"

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# RecursiveCharacterTextSplitter được chọn vì:
# - Tự động tách theo đoạn (\n\n), dòng (\n), câu (". "), rồi mới ký tự, phù hợp cả văn bản pháp luật (nhiều điều khoản dài) lẫn bài báo ngắn.
CHUNK_SIZE = 500        # 500 ký tự ≈ 2 - 4 câu văn xuôi tiếng Việt; đủ ngữ cảnh để embedding nắm ý, nhưng không quá dài làm loãng vector.
CHUNK_OVERLAP = 50      # 10 % overlap giữ lại câu cuối của chunk trước, tránh cắt đứt ý ở ranh giới (ví dụ: "Điều 1. ... [cut] khoản 2").
CHUNKING_METHOD = "recursive" 

# BAAI/bge-m3 được chọn vì:
# - Hỗ trợ đa ngôn ngữ (Vietnamese, Chinese, English) tốt hơn MiniLM.
# - 1024 chiều cho độ phân biệt ngữ nghĩa cao hơn 384 chiều của MiniLM.
# - Chạy local, không cần API key như OpenAI.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Weaviate được chọn vì hỗ trợ hybrid search (dense + BM25) built-in,quan trọng cho văn bản pháp luật tiếng Việt chứa nhiều thuật ngữ đặc thù.
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"

COLLECTION_NAME = "DrugRelatedDocs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        # Dùng tên thư mục cha để phân loại "legal" vs "news" thay vì regex string - tránh false-positive nếu tên file chứa từ "legal".
        doc_type = md_file.parent.name
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Thứ tự separator: ưu tiên tách theo đoạn văn rồi mới xuống câu/ký tự, giữ nguyên cấu trúc điều khoản pháp luật càng nhiều càng tốt.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]

    # batch_size=32 cân bằng tốc độ và RAM; bge-m3 nặng hơn MiniLM nên không nên dùng batch quá lớn trên máy không có GPU.
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào Weaviate embedded (không cần Docker/server riêng).
    Dữ liệu được persist tại data/weaviate_store/ để dùng lại sau.
    """
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property

    WEAVIATE_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    # connect_to_embedded() khởi động Weaviate trong cùng process Python - không cần Docker, phù hợp môi trường dev/lab.
    with weaviate.connect_to_embedded(
        persistence_data_path=str(WEAVIATE_PERSIST_DIR),
    ) as client:
        # Xoá collection cũ nếu đã tồn tại để chạy lại idempotent.
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)

        collection = client.collections.create(
            name=COLLECTION_NAME,
            # Vectorizer.none() vì ta tự cung cấp vector từ bge-m3, không cần Weaviate tự embed lại.
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        # batch.dynamic() tự điều chỉnh kích thước batch theo throughput server, hiệu quả hơn fixed batch khi số chunk biến động.
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                meta = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": meta["source"],
                        "doc_type": meta["type"],
                        "chunk_index": meta["chunk_index"],
                    },
                    vector=chunk["embedding"],
                )

        count = collection.aggregate.over_all(total_count=True).total_count
        print(f"  Collection '{COLLECTION_NAME}': {count} objects indexed")
        print(f"  Persisted at: {WEAVIATE_PERSIST_DIR}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
