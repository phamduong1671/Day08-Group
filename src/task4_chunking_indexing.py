"""
Task 4 — Chunking & Indexing vào Vector Store (Weaviate).

Lựa chọn & lý do:
    - Chunking: RecursiveCharacterTextSplitter. An toàn, tôn trọng ranh giới đoạn/câu
      (\n\n → \n → ". "), phù hợp cả văn bản luật (Điều/Khoản) lẫn bài báo.
    - chunk_size=900 ký tự: đủ ngắn để retrieval/reranking nhanh nhưng vẫn giữ
      được một điều luật hoặc đoạn tin tương đối trọn vẹn.
    - chunk_overlap=120: giữ liền mạch ngữ nghĩa qua ranh giới chunk.
    - Embedding: BAAI/bge-m3 (1024-dim, multilingual) — mạnh cho tiếng Việt pháp lý.
    - Vector store: Weaviate (vectorizer=none, ta tự đẩy vector), hỗ trợ scale + hybrid.

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
    docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).parent.parent
LOCAL_INDEX_DIR = PROJECT_DIR / "data" / "indexes"
LOCAL_INDEX_PATH = LOCAL_INDEX_DIR / "drug_law_chunks.jsonl"
LOCAL_INDEX_METADATA_PATH = LOCAL_INDEX_DIR / "drug_law_chunks.metadata.json"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass

from .config import COLLECTION_NAME, EMBEDDING_DIM, EMBEDDING_MODEL, STANDARDIZED_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"
VECTOR_STORE = "weaviate"
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "0") == "1"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def _infer_doc_type(md_file: Path) -> str:
    parts = set(md_file.relative_to(STANDARDIZED_DIR).parts)
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(PROJECT_DIR).as_posix()
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path,
                    "type": _infer_doc_type(md_file),
                    "document_id": md_file.stem,
                },
            }
        )

    return documents


def _fallback_split_text(text: str) -> list[str]:
    chunks: list[str] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0

    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def _build_text_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n## ",
            "\n# ",
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk.
    """
    splitter = _build_text_splitter()
    chunks: list[dict] = []

    for doc_index, doc in enumerate(documents):
        content = str(doc.get("content", "")).strip()
        if not content:
            continue

        splits = splitter.split_text(content) if splitter else _fallback_split_text(content)
        metadata = dict(doc.get("metadata", {}))

        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            source = metadata.get("source", f"doc_{doc_index}")
            raw_id = f"{source}:{chunk_index}:{chunk_text[:80]}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

            chunks.append(
                {
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "chunking_method": CHUNKING_METHOD,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )

    return chunks


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Deterministic local fallback when BAAI/bge-m3 is not cached/downloadable.
    """
    vector = [0.0] * dim
    tokens = text.lower().split()

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _load_bge_m3_model():
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            EMBEDDING_MODEL,
            local_files_only=not ALLOW_MODEL_DOWNLOAD,
            trust_remote_code=True,
        )
    except TypeError:
        if not ALLOW_MODEL_DOWNLOAD:
            raise
        return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)


def _clean_env_value(name: str) -> str:
    value = os.getenv(name, "")
    return value.strip().strip("\"'")


def _get_weaviate_cloud_url() -> str:
    cluster_url = _clean_env_value("WEAVIATE_URL") or _clean_env_value("WEAVIATE_CLUSTER_URL")
    if not cluster_url:
        return ""

    if not cluster_url.startswith(("http://", "https://")):
        cluster_url = f"https://{cluster_url}"

    return cluster_url.rstrip("/")


def _get_weaviate_api_key() -> str:
    api_key = _clean_env_value("WEAVIATE_API_KEY")
    placeholders = {"api_key_cua_ban", "your-api-key", "your_api_key", "YOUR_API_KEY"}

    if api_key in placeholders:
        raise ValueError("WEAVIATE_API_KEY trong .env vẫn là placeholder, hãy thay bằng API key thật")

    return api_key


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    texts = [chunk["content"] for chunk in chunks]

    try:
        model = _load_bge_m3_model()
        embeddings = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            chunk["metadata"]["embedding_model"] = EMBEDDING_MODEL
            chunk["metadata"]["embedding_dim"] = EMBEDDING_DIM

    except Exception as exc:
        print(f"! Không load được {EMBEDDING_MODEL}; dùng local hash embedding ({type(exc).__name__})")
        for chunk in chunks:
            chunk["embedding"] = _hash_embedding(chunk["content"])
            chunk["metadata"]["embedding_model"] = f"{EMBEDDING_MODEL} (hash-fallback)"
            chunk["metadata"]["embedding_dim"] = EMBEDDING_DIM

    return chunks


def _connect_weaviate():
    import weaviate

    cluster_url = _get_weaviate_cloud_url()

    if cluster_url:
        from weaviate.classes.init import Auth

        api_key = _get_weaviate_api_key()
        if not api_key:
            raise ValueError("Đã có WEAVIATE_URL/WEAVIATE_CLUSTER_URL nhưng thiếu WEAVIATE_API_KEY")

        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=Auth.api_key(api_key),
        )

    host = _clean_env_value("WEAVIATE_HOST") or "localhost"
    port = int(_clean_env_value("WEAVIATE_PORT") or "8080")
    grpc_port = int(_clean_env_value("WEAVIATE_GRPC_PORT") or "50051")
    return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)


def _write_index_metadata(metadata: dict[str, Any]) -> None:
    LOCAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_INDEX_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _index_to_weaviate(chunks: list[dict]) -> dict[str, Any]:
    from weaviate.classes.config import Configure, DataType, Property

    client = _connect_weaviate()
    try:
        if not client.is_ready():
            raise ConnectionError("Weaviate client is not ready")

        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)

        collection = client.collections.create(
            name=COLLECTION_NAME,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="source_path", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "source_path": metadata.get("source_path", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_id": metadata.get("chunk_id", chunk.get("id", "")),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                    },
                    vector=chunk["embedding"],
                )

        metadata = {
            "backend": "weaviate",
            "collection": COLLECTION_NAME,
            "indexed_chunks": len(chunks),
            "embedding_model": chunks[0]["metadata"].get("embedding_model") if chunks else EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "vector_store_requested": VECTOR_STORE,
            "weaviate_target": "cloud" if _get_weaviate_cloud_url() else "local",
        }
        _write_index_metadata(metadata)
        return metadata
    finally:
        client.close()


def _index_to_local_jsonl(chunks: list[dict], reason: str) -> dict[str, Any]:
    LOCAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with LOCAL_INDEX_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    metadata = {
        "backend": "local_jsonl_fallback",
        "fallback_reason": reason,
        "index_path": LOCAL_INDEX_PATH.relative_to(PROJECT_DIR).as_posix(),
        "indexed_chunks": len(chunks),
        "embedding_model": chunks[0]["metadata"].get("embedding_model") if chunks else EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "vector_store_requested": VECTOR_STORE,
        "weaviate_collection": COLLECTION_NAME,
    }
    _write_index_metadata(metadata)
    return metadata


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        return _index_to_local_jsonl([], "Không có chunk để index")

    missing_embeddings = [chunk for chunk in chunks if "embedding" not in chunk]
    if missing_embeddings:
        raise ValueError("Chunks cần được embed trước khi index_to_vectorstore()")

    try:
        result = _index_to_weaviate(chunks)
        print(f"✓ Indexed {result['indexed_chunks']} chunks to Weaviate/{COLLECTION_NAME}")
        return result
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"! Không kết nối/index được Weaviate; lưu local fallback ({reason})")
        result = _index_to_local_jsonl(chunks, reason)
        print(f"✓ Indexed {result['indexed_chunks']} chunks to {result['index_path']}")
        return result


def run_pipeline() -> None:
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    target = "Weaviate Cloud" if _get_weaviate_cloud_url() else "Weaviate local"

    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} ({target})")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("⚠ Chưa có markdown trong data/standardized/ — hãy chạy Task 1-3 trước.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    result = index_to_vectorstore(chunks)
    print(f"✓ Indexed to vector store backend: {result['backend']}")


if __name__ == "__main__":
    run_pipeline()
