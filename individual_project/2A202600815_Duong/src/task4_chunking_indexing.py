"""Task 4 - Load, chunk, embed, and index markdown documents locally."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "indexes"
CHUNKS_PATH = INDEX_DIR / "drug_law_chunks.jsonl"
METADATA_PATH = INDEX_DIR / "drug_law_chunks.metadata.json"

# Recursive character chunking is stable for mixed legal/news markdown. 500 keeps
# snippets readable in citations; 50 overlap preserves article/legal continuity.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# The intended embedding model for production is bge-m3. In this local exercise
# we use deterministic hashing vectors so tests work without downloading models.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 256
VECTOR_STORE = "local_jsonl"

TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 2]


def hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def load_documents() -> list[dict]:
    """Read all markdown files from data/standardized/."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        relative = md_file.relative_to(PROJECT_DIR).as_posix()
        doc_type = "legal" if "/legal/" in f"/{relative}" else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative,
                    "type": doc_type,
                },
            }
        )
    return documents


def _split_text(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        window = text[start:end]
        if end < len(text):
            split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "), window.rfind(" "))
            if split_at > CHUNK_SIZE * 0.45:
                end = start + split_at + 1
                window = text[start:end]
        chunk = window.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents with recursive character splitting."""
    chunks: list[dict] = []
    for doc in documents:
        splits = _split_text(doc["content"])
        for index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": index,
                        "chunk_id": f"{doc.get('metadata', {}).get('source_path', 'doc')}#{index}",
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach deterministic normalized vectors to chunks."""
    embedded: list[dict] = []
    for chunk in chunks:
        item = dict(chunk)
        item["metadata"] = dict(chunk.get("metadata", {}))
        item["embedding"] = hash_embedding(chunk.get("content", ""))
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """Persist chunks to a local JSONL vector store."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "chunk_count": len(chunks),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
                "vector_store": VECTOR_STORE,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return CHUNKS_PATH


def ensure_index() -> list[dict]:
    """Load index if present; otherwise build it from standardized markdown."""
    if CHUNKS_PATH.exists():
        chunks: list[dict] = []
        with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(json.loads(line))
        if chunks:
            return chunks

    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    if chunks:
        index_to_vectorstore(chunks)
    return chunks


def run_pipeline() -> None:
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"Chunking: {CHUNKING_METHOD} size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")
    print(f"Embedding: {EMBEDDING_MODEL} via local hashing dim={EMBEDDING_DIM}")
    print(f"Vector store: {VECTOR_STORE}")
    print("=" * 50)
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    embedded = embed_chunks(chunks)
    path = index_to_vectorstore(embedded)
    print(f"Indexed to {path}")


if __name__ == "__main__":
    run_pipeline()
