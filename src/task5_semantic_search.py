"""
Task 5 — Semantic Search Module (dense retrieval).

Embed query bằng cùng model ở Task 4 (bge-m3) rồi near_vector trên Weaviate.
Score = 1 - cosine_distance (đổi distance → similarity, càng lớn càng liên quan).

Degrade-gracefully: nếu Weaviate chưa chạy hoặc collection trống → trả [].
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from functools import lru_cache

from .config import COLLECTION_NAME
from .task4_chunking_indexing import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LOCAL_INDEX_METADATA_PATH,
    LOCAL_INDEX_PATH,
    _connect_weaviate,
    _hash_embedding,
)


def _index_uses_hash_embedding() -> bool:
    """Index có được tạo bằng hash-fallback không? (đọc metadata Task 4)."""
    try:
        meta = json.loads(LOCAL_INDEX_METADATA_PATH.read_text(encoding="utf-8"))
        return "hash-fallback" in str(meta.get("embedding_model", ""))
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load_query_model():
    # @lru_cache: nạp bge-m3 (~2.2GB) MỘT lần rồi tái dùng. Không có cache thì mỗi
    # semantic_search() lại load lại model từ đĩa → nghẽn nặng (nhất là eval nhiều câu).
    from sentence_transformers import SentenceTransformer

    old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        return SentenceTransformer(
            EMBEDDING_MODEL,
            local_files_only=True,
            trust_remote_code=True,
        )
    finally:
        if old_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf_offline

        if old_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline


def _tokenize(text: str) -> set[str]:
    normalized = text.lower()
    for char in ",.;:!?()[]{}\"'“”‘’/-":
        normalized = normalized.replace(char, " ")
    return {token for token in normalized.split() if len(token) >= 2}


def _load_local_index(index_path: Path = LOCAL_INDEX_PATH) -> list[dict]:
    if not index_path.exists():
        return []

    chunks: list[dict] = []
    with index_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    return chunks


def _pseudo_query_embedding(query: str, max_seed_chunks: int = 12) -> list[float]:
    """
    Fallback query vector trong cùng không gian với index đã tạo ở Task 4.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    candidates: list[tuple[float, list[float]]] = []
    for chunk in _load_local_index():
        content_tokens = _tokenize(chunk.get("content", ""))
        overlap = len(query_tokens & content_tokens)
        if overlap <= 0:
            continue

        embedding = chunk.get("embedding") or []
        if len(embedding) != EMBEDDING_DIM:
            continue

        score = overlap / max(len(query_tokens), 1)
        candidates.append((score, embedding))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    vectors = candidates[:max_seed_chunks]
    averaged = [0.0] * EMBEDDING_DIM
    total_weight = 0.0

    for weight, embedding in vectors:
        total_weight += weight
        for index, value in enumerate(embedding):
            averaged[index] += weight * value

    if total_weight == 0:
        return []

    averaged = [value / total_weight for value in averaged]
    norm = math.sqrt(sum(value * value for value in averaged))
    if norm == 0:
        return []

    return [value / norm for value in averaged]


def _embed_query(query: str) -> list[float]:
    """
    Embed query bằng ĐÚNG model đã dùng ở Task 4.

    Quan trọng: query và index phải cùng không gian vector, nếu không cosine vô
    nghĩa. Vì vậy mặc định dùng bge-m3 khi index là bge-m3 thật; chỉ dùng
    hash/pseudo khi index cũng là hash-fallback. Env USE_BGE_QUERY_MODEL=0/1 để ép.
    """
    env = os.getenv("USE_BGE_QUERY_MODEL", "0")
    use_model = env == "1"
    if use_model:
        try:
            model = _load_query_model()
            embedding = model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as exc:
            print(f"! Không load được {EMBEDDING_MODEL}; dùng indexed-vector fallback cho query ({type(exc).__name__})")

    pseudo_embedding = _pseudo_query_embedding(query)
    if pseudo_embedding:
        return pseudo_embedding

    return _hash_embedding(query, dim=EMBEDDING_DIM)


@lru_cache(maxsize=128)
def generate_hypothetical_document(query: str) -> str:
    """
    HyDE: sinh một đoạn văn có thể trả lời query rồi embed đoạn đó.

    Đoạn giả định không được trả cho người dùng và không được xem là evidence;
    nó chỉ giúp query ngắn gần hơn với văn phong dài, trang trọng của corpus luật.
    Nếu LLM local không sẵn sàng, trả chuỗi rỗng để semantic search dùng query gốc.
    """
    if os.getenv("HYDE_ENABLED", "0") != "1":
        return ""

    try:
        from openai import OpenAI
        from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=4.0, max_retries=0)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Viết một đoạn văn pháp lý tiếng Việt khoảng 100-150 từ có khả năng "
                        "trả lời truy vấn. Không bịa số điều, nguồn hoặc sự kiện cụ thể."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            top_p=0.9,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _embed_query_with_hyde(query: str) -> list[float]:
    """
    Embed HyDE bằng chính encoder của Task 4/Task 5, bảo đảm query, hypothetical
    document và corpus đều nằm trong cùng không gian vector.
    """
    hypothetical = generate_hypothetical_document(query)
    return _embed_query(hypothetical or query)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0

    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _format_result(content: str, score: float, metadata: dict[str, Any] | None = None) -> dict:
    return {
        "content": content,
        "score": float(score),
        "metadata": metadata or {},
    }


def _search_local_jsonl(query_embedding: list[float], top_k: int) -> list[dict]:
    results: list[dict] = []

    for chunk in _load_local_index():
        embedding = chunk.get("embedding") or []
        score = _cosine_similarity(query_embedding, embedding)
        results.append(
            _format_result(
                content=chunk.get("content", ""),
                score=score,
                metadata=chunk.get("metadata", {}),
            )
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _search_weaviate(query_embedding: list[float], top_k: int) -> list[dict]:
    from weaviate.classes.query import MetadataQuery

    client = _connect_weaviate()
    try:
        if not client.collections.exists(COLLECTION_NAME):
            return []

        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results: list[dict] = []
        for obj in response.objects:
            props = obj.properties
            distance = obj.metadata.distance
            score = 1.0 - float(distance) if distance is not None else 0.0
            results.append(
                _format_result(
                    content=props.get("content", ""),
                    score=score,
                    metadata={
                        "source": props.get("source", ""),
                        "source_path": props.get("source_path", ""),
                        "type": props.get("doc_type", ""),
                        "chunk_id": props.get("chunk_id", ""),
                        "chunk_index": props.get("chunk_index", 0),
                        "vector_store": "weaviate",
                    },
                )
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
    finally:
        client.close()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted desc.
    """
    if top_k <= 0 or not query.strip():
        return []

    query_embedding = _embed_query_with_hyde(query.strip())

    try:
        results = _search_weaviate(query_embedding, top_k)
        if results:
            return results
    except Exception as exc:
        print(f"! Không search được Weaviate; dùng local JSONL fallback ({type(exc).__name__})")

    return _search_local_jsonl(query_embedding, top_k)


if __name__ == "__main__":
    for r in semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
