"""
Task 5 - Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from pathlib import Path

# Must stay in sync with Task 4 constants - changing one without the other
# will silently return wrong results (dimension mismatch or wrong collection).
_EMBEDDING_MODEL = "BAAI/bge-m3"
_COLLECTION_NAME = "DrugRelatedDocs"
_WEAVIATE_PERSIST_DIR = str(
    Path(__file__).parent.parent / "data" / "weaviate_store"
)

# Module-level cache so repeated calls within the same process don't reload
# the 1 GB bge-m3 model from disk each time.
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import weaviate
    from weaviate.classes.query import MetadataQuery

    # Step 1: embed query với cùng model đã dùng lúc index - bất đồng model
    # sẽ sinh vector trong không gian khác, khiến kết quả vô nghĩa.
    model = _get_model()
    query_vector = model.encode(query).tolist()

    # Step 2: kết nối lại Weaviate embedded với cùng persist path để đọc
    # data đã index ở Task 4, không cần index lại.
    with weaviate.connect_to_embedded(
        persistence_data_path=_WEAVIATE_PERSIST_DIR,
    ) as client:
        collection = client.collections.get(_COLLECTION_NAME)

        # near_vector thực hiện cosine similarity search trên HNSW index
        # đã được build khi index; distance=True yêu cầu Weaviate trả về
        # khoảng cách để ta tính similarity score.
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

    # Step 3: chuyển distance -> similarity và chuẩn hoá output format.
    # Weaviate dùng cosine distance ∈ [0, 2]; score = 1 - distance cho
    # kết quả ∈ [-1, 1] tương tự cosine similarity thông thường.
    results = [
        {
            "content": obj.properties["content"],
            "score": 1 - obj.metadata.distance,
            "metadata": {
                "source": obj.properties.get("source", ""),
                "doc_type": obj.properties.get("doc_type", ""),
                "chunk_index": obj.properties.get("chunk_index", -1),
            },
        }
        for obj in response.objects
    ]

    # Weaviate đã sắp xếp theo distance tăng dần (= similarity giảm dần),
    # sort lại để đảm bảo contract "score descending" dù backend thay đổi.
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "nghệ sĩ bị bắt vì sử dụng chất cấm",
        "danh mục các chất ma tuý bị cấm",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        results = semantic_search(query, top_k=5)
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"[{i}] score={r['score']:.4f} | {meta['doc_type']} | {meta['source']}")
            print(f"    {r['content'][:120].strip()}...")
