"""
Task 5 — Semantic Search Module (dense retrieval).

==============================================================================
THIẾT KẾ — tầng RECALL ngữ nghĩa của pipeline 2 tầng (recall -> precision).
==============================================================================
Tìm kiếm dense trên Weaviate bằng đúng embedder bge-m3 đã index ở Task 4
(không nạp lại model, không tách không gian vector). Trả chunk + score cosine
+ metadata giàu (dieu/dieu_title/chuong) để Task 7 rerank và Task 10 trích dẫn.

Vì sao dense: bắt được tương đồng NGỮ NGHĨA — query "hình phạt tàng trữ" khớp
"Điều 249. Tội ..." dù không trùng từ. Điểm yếu (định danh chính xác như
"57/2022/NĐ-CP") được Task 6 (BM25) bù lại; Task 9 hợp nhất hai bên.

Lưu ý: bge-m3 KHÔNG cần instruction prefix cho query (khác họ e5) -> embed
query y hệt chunk để cùng không gian.
"""

from .task4_chunking_indexing import (
    COLLECTION_NAME,
    _get_embedder,
    connect_weaviate,
)

# Các field metadata trả về cho citation/rerank ở các task sau.
_META_FIELDS = ("source", "doc_type", "doc_title", "chuong", "dieu",
                "dieu_title", "chunk_id", "chunk_index")


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm ngữ nghĩa bằng vector similarity (cosine) trên Weaviate.

    Args:
        query: Câu truy vấn.
        top_k: Số kết quả tối đa.

    Returns:
        List of {'content', 'score', 'metadata'} sorted by score giảm dần.
        score = 1 - cosine_distance ∈ [0, 1] (vector đã normalize ở Task 4).
    """
    from weaviate.classes.query import MetadataQuery

    try:
        # Embed query bằng đúng model đã index (normalize -> cùng không gian cosine).
        vec = _get_embedder().encode([query], normalize_embeddings=True)[0].tolist()

        client = connect_weaviate()
        try:
            col = client.collections.get(COLLECTION_NAME)
            res = col.query.near_vector(
                near_vector=vec,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
            )
            results = [
                {
                    "content": o.properties["content"],
                    "score": 1.0 - o.metadata.distance,   # distance -> similarity
                    "metadata": {k: o.properties.get(k) for k in _META_FIELDS},
                }
                for o in res.objects
            ]
        finally:
            client.close()
    except Exception as exc:
        print(f"  ⚠ Semantic search unavailable: {exc}")
        return []

    # near_vector đã trả theo distance tăng dần; ép sort để chắc chắn giảm dần.
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


if __name__ == "__main__":
    for q in ["hình phạt cho tội tàng trữ ma tuý",
              "cai nghiện ma tuý tự nguyện"]:
        print(f"\nQuery: {q}\n" + "-" * 60)
        for r in semantic_search(q, top_k=5):
            m = r["metadata"]
            tag = f"Điều {m['dieu']}" if m.get("dieu") else m.get("source", "")
            print(f"[{r['score']:.3f}] ({tag}) {r['content'][:90]}...")
