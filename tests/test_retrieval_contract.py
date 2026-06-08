from unittest.mock import patch

from src.task8_pageindex_vectorless import _extract_nodes
from src.task9_retrieval_pipeline import retrieve


def test_fallback_is_used_below_threshold():
    weak = [{"content": "weak", "score": 0.29, "metadata": {}}]
    pageindex = [{
        "content": "fallback",
        "score": 0.8,
        "metadata": {"source": "law.pdf", "source_path": "data/law.pdf", "type": "legal"},
        "source": "pageindex",
    }]
    with (
        patch("src.task9_retrieval_pipeline.semantic_search", return_value=weak),
        patch("src.task9_retrieval_pipeline.lexical_search", return_value=[]),
        patch("src.task9_retrieval_pipeline.rerank_rrf", return_value=weak),
        patch("src.task9_retrieval_pipeline.rerank", return_value=weak),
        patch("src.task9_retrieval_pipeline.pageindex_search", return_value=pageindex) as fallback,
    ):
        results = retrieve("query", score_threshold=0.3)

    fallback.assert_called_once_with("query", top_k=5)
    assert results[0]["source"] == "pageindex"


def test_retrieve_returns_ui_metadata():
    strong = [{"content": "answer", "score": 0.9, "metadata": {"source": "law.md"}}]
    with (
        patch("src.task9_retrieval_pipeline.semantic_search", return_value=strong),
        patch("src.task9_retrieval_pipeline.lexical_search", return_value=[]),
        patch("src.task9_retrieval_pipeline.rerank_rrf", return_value=strong),
        patch("src.task9_retrieval_pipeline.rerank", return_value=strong),
    ):
        result = retrieve("query", top_k=1)[0]

    assert set(("source", "source_path", "type")) <= result["metadata"].keys()


def test_pageindex_real_schema_is_flattened():
    raw = {
        "status": "completed",
        "retrieved_nodes": [{
            "id": "0007",
            "title": "Chương II",
            "relevant_contents": [[{
                "section_title": "Điều 6",
                "physical_index": "<physical_index_4>",
                "relevant_content": "Trách nhiệm của cá nhân, gia đình.",
            }]],
        }],
    }
    document = {
        "source": "law.pdf",
        "source_path": "data/landing/legal/law.pdf",
        "type": "legal",
    }

    result = _extract_nodes(raw, top_k=1, document=document)[0]

    assert result["content"] == "Trách nhiệm của cá nhân, gia đình."
    assert result["score"] == 1.0
    assert result["metadata"]["source_path"].endswith("law.pdf")
