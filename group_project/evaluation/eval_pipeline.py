"""
RAG Evaluation Pipeline — DeepEval.

Đánh giá chất lượng RAG pipeline của nhóm trên golden dataset, chạy A/B giữa
2 config (hybrid + rerank  vs  hybrid không rerank) và export báo cáo ra results.md.

Judge model: gpt-4o-mini (OpenAI) — set OPENAI_API_KEY trong .env ở root project.
Pipeline được đánh giá: src.task10.generate_with_citation (Ollama qwen2.5:7b + Weaviate).

Framework đã chọn: DeepEval. Lý do: 4 metric yêu cầu
(faithfulness, relevance, context_recall, context_precision) ánh xạ
1-1 sang các metric có sẵn của DeepEval, và evaluate() chạy được offline
trong script Python (không bắt buộc cloud).

4 metrics (DeepEval, threshold 0.7):
    - Faithfulness        : answer có bám đúng retrieval_context không (chống hallucinate)
    - Answer Relevancy    : answer có trả lời đúng câu hỏi không
    - Contextual Recall   : retriever có lấy đủ evidence so với expected_answer không
    - Contextual Precision : trong context lấy về, phần liên quan có được xếp lên đầu không


Yêu cầu:
    1. Load golden_dataset.json (>=15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Cài đặt:
    pip install deepeval
    
    
Cách chạy (từ root project):
    .venv/bin/python -m group_project.evaluation.eval_pipeline
        # smoke test mặc định 2 câu đầu trong golden dataset

    EVAL_LIMIT=8 .venv/bin/python -m group_project.evaluation.eval_pipeline
        # chạy thử 8 câu

    EVAL_LIMIT=0 .venv/bin/python -m group_project.evaluation.eval_pipeline
        # chạy toàn bộ golden dataset khi bộ mới đã chốt

Cấu hình model làm "judge" (mặc định DeepEval dùng OpenAI):
    export OPENAI_API_KEY=...                # dùng OpenAI
  hoặc dùng model khác qua LiteLLM:
    export DEEPEVAL_JUDGE_MODEL=anthropic/claude-3-5-sonnet-latest
    export ANTHROPIC_API_KEY=...

Giao diện RAG pipeline mà file này kỳ vọng (bạn cắm pipeline thật vào):
    pipeline.generate_with_citation(question: str) -> {
        "answer": str,
        "sources": [{"content": str, ...}, ...],
    }
  và (tùy chọn, cho phần A/B) một trong hai:
    - pipeline.configure(**params)          # áp dụng config
    - các thuộc tính settable: pipeline.use_reranking, pipeline.alpha, ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Generation đã chuyển sang OpenAI (gpt-4o-mini) nên pipeline không cần GPU.
# Ép embedding (bge-m3) + reranker chạy CPU để tránh CUDA OOM trên VRAM 6GB.
# Đặt trước mọi import torch/sentence-transformers. Override bằng cách set sẵn env.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Cho phép import `src.*` khi chạy bằng `python eval_pipeline.py` trực tiếp.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
THRESHOLD = 0.7
EVAL_LIMIT_RAW = os.getenv("EVAL_LIMIT", "2")
EVAL_LIMIT = int(EVAL_LIMIT_RAW) if EVAL_LIMIT_RAW.strip() else 2
EVAL_LIMIT = None if EVAL_LIMIT == 0 else EVAL_LIMIT  # 0 = chạy hết
EVAL_WORKERS_RAW = os.getenv("EVAL_WORKERS", "1")
EVAL_WORKERS = max(1, int(EVAL_WORKERS_RAW) if EVAL_WORKERS_RAW.strip() else 1)

# Tên metric -> nhãn hiển thị (giữ thứ tự ổn định cho mọi bảng).
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "contextual_recall": "Context Recall",
    "contextual_precision": "Context Precision",
}

# A/B configs:
# - A dùng score cross-encoder đã sigmoid nên threshold 0.3 có ý nghĩa.
# - B dùng RRF score nhỏ (~1/(60+rank)); threshold 0 tránh biến no-rerank
#   thành PageIndex fallback, giúp đo đúng tác động của reranking.
CONFIGS = {
    "A_hybrid_rerank": {
        "use_reranking": True,
        "score_threshold": 0.3,
        "label": "Hybrid + Rerank",
    },
    "B_no_rerank": {
        "use_reranking": False,
        "score_threshold": 0.0,
        "label": "Hybrid, no Rerank",
    },
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:EVAL_LIMIT] if EVAL_LIMIT else data


# =============================================================================
# Judge + Metrics
# =============================================================================

def get_judge():
    """LLM judge dùng chung cho mọi metric (OpenAI gpt-4o-mini)."""
    from deepeval.models import GPTModel

    return GPTModel(model=JUDGE_MODEL)


def build_metrics(judge):
    """Tạo bộ 4 metric mới (metric giữ state nội bộ → mỗi lần đo cần instance sạch)."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    return {
        "faithfulness": FaithfulnessMetric(threshold=THRESHOLD, model=judge),
        "answer_relevancy": AnswerRelevancyMetric(threshold=THRESHOLD, model=judge),
        "contextual_recall": ContextualRecallMetric(threshold=THRESHOLD, model=judge),
        "contextual_precision": ContextualPrecisionMetric(threshold=THRESHOLD, model=judge),
    }


# =============================================================================
# Chạy pipeline + chấm điểm cho 1 config
# =============================================================================

def score_config(
    golden_dataset: list[dict],
    use_reranking: bool,
    score_threshold: float,
    judge,
) -> list[dict]:
    """
    Với mỗi câu hỏi: chạy RAG pipeline rồi đo 4 metric.

    Returns:
        list rows: {id, question, doc, difficulty, faithfulness, answer_relevancy,
                    contextual_recall, contextual_precision}  (score None nếu lỗi)
    """
    total = len(golden_dataset)
    if EVAL_WORKERS == 1:
        rows = []
        for i, item in enumerate(golden_dataset, 1):
            rows.append(
                score_item(
                    i,
                    total,
                    item,
                    use_reranking=use_reranking,
                    score_threshold=score_threshold,
                    judge=judge,
                )
            )
        return rows

    print(f"  Parallel mode: {EVAL_WORKERS} workers", flush=True)
    rows_by_index: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        future_map = {
            executor.submit(
                score_item,
                i,
                total,
                item,
                use_reranking,
                score_threshold,
                judge,
            ): i
            for i, item in enumerate(golden_dataset, 1)
        }
        for done_count, future in enumerate(as_completed(future_map), 1):
            i = future_map[future]
            try:
                rows_by_index[i] = future.result()
            except Exception as e:
                item = golden_dataset[i - 1]
                print(f"      ⚠ worker error [{i}/{total}] {item.get('id', '')}: {e}", flush=True)
                rows_by_index[i] = {**_meta(item), **{m: None for m in METRIC_LABELS}}
            print(f"  Completed {done_count}/{total}", flush=True)
    return [rows_by_index[i] for i in range(1, total + 1)]


def score_item(
    i: int,
    total: int,
    item: dict,
    use_reranking: bool,
    score_threshold: float,
    judge,
) -> dict:
    """Chạy RAG + 4 metrics cho một câu hỏi."""
    from deepeval.test_case import LLMTestCase
    from src.task10_generation import generate_with_citation

    question = item["question"]
    print(f"  [{i}/{total}] {item.get('id', '')} {question[:60]}...", flush=True)

    try:
        result = generate_with_citation(
            question,
            use_reranking=use_reranking,
            score_threshold=score_threshold,
        )
        retrieval_context = [c["content"] for c in result.get("sources", [])]
        if not retrieval_context:
            retrieval_context = ["(không có context được truy hồi)"]
        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=item["expected_answer"],
            retrieval_context=retrieval_context,
        )
    except Exception as e:  # pipeline lỗi (Ollama/Weaviate) → ghi None, không dừng
        print(f"      ⚠ pipeline error [{item.get('id', '')}]: {e}", flush=True)
        return {**_meta(item), **{m: None for m in METRIC_LABELS}}

    row = {**_meta(item)}
    for name, metric in build_metrics(judge).items():
        try:
            metric.measure(test_case)
            row[name] = round(float(metric.score), 3)
        except Exception as e:
            print(f"      ⚠ metric {name} error [{item.get('id', '')}]: {e}", flush=True)
            row[name] = None
    return row


def _meta(item: dict) -> dict:
    return {
        "id": item.get("id", ""),
        "question": item["question"],
        "doc": item.get("doc", ""),
        "difficulty": item.get("difficulty", ""),
    }


# =============================================================================
# Aggregation
# =============================================================================

def averages(rows: list[dict]) -> dict:
    """Trung bình mỗi metric (bỏ qua None)."""
    out = {}
    for name in METRIC_LABELS:
        vals = [r[name] for r in rows if r.get(name) is not None]
        out[name] = round(sum(vals) / len(vals), 3) if vals else None
    scored = [r for r in rows if any(r.get(m) is not None for m in METRIC_LABELS)]
    metric_vals = [out[m] for m in METRIC_LABELS if out[m] is not None]
    out["overall"] = round(sum(metric_vals) / len(metric_vals), 3) if metric_vals else None
    out["n_scored"] = len(scored)
    return out


def row_mean(row: dict) -> float | None:
    vals = [row[m] for m in METRIC_LABELS if row.get(m) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def worst_performers(rows: list[dict], n: int = 3) -> list[dict]:
    """Bottom-N câu theo trung bình 4 metric."""
    scored = [r for r in rows if row_mean(r) is not None]
    return sorted(scored, key=row_mean)[:n]


# =============================================================================
# Export Results
# =============================================================================

def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def export_results(results: dict[str, list[dict]], elapsed: float):
    """Ghi báo cáo Markdown: overall A/B, phân tích, worst performers, đề xuất."""
    aggs = {cfg: averages(rows) for cfg, rows in results.items()}
    a, b = "A_hybrid_rerank", "B_no_rerank"
    agg_a, agg_b = aggs[a], aggs[b]

    lines = []
    lines.append("# RAG Evaluation Results\n")
    lines.append("## Framework sử dụng\n")
    lines.append(
        f"- **Framework:** DeepEval `{_deepeval_version()}`\n"
        f"- **Judge model:** OpenAI `{JUDGE_MODEL}`\n"
        f"- **Pipeline đánh giá:** `src.task10.generate_with_citation` "
        f"(retrieval Weaviate hybrid + generation `{_gen_model()}`)\n"
        f"- **Golden dataset:** {agg_a['n_scored']} câu chấm thành công"
        f"{f' (giới hạn EVAL_LIMIT={EVAL_LIMIT})' if EVAL_LIMIT else ''}\n"
        f"- **Evaluation mode:** {'Smoke test sample' if EVAL_LIMIT else 'Full dataset'}\n"
        f"- **Workers:** {EVAL_WORKERS}\n"
        f"- **Threshold pass:** {THRESHOLD} · **Thời gian chạy:** {elapsed/60:.1f} phút\n"
    )

    # Overall A/B table
    lines.append("\n---\n\n## Overall Scores (A/B)\n")
    lines.append("| Metric | Config A (hybrid + rerank) | Config B (no rerank) | Δ (A − B) |")
    lines.append("|--------|---------------------------|----------------------|-----------|")
    for name, label in METRIC_LABELS.items():
        va, vb = agg_a[name], agg_b[name]
        delta = round(va - vb, 3) if (va is not None and vb is not None) else None
        lines.append(f"| {label} | {_fmt(va)} | {_fmt(vb)} | {_fmt(delta)} |")
    da = agg_a["overall"]
    db = agg_b["overall"]
    dd = round(da - db, 3) if (da is not None and db is not None) else None
    lines.append(f"| **Average** | **{_fmt(da)}** | **{_fmt(db)}** | **{_fmt(dd)}** |")

    # A/B analysis
    lines.append("\n---\n\n## A/B Comparison Analysis\n")
    lines.append(
        "**Config A — Hybrid + Rerank:** semantic (bge-m3) + BM25 hợp nhất bằng RRF, "
        "sau đó cross-encoder `bge-reranker-v2-m3` chấm lại top kết quả.\n"
    )
    lines.append(
        "\n**Config B — Hybrid, no Rerank:** giống A nhưng bỏ bước cross-encoder; "
        "lấy trực tiếp thứ hạng sau RRF; không dùng threshold fallback theo score "
        "vì RRF score có thang đo khác cross-encoder.\n"
    )
    winner = "A (có rerank)" if (dd or 0) > 0 else ("B (không rerank)" if (dd or 0) < 0 else "ngang nhau")
    lines.append(
        f"\n**Kết luận:** Config tốt hơn theo điểm trung bình là **{winner}** "
        f"(Δ average = {_fmt(dd)}). "
        "Reranking thường nâng Context Precision rõ nhất vì nó đẩy chunk liên quan lên "
        "đầu; nếu Δ nhỏ, retrieval gốc đã đủ tốt cho corpus pháp luật có cấu trúc rõ.\n"
    )

    # Worst performers (dựa trên config A)
    lines.append("\n---\n\n## Worst Performers (Bottom 3 — Config A)\n")
    lines.append("| # | ID | Question | Faith | Relev | Recall | Prec | Avg |")
    lines.append("|---|----|----------|-------|-------|--------|------|-----|")
    for idx, r in enumerate(worst_performers(results[a], 3), 1):
        q = r["question"][:70].replace("|", "/")
        lines.append(
            f"| {idx} | {r['id']} | {q} | {_fmt(r.get('faithfulness'))} | "
            f"{_fmt(r.get('answer_relevancy'))} | {_fmt(r.get('contextual_recall'))} | "
            f"{_fmt(r.get('contextual_precision'))} | {_fmt(row_mean(r))} |"
        )
    failing = [r for r in results[a] if (row_mean(r) or 0) < THRESHOLD]
    if failing:
        lines.append(
            "\n**Phân tích root cause (gợi ý):** câu điểm thấp thường thuộc loại "
            "`comparison`/`cross_reference`/`scenario` (cần tổng hợp nhiều điều luật) hoặc "
            "câu mà `expected_answer` còn ghi chú `note: đối chiếu văn bản gốc` (ground truth "
            "chưa chốt số liệu) → kéo Recall/Faithfulness xuống. Cần kiểm tra: lỗi ở khâu "
            "**retrieval** (không lấy đúng điều luật) hay **generation** (lấy đúng nhưng trả lời sai).\n"
        )
    else:
        eval_scope = "sample hiện tại" if EVAL_LIMIT else "full dataset hiện tại"
        lines.append(
            f"\n**Phân tích:** không có case dưới threshold trong {eval_scope}. "
            "Các case bottom vẫn nên được audit thủ công vì một metric riêng lẻ có thể thấp "
            "dù điểm trung bình còn trên ngưỡng.\n"
        )

    # Recommendations
    lines.append("\n---\n\n## Recommendations\n")
    recs = [
        ("Chốt ground truth cho các câu có `note`",
         "Đối chiếu khối lượng/khung hình phạt với văn bản gốc trong corpus, bỏ ghi chú. "
         "→ Context Recall & Faithfulness tăng vì judge so với đáp án chính xác."),
        ("Mở rộng Q&A cho mảng tin tức (news/)",
         "Dataset hiện đã có câu news, nhưng vẫn nên tăng độ phủ theo nhiều bài và nhiều dạng câu hỏi. "
         "→ Lộ rõ hơn điểm yếu retrieval trên văn bản phi cấu trúc."),
        ("Tăng top_k retrieval cho câu cross-reference",
         "Các câu tổng hợp nhiều điều luật cần nhiều evidence hơn (top_k 5 → 8). "
         "→ Context Recall tăng cho nhóm câu hard."),
    ]
    for i, (action, impact) in enumerate(recs, 1):
        lines.append(f"\n### Cải tiến {i}: {action}\n**Action:** {action}  \n**Expected impact:** {impact}  ")

    # Per-question appendix (config A)
    lines.append("\n\n---\n\n## Appendix — Per-question scores (Config A)\n")
    lines.append("| ID | Doc | Diff | Faith | Relev | Recall | Prec |")
    lines.append("|----|-----|------|-------|-------|--------|------|")
    for r in results[a]:
        lines.append(
            f"| {r['id']} | {r['doc']} | {r['difficulty']} | "
            f"{_fmt(r.get('faithfulness'))} | {_fmt(r.get('answer_relevancy'))} | "
            f"{_fmt(r.get('contextual_recall'))} | {_fmt(r.get('contextual_precision'))} |"
        )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ Báo cáo đã ghi: {RESULTS_PATH}")


def _gen_model() -> str:
    try:
        from src.config import LLM_MODEL

        return LLM_MODEL
    except Exception:
        return "?"


def _deepeval_version() -> str:
    try:
        import deepeval

        return deepeval.__version__
    except Exception:
        return "?"


# =============================================================================
# Main
# =============================================================================

def main():
    golden = load_golden_dataset()
    print(f"Loaded {len(golden)} test cases. Judge = {JUDGE_MODEL}. Workers = {EVAL_WORKERS}\n")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ Chưa có OPENAI_API_KEY trong môi trường — judge sẽ lỗi.", flush=True)

    judge = get_judge()
    t0 = time.time()
    results = {}
    for cfg, params in CONFIGS.items():
        print(f"\n=== Config {cfg} ({params['label']}) ===")
        results[cfg] = score_config(
            golden,
            params["use_reranking"],
            params["score_threshold"],
            judge,
        )

    export_results(results, time.time() - t0)

    def generate_with_citation(self, question: str) -> dict:
        item = self._by_q.get(question, {})
        expected = item.get("expected_answer", "")
        ctx = item.get("expected_context", "")
        return {"answer": expected, "sources": [{"content": f"{ctx}. {expected}"}]}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    main()
    import sys

    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    assert len(golden_dataset) >= 15, "Golden dataset phải có >= 15 Q&A pairs"

    if "--demo" in sys.argv:
        # Smoke-test harness bằng pipeline giả lập (cần OPENAI_API_KEY hoặc judge model).
        print("\n[DEMO] Chạy với _EchoPipeline — chỉ để test plumbing, không đo thật.\n")
        comparison = compare_configs(
            rag_pipeline=None,
            golden_dataset=golden_dataset,
            pipeline_factory=lambda params: _EchoPipeline(golden_dataset, **params),
        )
        export_results(comparison)
    else:
        # ---- Cắm RAG pipeline thật vào đây ----
        # from src.task10_generation import RAGPipeline
        # pipeline = RAGPipeline(...)
        #
        # # A/B: nếu pipeline dựng được theo config thì dùng factory cho sạch:
        # comparison = compare_configs(
        #     rag_pipeline=pipeline,
        #     golden_dataset=golden_dataset,
        #     configs=DEFAULT_CONFIGS,
        #     # pipeline_factory=lambda params: RAGPipeline(**params),
        # )
        # export_results(comparison)
        print(
            "⚠ Hãy import RAG pipeline thật ở khối __main__ rồi chạy lại "
            "(hoặc dùng `--demo` để smoke-test harness)."
        )
