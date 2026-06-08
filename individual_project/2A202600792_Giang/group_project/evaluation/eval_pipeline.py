"""
RAG Evaluation Pipeline.

Đánh giá chất lượng RAG pipeline bằng DeepEval.

Framework đã chọn: DeepEval. Lý do: 4 metric yêu cầu
(faithfulness, relevance, context_recall, context_precision) ánh xạ
1-1 sang các metric có sẵn của DeepEval, và evaluate() chạy được offline
trong script Python (không bắt buộc cloud).

Yêu cầu:
    1. Load golden_dataset.json (>=15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Cài đặt:
    pip install deepeval

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

import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import json

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Tên 4 metric (key nội bộ) + ngưỡng pass mặc định.
METRIC_THRESHOLD = 0.7

# Ánh xạ key -> tên hiển thị trong report.
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "relevance": "Answer Relevancy",
    "context_recall": "Contextual Recall",
    "context_precision": "Contextual Precision",
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Judge model (LLM dùng để chấm điểm các metric)
# =============================================================================

def _build_judge_model():
    """
    Trả về model để DeepEval dùng làm judge.

    - Nếu đặt DEEPEVAL_JUDGE_MODEL -> dùng LiteLLM (hỗ trợ Claude, Gemini, Ollama...).
    - Ngược lại trả về None để DeepEval dùng mặc định (OpenAI qua OPENAI_API_KEY).
    """
    model_name = os.environ.get("DEEPEVAL_JUDGE_MODEL")
    if not model_name:
        return None
    try:
        from deepeval.models import LiteLLMModel  # type: ignore
    except Exception:  # pragma: no cover - tùy phiên bản deepeval
        from deepeval.models.litellm_model import LiteLLMModel  # type: ignore
    return LiteLLMModel(model=model_name)


def _build_metrics(model=None) -> list:
    """Khởi tạo 4 metric của DeepEval. Trả về list theo đúng thứ tự METRIC_LABELS."""
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )

    kwargs: dict[str, Any] = {"threshold": METRIC_THRESHOLD}
    if model is not None:
        kwargs["model"] = model

    return [
        FaithfulnessMetric(**kwargs),
        AnswerRelevancyMetric(**kwargs),
        ContextualRecallMetric(**kwargs),
        ContextualPrecisionMetric(**kwargs),
    ]


# Ánh xạ tên metric (do DeepEval trả về) -> key nội bộ.
# DeepEval đặt tên hơi khác nhau giữa các phiên bản nên dùng so khớp lỏng.
def _metric_key_from_name(name: str) -> str | None:
    n = name.lower()
    if "faithful" in n:
        return "faithfulness"
    if "relevan" in n and "context" not in n:  # answer relevancy
        return "relevance"
    if "recall" in n:
        return "context_recall"
    if "precision" in n:
        return "context_precision"
    return None


# =============================================================================
# Chạy pipeline trên dataset -> tạo LLMTestCase
# =============================================================================

def _apply_config(rag_pipeline: Any, params: dict | None) -> None:
    """Áp dụng config cho pipeline: ưu tiên configure(), fallback setattr."""
    if not params:
        return
    configure = getattr(rag_pipeline, "configure", None)
    if callable(configure):
        configure(**params)
        return
    for key, value in params.items():
        setattr(rag_pipeline, key, value)


def build_test_cases(
    rag_pipeline: Any,
    golden_dataset: list[dict],
    config: dict | None = None,
) -> tuple[list, list[dict]]:
    """
    Chạy RAG pipeline trên từng câu hỏi và đóng gói thành LLMTestCase.

    Trả về (test_cases, meta) trong đó meta giữ id/question để ghép lại sau khi chấm.
    """
    from deepeval.test_case import LLMTestCase

    _apply_config(rag_pipeline, config)

    test_cases: list = []
    meta: list[dict] = []

    for item in golden_dataset:
        question = item["question"]
        result = rag_pipeline.generate_with_citation(question)

        answer = result.get("answer", "")
        sources = result.get("sources", []) or []
        retrieval_context = [
            (s.get("content", "") if isinstance(s, dict) else str(s)) for s in sources
        ]
        # ContextualRecall/Precision cần ít nhất 1 context; tránh list rỗng gây lỗi.
        if not retrieval_context:
            retrieval_context = [""]

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=item.get("expected_answer", ""),
                retrieval_context=retrieval_context,
            )
        )
        meta.append({"id": item.get("id", question[:40]), "question": question})

    return test_cases, meta


# =============================================================================
# Parse kết quả của DeepEval (an toàn giữa các phiên bản)
# =============================================================================

def _iter_test_results(eval_output: Any):
    """evaluate() có thể trả EvaluationResult(.test_results) hoặc list trực tiếp."""
    test_results = getattr(eval_output, "test_results", None)
    return test_results if test_results is not None else eval_output


def _iter_metrics_data(test_result: Any):
    """Lấy danh sách metric data của 1 test result (tên thuộc tính đổi theo version)."""
    for attr in ("metrics_data", "metrics_metadata"):
        data = getattr(test_result, attr, None)
        if data:
            return data
    return []


def _aggregate(eval_output: Any, meta: list[dict]) -> dict:
    """
    Gom kết quả thành cấu trúc:
    {
      "per_case": [{"id","question","metrics":{key:{"score","success","reason"}}}],
      "aggregate": {key: {"mean": float|None, "pass_rate": float|None, "n": int}},
    }
    """
    per_case: list[dict] = []
    scores: dict[str, list[float]] = {k: [] for k in METRIC_LABELS}
    passes: dict[str, list[bool]] = {k: [] for k in METRIC_LABELS}

    results = list(_iter_test_results(eval_output))
    for idx, tr in enumerate(results):
        case_metrics: dict[str, dict] = {}
        for md in _iter_metrics_data(tr):
            name = getattr(md, "name", "") or ""
            key = _metric_key_from_name(name)
            if key is None:
                continue
            score = getattr(md, "score", None)
            success = bool(getattr(md, "success", False))
            reason = getattr(md, "reason", "") or ""
            case_metrics[key] = {"score": score, "success": success, "reason": reason}
            if isinstance(score, (int, float)):
                scores[key].append(float(score))
            passes[key].append(success)

        m = meta[idx] if idx < len(meta) else {"id": f"case_{idx}", "question": ""}
        per_case.append(
            {"id": m["id"], "question": m["question"], "metrics": case_metrics}
        )

    aggregate: dict[str, dict] = {}
    for key in METRIC_LABELS:
        s = scores[key]
        p = passes[key]
        aggregate[key] = {
            "mean": (statistics.mean(s) if s else None),
            "pass_rate": (sum(p) / len(p) if p else None),
            "n": len(p),
        }

    return {"per_case": per_case, "aggregate": aggregate}


# =============================================================================
# DeepEval evaluation
# =============================================================================

def evaluate_with_deepeval(
    rag_pipeline: Any,
    golden_dataset: list[dict],
    config: dict | None = None,
) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    from deepeval import evaluate

    judge = _build_judge_model()
    metrics = _build_metrics(model=judge)
    test_cases, meta = build_test_cases(rag_pipeline, golden_dataset, config=config)

    # In ra giảm nhiễu; bỏ qua nếu phiên bản không hỗ trợ tham số.
    try:
        eval_output = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            print_results=False,
            show_indicator=False,
        )
    except TypeError:
        eval_output = evaluate(test_cases, metrics)

    return _aggregate(eval_output, meta)


# =============================================================================
# A/B Comparison
# =============================================================================

# Các config muốn so sánh. Điều chỉnh theo tham số thật của pipeline.
DEFAULT_CONFIGS: dict[str, dict] = {
    "hybrid_rerank": {"use_reranking": True, "alpha": 0.5},
    "dense_only": {"use_reranking": False, "alpha": 1.0},
}


def compare_configs(
    rag_pipeline: Any,
    golden_dataset: list[dict],
    configs: dict[str, dict] | None = None,
    pipeline_factory: Callable[[dict], Any] | None = None,
) -> dict:
    """
    So sánh A/B giữa ít nhất 2 configs.

    - Nếu truyền `pipeline_factory(params) -> pipeline`, mỗi config sẽ dựng pipeline mới
      (sạch sẽ, tránh lẫn state). Đây là cách khuyến nghị.
    - Nếu không, dùng `rag_pipeline` chung và áp config qua configure()/setattr trước mỗi run.

    Trả về: {config_name: <kết quả evaluate_with_deepeval>}
    """
    configs = configs or DEFAULT_CONFIGS
    results: dict[str, dict] = {}

    for config_name, params in configs.items():
        print(f"\n=== Đang chạy config: {config_name} ({params}) ===")
        if pipeline_factory is not None:
            pipeline = pipeline_factory(params)
            results[config_name] = evaluate_with_deepeval(pipeline, golden_dataset)
        else:
            results[config_name] = evaluate_with_deepeval(
                rag_pipeline, golden_dataset, config=params
            )

    return results


# =============================================================================
# Export Results
# =============================================================================

def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def _overall_mean(aggregate: dict) -> float | None:
    means = [v["mean"] for v in aggregate.values() if v["mean"] is not None]
    return statistics.mean(means) if means else None


def export_results(comparison: dict, configs: dict[str, dict] | None = None) -> str:
    """
    Export evaluation results to results.md.

    `comparison` là dict {config_name: kết quả evaluate_with_deepeval}.
    Trả về nội dung markdown đã ghi.
    """
    configs = configs or DEFAULT_CONFIGS
    lines: list[str] = []
    lines.append("# RAG Evaluation Results")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append(f"_Framework: DeepEval · Pass threshold: {METRIC_THRESHOLD}_")
    lines.append("")

    config_names = list(comparison.keys())

    # --- 1. Overall scores per config ---
    lines.append("## Overall Scores")
    lines.append("")
    header = "| Metric | " + " | ".join(config_names) + " |"
    sep = "|" + "---|" * (len(config_names) + 1)
    lines.append(header)
    lines.append(sep)
    for key, label in METRIC_LABELS.items():
        row = [label]
        for cfg in config_names:
            agg = comparison[cfg]["aggregate"][key]
            row.append(f"{_fmt(agg['mean'])} ({_fmt(agg['pass_rate'], pct=True)} pass)")
        lines.append("| " + " | ".join(row) + " |")
    # Hàng trung bình tổng.
    avg_row = ["**Mean (all metrics)**"]
    for cfg in config_names:
        avg_row.append(f"**{_fmt(_overall_mean(comparison[cfg]['aggregate']))}**")
    lines.append("| " + " | ".join(avg_row) + " |")
    lines.append("")

    # --- 2. A/B comparison verdict ---
    lines.append("## A/B Comparison")
    lines.append("")
    ranked = sorted(
        config_names,
        key=lambda c: (_overall_mean(comparison[c]["aggregate"]) or -1),
        reverse=True,
    )
    best = ranked[0]
    lines.append(
        f"**Config tốt nhất theo điểm trung bình: `{best}`** "
        f"({_fmt(_overall_mean(comparison[best]['aggregate']))})."
    )
    lines.append("")
    lines.append("Chênh lệch theo từng metric (so với config tốt nhất):")
    lines.append("")
    lines.append("| Metric | " + " | ".join(config_names) + " |")
    lines.append(sep)
    for key, label in METRIC_LABELS.items():
        base = comparison[best]["aggregate"][key]["mean"]
        row = [label]
        for cfg in config_names:
            m = comparison[cfg]["aggregate"][key]["mean"]
            if m is None or base is None:
                row.append("N/A")
            else:
                delta = m - base
                sign = "+" if delta >= 0 else ""
                row.append(f"{_fmt(m)} ({sign}{delta:.3f})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- 3. Worst performers (theo config tốt nhất) ---
    lines.append(f"## Worst Performers — `{best}`")
    lines.append("")
    best_cases = comparison[best]["per_case"]

    def case_min_score(case: dict) -> float:
        vals = [
            mv["score"]
            for mv in case["metrics"].values()
            if isinstance(mv.get("score"), (int, float))
        ]
        return min(vals) if vals else 1.0

    worst = sorted(best_cases, key=case_min_score)[:5]
    if worst:
        lines.append("| ID | Question | Lowest metric | Score | Reason |")
        lines.append("|---|---|---|---|---|")
        for case in worst:
            metrics = case["metrics"]
            if not metrics:
                continue
            low_key = min(
                metrics,
                key=lambda k: metrics[k]["score"]
                if isinstance(metrics[k].get("score"), (int, float))
                else 1.0,
            )
            low = metrics[low_key]
            q = case["question"].replace("|", "\\|")
            q = (q[:60] + "…") if len(q) > 60 else q
            reason = (low.get("reason") or "").replace("|", "\\|").replace("\n", " ")
            reason = (reason[:120] + "…") if len(reason) > 120 else reason
            lines.append(
                f"| {case['id']} | {q} | {METRIC_LABELS[low_key]} | "
                f"{_fmt(low['score'])} | {reason} |"
            )
    else:
        lines.append("_Không có dữ liệu per-case._")
    lines.append("")

    # --- 4. Recommendations (tự động) ---
    lines.append("## Recommendations")
    lines.append("")
    recs: list[str] = []
    recs.append(
        f"Triển khai config `{best}` cho production vì đạt điểm trung bình cao nhất."
    )
    best_agg = comparison[best]["aggregate"]
    weak = [
        METRIC_LABELS[k]
        for k, v in best_agg.items()
        if v["mean"] is not None and v["mean"] < METRIC_THRESHOLD
    ]
    if weak:
        recs.append(
            "Các metric dưới ngưỡng cần cải thiện: "
            + ", ".join(weak)
            + ". Gợi ý: nếu Contextual Recall/Precision thấp -> chỉnh retriever "
            "(top_k, reranking, chunking); nếu Faithfulness thấp -> siết prompt buộc "
            "trả lời bám context; nếu Answer Relevancy thấp -> tinh chỉnh prompt sinh câu trả lời."
        )
    else:
        recs.append("Tất cả metric đều vượt ngưỡng — pipeline đạt yêu cầu cơ bản.")
    for r in recs:
        lines.append(f"- {r}")
    lines.append("")

    content = "\n".join(lines)
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n Đã ghi kết quả vào {RESULTS_PATH}")
    return content


# =============================================================================
# (Tùy chọn) Pipeline giả lập để smoke-test bộ harness mà không cần pipeline thật.
# CHỈ để kiểm tra plumbing — KHÔNG phản ánh chất lượng RAG thật.
# Bật bằng: python evaluate_rag.py --demo
# =============================================================================

class _EchoPipeline:
    """Pipeline demo: trả expected_answer làm context + answer. KHÔNG dùng để đo thật."""

    def __init__(self, dataset: list[dict], use_reranking: bool = True, alpha: float = 0.5):
        self._by_q = {d["question"]: d for d in dataset}
        self.use_reranking = use_reranking
        self.alpha = alpha

    def configure(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def generate_with_citation(self, question: str) -> dict:
        item = self._by_q.get(question, {})
        expected = item.get("expected_answer", "")
        ctx = item.get("expected_context", "")
        return {"answer": expected, "sources": [{"content": f"{ctx}. {expected}"}]}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
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
        raise NotImplementedError(
            "Cắm RAG pipeline thật vào phần main để chạy đánh giá."
        )