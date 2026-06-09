"""Evaluation pipeline for the group RAG chatbot.

The module can be called from the HTML chatbot server or run directly:

    python -m group_project.evaluation.eval_pipeline

It uses `golden_dataset.json`, runs `group_project.rag_chatbot_backend.chat`,
scores four RAG metrics, writes `results.md`, and returns a JSON-friendly
summary. If `OPENAI_API_KEY` is present in `.env`, an OpenAI judge is used.
Without a key, it falls back to deterministic lexical overlap scoring so the UI
still works during demos.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
DEFAULT_LIMIT = int(os.getenv("EVAL_LIMIT", "5") or "5")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "contextual_recall": "Context Recall",
    "contextual_precision": "Context Precision",
}


def load_golden_dataset(limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if limit is None:
        limit = DEFAULT_LIMIT
    if limit and limit > 0:
        return data[:limit]
    return data


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\wÀ-ỹ]+", str(text), flags=re.UNICODE)
        if len(token) >= 2
    }


def _overlap(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _heuristic_scores(question: str, answer: str, expected_answer: str, contexts: list[str]) -> dict[str, float]:
    joined_context = "\n".join(contexts)
    return {
        "faithfulness": round(_overlap(answer, joined_context), 3),
        "answer_relevancy": round((_overlap(question, answer) + _overlap(expected_answer, answer)) / 2, 3),
        "contextual_recall": round(_overlap(expected_answer, joined_context), 3),
        "contextual_precision": round(max((_overlap(question, ctx) for ctx in contexts), default=0.0), 3),
    }


def _openai_judge_scores(question: str, answer: str, expected_answer: str, contexts: list[str]) -> dict[str, float] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.lower() in {"ollama", "your-api-key", "your_api_key"}:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        context_text = "\n\n---\n\n".join(contexts[:5])
        prompt = f"""
Evaluate this Vietnamese RAG answer. Return ONLY valid JSON with numeric scores from 0 to 1.

Metrics:
- faithfulness: answer is supported by retrieved context
- answer_relevancy: answer addresses the question
- contextual_recall: retrieved context contains evidence needed for expected answer
- contextual_precision: retrieved context is focused and relevant

Question: {question}
Expected answer: {expected_answer}
Actual answer: {answer}
Retrieved context:
{context_text}

JSON schema:
{{"faithfulness":0.0,"answer_relevancy":0.0,"contextual_recall":0.0,"contextual_precision":0.0}}
"""
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict RAG evaluator. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        scores = json.loads(match.group(0) if match else raw)
        return {
            key: round(max(0.0, min(1.0, float(scores.get(key, 0.0)))), 3)
            for key in METRIC_LABELS
        }
    except Exception as exc:
        print(f"OpenAI judge failed; using heuristic scoring ({type(exc).__name__})")
        return None


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0


def _row_average(row: dict[str, Any]) -> float:
    return _mean([float(row[key]) for key in METRIC_LABELS if row.get(key) is not None])


def evaluate_chatbot(limit: int | None = None, top_k: int = 5, exact_phrase: bool = False) -> dict[str, Any]:
    """Run chatbot evaluation and write results.md."""
    from group_project.rag_chatbot_backend import chat, reset_session

    dataset = load_golden_dataset(limit)
    rows: list[dict[str, Any]] = []
    started = time.time()
    judge_used = False

    for index, item in enumerate(dataset, 1):
        session_id = f"eval-{uuid.uuid4().hex[:10]}"
        reset_session(session_id)
        result = chat(
            item["question"],
            session_id=session_id,
            top_k=top_k,
            exact_phrase=exact_phrase,
        )
        answer = result.get("answer", "")
        sources = result.get("source_documents") or result.get("sources") or []
        contexts = [str(source.get("content") or source.get("preview") or "") for source in sources]
        if not contexts:
            contexts = ["(no retrieved context)"]

        scores = _openai_judge_scores(item["question"], answer, item.get("expected_answer", ""), contexts)
        if scores is None:
            scores = _heuristic_scores(item["question"], answer, item.get("expected_answer", ""), contexts)
        else:
            judge_used = True

        row = {
            "id": item.get("id", f"case-{index}"),
            "question": item["question"],
            "expected_answer": item.get("expected_answer", ""),
            "answer": answer,
            "doc": item.get("doc", ""),
            "difficulty": item.get("difficulty", ""),
            "sources_count": len(sources),
            "generation_backend": result.get("generation_backend", "unknown"),
            **scores,
        }
        row["average"] = _row_average(row)
        rows.append(row)

    averages = {
        key: _mean([float(row[key]) for row in rows if row.get(key) is not None])
        for key in METRIC_LABELS
    }
    averages["overall"] = _mean(list(averages.values()))

    summary = {
        "ok": True,
        "dataset_size": len(load_golden_dataset(limit=0)),
        "evaluated": len(rows),
        "judge": JUDGE_MODEL if judge_used else "heuristic_overlap",
        "top_k": top_k,
        "exact_phrase": exact_phrase,
        "averages": averages,
        "worst": sorted(rows, key=lambda row: row["average"])[:3],
        "results_path": str(RESULTS_PATH),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    export_results(summary, rows)
    return summary


def export_results(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# RAG Chatbot Evaluation Results",
        "",
        f"- Run time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Judge: `{summary['judge']}`",
        f"- Evaluated: {summary['evaluated']} / {summary['dataset_size']} cases",
        f"- Top-k: {summary['top_k']}",
        f"- Exact phrase mode: {summary['exact_phrase']}",
        f"- Elapsed: {summary['elapsed_seconds']}s",
        "",
        "## Average Scores",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for key, label in METRIC_LABELS.items():
        lines.append(f"| {label} | {summary['averages'][key]:.3f} |")
    lines.append(f"| **Overall** | **{summary['averages']['overall']:.3f}** |")

    lines.extend([
        "",
        "## Worst Cases",
        "",
        "| ID | Difficulty | Avg | Question |",
        "|---|---|---:|---|",
    ])
    for row in summary["worst"]:
        question = str(row["question"]).replace("|", "/")[:100]
        lines.append(f"| {row['id']} | {row['difficulty']} | {row['average']:.3f} | {question} |")

    lines.extend([
        "",
        "## Per-case Scores",
        "",
        "| ID | Doc | Faith | Relevancy | Recall | Precision | Avg | Sources |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['doc']} | {row['faithfulness']:.3f} | "
            f"{row['answer_relevancy']:.3f} | {row['contextual_recall']:.3f} | "
            f"{row['contextual_precision']:.3f} | {row['average']:.3f} | {row['sources_count']} |"
        )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    limit_raw = os.getenv("EVAL_LIMIT", str(DEFAULT_LIMIT))
    limit = int(limit_raw) if limit_raw.strip() else DEFAULT_LIMIT
    if limit == 0:
        limit = None
    summary = evaluate_chatbot(limit=limit, top_k=int(os.getenv("EVAL_TOP_K", "5")))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
