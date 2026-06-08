#!/usr/bin/env python3
"""Run one RAG query and print the result.

Examples:
    python tools/run_query.py "Hình phạt tàng trữ trái phép chất ma tuý?"
    python tools/run_query.py --mode retrieval "Điều 249 quy định gì?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _ensure_project_python() -> None:
    venv_python = PROJECT_DIR / ".venv" / "bin" / "python"
    if (
        venv_python.exists()
        and Path(sys.executable).resolve() != venv_python.resolve()
        and os.getenv("RUN_QUERY_NO_VENV") != "1"
    ):
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a retrieval or generation query.")
    parser.add_argument("query", help="Question/query to run through the RAG pipeline.")
    parser.add_argument(
        "--mode",
        choices=("generation", "retrieval"),
        default="generation",
        help="generation prints final cited answer; retrieval prints retrieved chunks as JSON.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks/results to use.")
    return parser.parse_args()


def run_retrieval(query: str, top_k: int) -> int:
    from src.task9_retrieval_pipeline import retrieve

    results = retrieve(query, top_k=top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def run_generation(query: str, top_k: int) -> int:
    from src.task10_generation import generate_with_citation

    result = generate_with_citation(query, top_k=top_k)
    print(result["answer"])
    if result.get("sources"):
        print(f"\nSources: {len(result['sources'])} chunks | via {result.get('retrieval_source')}")
    return 0


def main() -> int:
    _ensure_project_python()
    sys.path.insert(0, str(PROJECT_DIR))

    args = parse_args()
    if args.mode == "retrieval":
        return run_retrieval(args.query, args.top_k)
    return run_generation(args.query, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
