#!/usr/bin/env python3
"""Quick test CLI for the Day 8 RAG pipeline lab.

Default mode is intentionally light: it validates the repo artifacts and core
pure-Python paths without loading large embedding/reranking models or calling
external APIs. Use --mode pytest/full when you want the official README tests.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SRC_DIR = PROJECT_DIR / "src"
TEST_FILE = PROJECT_DIR / "tests" / "test_individual.py"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _pass(name: str, detail: str = "") -> Check:
    return Check(name, True, detail)


def _fail(name: str, detail: str) -> Check:
    return Check(name, False, detail)


def _import(module: str):
    sys.path.insert(0, str(PROJECT_DIR))
    return importlib.import_module(module)


def _count_files(path: Path, suffixes: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def check_task_1() -> list[Check]:
    legal = DATA_DIR / "landing" / "legal"
    count = _count_files(legal, {".pdf", ".doc", ".docx"})
    return [
        _pass("Task 1 directory", str(legal)) if legal.exists()
        else _fail("Task 1 directory", "missing data/landing/legal"),
        _pass("Task 1 legal files", f"{count} files") if count >= 3
        else _fail("Task 1 legal files", f"need >=3, got {count}"),
    ]


def check_task_2() -> list[Check]:
    news = DATA_DIR / "landing" / "news"
    count = _count_files(news, {".json", ".html", ".md", ".txt"})
    metadata_ok = True
    detail = "json metadata ok"
    for path in sorted(news.glob("*.json"))[:3]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            metadata_ok = False
            detail = f"{path.name}: {exc}"
            break
        if "url" not in payload:
            metadata_ok = False
            detail = f"{path.name}: missing url"
            break
    return [
        _pass("Task 2 directory", str(news)) if news.exists()
        else _fail("Task 2 directory", "missing data/landing/news"),
        _pass("Task 2 news files", f"{count} files") if count >= 5
        else _fail("Task 2 news files", f"need >=5, got {count}"),
        _pass("Task 2 metadata", detail) if metadata_ok
        else _fail("Task 2 metadata", detail),
    ]


def check_task_3() -> list[Check]:
    standardized = DATA_DIR / "standardized"
    md_files = sorted(standardized.rglob("*.md")) if standardized.exists() else []
    legal = list((standardized / "legal").glob("*.md")) if (standardized / "legal").exists() else []
    news = list((standardized / "news").glob("*.md")) if (standardized / "news").exists() else []
    return [
        _pass("Task 3 markdown files", f"{len(md_files)} files") if md_files
        else _fail("Task 3 markdown files", "no markdown files in data/standardized"),
        _pass("Task 3 legal/news split", f"legal={len(legal)}, news={len(news)}")
        if legal and news else _fail("Task 3 legal/news split", f"legal={len(legal)}, news={len(news)}"),
    ]


def check_task_4() -> list[Check]:
    checks: list[Check] = []
    try:
        mod = _import("src.task4_chunking_indexing")
        docs = mod.load_documents()
        chunks = mod.chunk_documents(docs[:1]) if docs else []
        checks.append(_pass("Task 4 config", f"size={mod.CHUNK_SIZE}, overlap={mod.CHUNK_OVERLAP}")
                      if 0 < mod.CHUNK_OVERLAP < mod.CHUNK_SIZE
                      else _fail("Task 4 config", "invalid chunk config"))
        checks.append(_pass("Task 4 load documents", f"{len(docs)} docs") if docs
                      else _fail("Task 4 load documents", "no docs loaded"))
        checks.append(_pass("Task 4 chunk sample", f"{len(chunks)} chunks") if chunks
                      else _fail("Task 4 chunk sample", "no chunks produced"))
    except Exception as exc:
        checks.append(_fail("Task 4 import/run", str(exc)))
    return checks


def check_task_5() -> list[Check]:
    try:
        mod = _import("src.task5_semantic_search")
        return [_pass("Task 5 semantic_search", "callable") if callable(mod.semantic_search)
                else _fail("Task 5 semantic_search", "not callable")]
    except Exception as exc:
        return [_fail("Task 5 import", str(exc))]


def check_task_6() -> list[Check]:
    try:
        mod = _import("src.task6_lexical_search")
        results = mod.lexical_search("ma tuý", top_k=3)
        shape_ok = isinstance(results, list) and (not results or "content" in results[0])
        return [_pass("Task 6 BM25 search", f"{len(results)} results") if shape_ok
                else _fail("Task 6 BM25 search", "bad result shape")]
    except Exception as exc:
        return [_fail("Task 6 BM25 search", str(exc))]


def check_task_7() -> list[Check]:
    try:
        mod = _import("src.task7_reranking")
        sample = [[{"content": "ma tuý", "score": 1.0, "metadata": {"chunk_id": "a"}}]]
        results = mod.rerank_rrf(sample, top_k=1)
        return [_pass("Task 7 RRF rerank", f"{len(results)} result") if results
                else _fail("Task 7 RRF rerank", "no result")]
    except Exception as exc:
        return [_fail("Task 7 import/run", str(exc))]


def check_task_8() -> list[Check]:
    try:
        mod = _import("src.task8_pageindex_vectorless")
        return [_pass("Task 8 pageindex_search", "callable") if callable(mod.pageindex_search)
                else _fail("Task 8 pageindex_search", "not callable")]
    except Exception as exc:
        return [_fail("Task 8 import", str(exc))]


def check_task_9() -> list[Check]:
    try:
        mod = _import("src.task9_retrieval_pipeline")
        return [_pass("Task 9 retrieve", "callable") if callable(mod.retrieve)
                else _fail("Task 9 retrieve", "not callable")]
    except Exception as exc:
        return [_fail("Task 9 import", str(exc))]


def check_task_10() -> list[Check]:
    try:
        mod = _import("src.task10_generation")
        ctx = mod.format_context([
            {
                "content": "Nội dung mẫu",
                "metadata": {"source": "sample.md", "doc_title": "Sample", "type": "legal", "dieu": 1},
            }
        ])
        ok = callable(mod.generate_with_citation) and callable(mod.reorder_for_llm) and "sample.md" in ctx
        return [_pass("Task 10 generation helpers", "callable + context source") if ok
                else _fail("Task 10 generation helpers", "bad helper shape")]
    except Exception as exc:
        return [_fail("Task 10 import/run", str(exc))]


SMOKE_CHECKS = {
    1: check_task_1,
    2: check_task_2,
    3: check_task_3,
    4: check_task_4,
    5: check_task_5,
    6: check_task_6,
    7: check_task_7,
    8: check_task_8,
    9: check_task_9,
    10: check_task_10,
}


def run_smoke(tasks: list[int]) -> int:
    all_checks: list[Check] = []
    for task in tasks:
        all_checks.extend(SMOKE_CHECKS[task]())

    width = max(len(c.name) for c in all_checks) if all_checks else 10
    for check in all_checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status:4}  {check.name:<{width}}  {check.detail}")

    failed = [c for c in all_checks if not c.ok]
    print(f"\nSummary: {len(all_checks) - len(failed)}/{len(all_checks)} checks passed")
    return 1 if failed else 0


def run_pytest(mode: str, task: int | None) -> int:
    if task is not None:
        target = f"{TEST_FILE}::TestTask{task}"
    elif mode == "full":
        target = str(PROJECT_DIR / "tests")
    else:
        target = str(TEST_FILE)

    cmd = [sys.executable, "-m", "pytest", target, "-q", "--tb=short"]
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=PROJECT_DIR, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick test CLI for the current RAG project.")
    parser.add_argument(
        "--mode",
        choices=("smoke", "pytest", "full"),
        default="smoke",
        help="smoke: fast local checks; pytest: README individual tests; full: all tests/",
    )
    parser.add_argument("--task", type=int, choices=range(1, 11), help="Run/check only one README task.")
    return parser.parse_args()


def main() -> int:
    venv_python = PROJECT_DIR / ".venv" / "bin" / "python"
    if (
        venv_python.exists()
        and Path(sys.executable).resolve() != venv_python.resolve()
        and os.getenv("QUICK_TEST_NO_VENV") != "1"
    ):
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

    args = parse_args()
    if args.mode == "smoke":
        tasks = [args.task] if args.task else list(range(1, 11))
        return run_smoke(tasks)
    return run_pytest(args.mode, args.task)


if __name__ == "__main__":
    raise SystemExit(main())
