"""Runtime helpers for local embedding/reranking models.

The lab machines may expose CUDA even when there is not enough free VRAM for
`bge-m3` or the reranker. These helpers make device selection explicit and
fall back to CPU on CUDA OOM so retrieval stays demo-safe.
"""

from __future__ import annotations

import os
from typing import Callable, TypeVar

T = TypeVar("T")


def resolve_device(env_name: str, default: str = "auto") -> str:
    """Resolve `auto|cpu|cuda` from env to a concrete torch device string."""
    choice = os.getenv(env_name, os.getenv("RAG_DEVICE", default)).strip().lower()
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_with_cpu_fallback(factory: Callable[[str], T], env_name: str) -> T:
    """Load a model on the selected device, retrying on CPU if CUDA OOMs."""
    device = resolve_device(env_name)
    try:
        return factory(device)
    except Exception as exc:
        if device != "cuda" or "out of memory" not in str(exc).lower():
            raise
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return factory("cpu")
