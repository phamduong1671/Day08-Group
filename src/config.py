"""
Config & singletons dùng chung cho toàn pipeline (Task 4 → 10).

Mục đích: load model nặng (bge-m3 ~2GB, reranker) **một lần duy nhất** và chia sẻ
cho mọi task, thay vì mỗi module tự reload. Tất cả getter đều **lazy** — import
module này không kéo theo việc tải model hay mở kết nối Weaviate.

Triết lý degrade-gracefully: nếu Weaviate/Ollama/PageIndex chưa sẵn sàng hoặc data
còn trống, các getter raise/return rỗng để task gọi nó trả `[]` thay vì crash
(test suite sẽ skip, demo không vỡ).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"

# =============================================================================
# MODEL / STORE CONFIG  (nguồn chân lý duy nhất — Task 4..10 import từ đây)
# =============================================================================

# Embedding: bge-m3 multilingual, mạnh nhất cho tiếng Việt (legal). 1024-dim.
# Override bằng env nếu cần model nhẹ hơn khi demo trên CPU yếu.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Reranker cross-encoder cùng họ bge → đồng nhất không gian ngữ nghĩa tiếng Việt.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Weaviate collection (bring-your-own-vector: vectorizer = none).
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")

def _has_real_openai_key(api_key: str) -> bool:
    placeholders = {"", "ollama", "your-api-key", "your_api_key", "OPENAI_API_KEY", "api_key_cua_ban"}
    return api_key.strip() not in placeholders


# LLM sinh câu trả lời (Task 10). Nếu .env có OPENAI_API_KEY thật thì mặc định
# gọi OpenAI API; nếu không có key thì fallback về Ollama local.
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
_USE_OPENAI_API = _has_real_openai_key(LLM_API_KEY)
LLM_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1" if _USE_OPENAI_API else "http://localhost:11434/v1",
)
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini" if _USE_OPENAI_API else "qwen2.5:7b-instruct"),
)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


# =============================================================================
# SINGLETONS (lazy + cached)
# =============================================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """SentenceTransformer dùng chung (Task 4 index + Task 5 query — phải cùng model)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    """CrossEncoder reranker dùng chung (Task 7)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def embed_query(text: str) -> list[float]:
    """Embed 1 query (normalize để dùng cosine similarity nhất quán với index)."""
    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def get_weaviate_client():
    """
    Kết nối Weaviate local. KHÔNG cache bằng lru_cache (client có vòng đời/đóng kết nối).
    Raise nếu Weaviate chưa chạy — caller bắt và trả [] để demo không vỡ.
    """
    import weaviate

    return weaviate.connect_to_local()


@lru_cache(maxsize=1)
def get_llm_client():
    """OpenAI-compatible client (mặc định Ollama)."""
    from openai import OpenAI

    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
