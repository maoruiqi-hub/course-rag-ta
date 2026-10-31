r"""真实 API 冒烟测试：确认对话与 embedding 都能连通。

用法：
    .\.venv\Scripts\python.exe scripts\check_api.py

该脚本会读取 .env，不会打印完整密钥，也不会写入任何文件。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from course_ta.config import build_embeddings, build_llm, validate_settings


def main() -> int:
    try:
        settings = validate_settings()
    except Exception as exc:
        print(f"[配置失败] {exc}")
        return 1

    print(f"LLM: {settings.llm_model} @ {settings.llm_base_url}")
    print(f"LLM key: {settings.masked_llm_key()}")
    print(
        f"Embedding: {settings.fastembed_model if settings.embedding_provider == 'fastembed' else settings.embedding_model}"
        f" ({settings.embedding_provider})"
    )

    try:
        started = time.perf_counter()
        response = build_llm(settings).invoke("只回复：ok")
        print(f"[对话成功] {response.content!r} ({time.perf_counter() - started:.1f}s)")
    except Exception as exc:
        print(f"[对话失败] {type(exc).__name__}: {exc}")
        return 1

    try:
        started = time.perf_counter()
        vector = build_embeddings(settings).embed_query("RAG 的两个核心组件是什么？")
        print(f"[Embedding 成功] 维度 {len(vector)} ({time.perf_counter() - started:.1f}s)")
    except Exception as exc:
        print(f"[Embedding 失败] {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
