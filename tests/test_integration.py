from __future__ import annotations

import os

import pytest

from course_ta.config import build_embeddings, build_llm, get_settings


@pytest.mark.integration
def test_real_chat_and_embedding() -> None:
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("设置 RUN_INTEGRATION=1 才运行真实 API 测试")
    settings = get_settings()
    llm = build_llm(settings)
    response = llm.invoke("只回复：ok")
    assert "ok" in str(response.content).lower()

    embeddings = build_embeddings(settings)
    vector = embeddings.embed_query("RAG 是什么？")
    assert len(vector) >= 128
