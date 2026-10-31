"""pytest 公共夹具：不依赖真实 API。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from course_ta.config import Settings


class FakeEmbeddings(Embeddings):
    """确定性的假向量，用于测试 Chroma 读写。"""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_api_key="test-llm-key",
        llm_base_url="https://example.invalid/v1",
        llm_model="test-chat",
        embedding_provider="fastembed",
        embedding_api_key="test-embedding-key",
        embedding_base_url="https://example.invalid/v1",
        embedding_model="test-embedding",
        fastembed_model="BAAI/bge-small-zh-v1.5",
        chroma_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        top_k=5,
        summary_threshold=20_000,
        enable_agent=True,
        disable_thinking=True,
        request_timeout=30,
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(
        responses=[
            "这是 stuff 摘要。",
            "这是 map 摘要一。",
            "这是 map 摘要二。",
            "这是最终 reduce 摘要。",
        ]
    )
