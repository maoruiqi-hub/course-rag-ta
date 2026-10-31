"""本地 ONNX embedding 备选方案。

当 SiliconFlow 账号没有余额，或希望完全离线运行时，使用 FastEmbed 的
``BAAI/bge-small-zh-v1.5``（512 维，中文效果足够课程问答使用）。
首次运行会从 HuggingFace 镜像下载约 100MB 模型，之后离线可用。
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.embeddings import Embeddings

# 国内网络访问 huggingface.co 经常超时，默认走镜像；用户可自行覆盖。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class FastEmbedEmbeddings(Embeddings):
    """把 FastEmbed 包装成 LangChain Embeddings 接口。"""

    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [[float(value) for value in vector] for vector in model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        if hasattr(model, "query_embed"):
            vector = next(iter(model.query_embed([text])))
        else:  # 兼容旧版 FastEmbed
            vector = next(iter(model.embed([text])))
        return [float(value) for value in vector]
