"""应用配置：环境变量、模型客户端与路径。

所有密钥只从环境变量 / .env 读取，绝不写入源码或日志。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """配置缺失或非法时抛出，界面层会转换成中文提示。"""


def _env(*names: str, default: str = "") -> str:
    """按顺序读取第一个非空环境变量，兼容课程资料里的常见变量名。"""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def _path(name: str, default: str) -> Path:
    raw = _env(name, default=default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """运行期配置快照。"""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    embedding_provider: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    fastembed_model: str
    chroma_dir: Path
    upload_dir: Path
    top_k: int = 5
    summary_threshold: int = 20_000
    enable_agent: bool = True
    disable_thinking: bool = True
    request_timeout: int = 120

    def masked_llm_key(self) -> str:
        return mask_key(self.llm_api_key)

    def masked_embedding_key(self) -> str:
        return mask_key(self.embedding_api_key)


def mask_key(key: str) -> str:
    """只暴露首尾各 4 位，避免把密钥写进界面或截图。"""
    if not key:
        return "未配置"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取配置。调用方应先执行 ``validate_settings``。"""
    return Settings(
        llm_api_key=_env("LLM_API_KEY", "DEEPSEEK_API_KEY"),
        llm_base_url=_env("LLM_BASE_URL", default="https://api.deepseek.com"),
        llm_model=_env("LLM_MODEL", default="deepseek-flash"),
        embedding_provider=_env("EMBEDDING_PROVIDER", default="siliconflow").lower(),
        embedding_api_key=_env("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"),
        embedding_base_url=_env("EMBEDDING_BASE_URL", default="https://api.siliconflow.com/v1"),
        embedding_model=_env("EMBEDDING_MODEL", default="Qwen/Qwen3-Embedding-0.6B"),
        fastembed_model=_env("FASTEMBED_MODEL", default="BAAI/bge-small-zh-v1.5"),
        chroma_dir=_path("CHROMA_DIR", "./chroma_db"),
        upload_dir=_path("UPLOAD_DIR", "./data/uploads"),
        top_k=int(_env("TOP_K", default="5")),
        summary_threshold=int(_env("SUMMARY_THRESHOLD", default="20000")),
        enable_agent=_env("ENABLE_AGENT", default="true").lower() in {"1", "true", "yes", "on"},
        disable_thinking=_env("LLM_DISABLE_THINKING", default="true").lower()
        in {"1", "true", "yes", "on"},
        request_timeout=int(_env("REQUEST_TIMEOUT", default="120")),
    )


def validate_settings(settings: Settings | None = None) -> Settings:
    """检查必需配置，并创建本地目录。"""
    settings = settings or get_settings()
    missing: list[str] = []
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY（或 DEEPSEEK_API_KEY）")
    if settings.embedding_provider == "siliconflow" and not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY（或 SILICONFLOW_API_KEY）")
    if settings.embedding_provider not in {"siliconflow", "fastembed"}:
        raise ConfigError(
            f"不支持的 EMBEDDING_PROVIDER：{settings.embedding_provider}，"
            "可选 siliconflow / fastembed。"
        )
    if missing:
        raise ConfigError(
            "缺少 API 配置：" + "、".join(missing) + "。请复制 .env.example 为 .env 并填写。"
        )
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    """构造 DeepSeek / OpenAI 兼容聊天模型。

    2026 年 DeepSeek 默认开启 thinking 模式；本应用需要稳定、低延迟的
    普通问答与工具调用，因此默认通过 extra_body 关闭 thinking。
    """
    settings = settings or validate_settings()
    kwargs: dict = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "temperature": 0.3,
        "max_tokens": 4096,
        "max_retries": 2,
        "timeout": settings.request_timeout,
        "streaming": True,
    }
    if settings.disable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)


def build_embeddings(settings: Settings | None = None):
    """构造 embedding 客户端。

    默认走 OpenAI 兼容的 SiliconFlow；如果账号没有余额，可把
    ``EMBEDDING_PROVIDER`` 设为 ``fastembed``，使用本地 ONNX 小模型
    ``BAAI/bge-small-zh-v1.5``（首次运行会下载约 100MB）。
    """
    settings = settings or validate_settings()
    if settings.embedding_provider == "fastembed":
        from course_ta.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=settings.fastembed_model,
            cache_dir=PROJECT_ROOT / ".cache" / "fastembed",
        )

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        check_embedding_ctx_length=False,
        chunk_size=32,
        max_retries=3,
        timeout=60,
    )
