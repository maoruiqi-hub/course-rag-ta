from __future__ import annotations

from dataclasses import replace

import pytest

from course_ta.config import ConfigError, Settings, friendly_error, mask_key, validate_settings


def test_mask_key() -> None:
    assert mask_key("") == "未配置"
    assert mask_key("short") == "***"
    assert mask_key("sk-1234567890") == "sk-1...7890"


def test_validate_settings_fastembed_needs_no_embedding_key(settings: Settings) -> None:
    validated = validate_settings(replace(settings, embedding_api_key=""))
    assert validated.chroma_dir.exists()
    assert validated.upload_dir.exists()


def test_validate_settings_siliconflow_requires_key(settings: Settings) -> None:
    with pytest.raises(ConfigError):
        validate_settings(replace(settings, embedding_provider="siliconflow", embedding_api_key=""))


def test_validate_settings_rejects_unknown_provider(settings: Settings) -> None:
    with pytest.raises(ConfigError):
        validate_settings(replace(settings, embedding_provider="unknown"))


class _FakeAPIError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def test_friendly_error_translates_common_api_failures() -> None:
    assert "余额不足" in friendly_error(_FakeAPIError("Insufficient Balance", 402))
    assert "API key 无效" in friendly_error(_FakeAPIError("Authentication Fails", 401))
    assert "限流" in friendly_error(_FakeAPIError("Rate limit reached", 429))
    assert friendly_error(ValueError("其他错误")) == "其他错误"
