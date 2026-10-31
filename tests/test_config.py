from __future__ import annotations

from dataclasses import replace

import pytest

from course_ta.config import ConfigError, Settings, mask_key, validate_settings


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
