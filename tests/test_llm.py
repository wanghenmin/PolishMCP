"""llm 与 config 纯逻辑测试（不联网）。"""

from __future__ import annotations

from aimcp.config import Settings
from aimcp.llm import is_retryable_error, remove_thinking_tags


def test_remove_thinking_tags_strips_block() -> None:
    text = "正文前<think>这里是想过程</think>正文后"
    assert remove_thinking_tags(text) == "正文前正文后"


def test_remove_thinking_tags_handles_thinking_variant() -> None:
    text = "<thinking>思考</thinking>结果"
    assert remove_thinking_tags(text) == "结果"


def test_remove_thinking_tags_passthrough_plain_text() -> None:
    assert remove_thinking_tags("普通文本") == "普通文本"


def test_is_retryable_error_generic_is_true() -> None:
    assert is_retryable_error(Exception("connection reset")) is True


def test_is_retryable_error_blocked_is_false() -> None:
    assert is_retryable_error(Exception("Your request was blocked by safety")) is False


def test_is_retryable_error_content_filter_is_false() -> None:
    assert is_retryable_error(Exception("Content filter triggered")) is False


def test_normalize_api_format_defaults_to_chat() -> None:
    assert Settings(LLM_API_KEY="x").normalize_api_format() == "chat"
    assert Settings(LLM_API_KEY="x", LLM_API_FORMAT="chat").normalize_api_format() == "chat"


def test_normalize_api_format_accepts_responses_aliases() -> None:
    for value in ("responses", "response", "responses_api", "Responses"):
        assert Settings(LLM_API_KEY="x", LLM_API_FORMAT=value).normalize_api_format() == "responses"


def test_reasoning_effort_respects_toggle() -> None:
    off = Settings(LLM_API_KEY="x", THINKING_MODE_ENABLED=False, THINKING_MODE_EFFORT="high")
    on = Settings(LLM_API_KEY="x", THINKING_MODE_ENABLED=True, THINKING_MODE_EFFORT="medium")
    assert off.reasoning_effort is None
    assert on.reasoning_effort == "medium"
