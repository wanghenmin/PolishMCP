"""llm 与 config 纯逻辑测试（不联网）。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from aimcp.config import Settings
from aimcp.llm import (
    EmptyResponseError,
    LLMClient,
    _extract_responses_text,
    is_retryable_error,
    remove_thinking_tags,
)


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


def test_is_retryable_error_empty_response_is_false() -> None:
    assert is_retryable_error(EmptyResponseError("空")) is False


def test_extract_responses_text_prefers_output_text() -> None:
    response = SimpleNamespace(output_text="标准正文", output=[])
    assert _extract_responses_text(response) == "标准正文"


def test_extract_responses_text_falls_back_to_nonstandard_content() -> None:
    text_part = SimpleNamespace(type="text", text="非标准正文")
    item = SimpleNamespace(type="message", content=[text_part])
    response = SimpleNamespace(output_text="", output=[item])
    assert _extract_responses_text(response) == "非标准正文"


def test_extract_responses_text_empty_when_nothing() -> None:
    response = SimpleNamespace(output_text="", output=[])
    assert _extract_responses_text(response) == ""


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


class _FakeResponsesStream:
    def __init__(self) -> None:
        self.params: dict = {}

    async def create(self, **params):
        self.params = params

        async def gen():
            yield SimpleNamespace(type="response.created")
            for char in "你好":
                yield SimpleNamespace(type="response.output_text.delta", delta=char)
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(model_dump_json=lambda: "{}"),
            )

        return gen()


class _FakeChatStream:
    def __init__(self) -> None:
        self.params: dict = {}

    async def create(self, **params):
        self.params = params

        async def gen():
            for char in "你好":
                delta = SimpleNamespace(content=char)
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        return gen()


def _client(api_format: str, resource) -> LLMClient:
    client = LLMClient(
        model="m", api_key="k", base_url="http://h/v1", api_format=api_format, stream=True
    )
    chat = SimpleNamespace(completions=resource if api_format == "chat" else None)
    responses = resource if api_format == "responses" else None
    client.client = SimpleNamespace(chat=chat, responses=responses)  # type: ignore[assignment]
    return client


def test_responses_stream_accumulates_deltas() -> None:
    fake = _FakeResponsesStream()
    client = _client("responses", fake)
    out = asyncio.run(client.complete("SYS", "USER"))
    assert out == "你好"
    assert fake.params["stream"] is True


def test_chat_stream_accumulates_chunks() -> None:
    fake = _FakeChatStream()
    client = _client("chat", fake)
    out = asyncio.run(client.complete("SYS", "USER"))
    assert out == "你好"
    assert fake.params["stream"] is True
