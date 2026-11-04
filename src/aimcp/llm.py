"""OpenAI 兼容大模型客户端封装。

支持两种接口格式：
- ``chat``：``/v1/chat/completions``（默认）
- ``responses``：``/v1/responses``（OpenAI Responses API）

支持流式与非流式：部分网关在非流式下不回填正文（例如 responses 的 ``output`` 为空数组），
改用流式即可正常拿到内容，通过 ``LLM_STREAM=true`` 开启。

两种格式都支持 ``reasoning_effort``（思考强度）；与 ``temperature`` 互斥，
调用失败且错误可重试时，会去掉推理参数、改用 ``temperature`` 再试一次。

当模型返回空内容时，抛出 ``EmptyResponseError`` 并附带原始响应，避免静默返回空串。
"""

from __future__ import annotations

import re
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from openai import (
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
)

# 不可重试：内容被阻止 / API Key 无效，降级重试也没用
NON_RETRYABLE_ERRORS = (PermissionDeniedError, AuthenticationError)

# 内容被阻止类关键词，命中则判定为不可重试
_BLOCKING_KEYWORDS = (
    "blocked",
    "content filter",
    "safety",
    "policy",
    "moderation",
    "harmful",
    "inappropriate",
)

# 原始响应 dump 的最大长度
_DUMP_LIMIT = 3000


class EmptyResponseError(RuntimeError):
    """模型返回空内容。单独成类，避免被当成可重试错误而重复计费。"""


def is_retryable_error(error: Exception) -> bool:
    """判断错误能否通过降级重试（去掉 reasoning 参数）解决。"""
    if isinstance(error, (EmptyResponseError, *NON_RETRYABLE_ERRORS)):
        return False
    message = str(error).lower()
    return not any(keyword in message for keyword in _BLOCKING_KEYWORDS)


def remove_thinking_tags(text: str) -> str:
    """移除模型输出中的思考标签（如 DeepSeek / o1 的 <think>…</think>）。"""
    if not text:
        return text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?thinking>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

    return text.strip()


def _dump(obj: Any) -> str:
    """把响应对象转成可打印的字符串（用于诊断）。"""
    try:
        data = obj.model_dump_json()
    except Exception:
        data = repr(obj)
    return data[:_DUMP_LIMIT]


def _extract_responses_text(response: Any) -> str:
    """从 Responses 响应中提取正文。

    优先用 SDK 的 ``output_text``；若为空，则兜底遍历 ``output`` 手动拼接，
    以兼容部分网关把 content 类型写成 ``text`` 而非 ``output_text`` 的情况。
    """
    text = getattr(response, "output_text", "") or ""
    if text:
        return text

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            piece = getattr(part, "text", None)
            if isinstance(piece, str) and piece:
                parts.append(piece)
    return "".join(parts)


class LLMClient:
    """封装单次对话补全调用，支持 chat / responses 两种格式与流式/非流式。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        api_format: str = "chat",
        temperature: float = 0.7,
        timeout_s: float = 60,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> None:
        self.model = model
        self.api_format = api_format
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            max_retries=2,
        )

    @property
    def use_reasoning(self) -> bool:
        """是否启用推理参数（reasoning_effort 为空或 none 时不启用）。"""
        return bool(self.reasoning_effort) and self.reasoning_effort != "none"

    async def complete(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """发起一次补全，返回去除思考标签后的正文。"""
        if self.api_format == "responses":
            runner = self._call_responses_stream if self.stream else self._call_responses
        else:
            runner = self._call_chat_stream if self.stream else self._call_chat

        raw, dump = await runner(system, user, history)

        text = remove_thinking_tags(raw)
        if not text.strip():
            self._report_empty(raw, dump)
            raise EmptyResponseError(
                f"模型返回空内容（格式 {self.api_format}，模型 {self.model}）。"
                f"若为非流式且网关未回填正文，可尝试设置 LLM_STREAM=true 走流式。"
                f"原始响应: {dump}"
            )
        return text

    def _chat_params(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if self.use_reasoning:
            # 用 extra_body 透传，兼容旧版 SDK 与第三方网关
            params["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        else:
            params["temperature"] = self.temperature
        return params

    def _responses_params(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = list(history or [])
        items.append({"role": "user", "content": user})

        params: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": items,
            "stream": stream,
        }
        if self.use_reasoning:
            params["reasoning"] = {"effort": self.reasoning_effort}
        else:
            params["temperature"] = self.temperature
        return params

    async def _call_chat(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> tuple[str, str]:
        params = self._chat_params(system, user, history, stream=False)

        async def call(p: dict[str, Any]) -> tuple[str, str]:
            response = await self.client.chat.completions.create(**p)
            content = response.choices[0].message.content or ""
            return content, _dump(response)

        return await self._with_fallback(call, params)

    async def _call_chat_stream(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> tuple[str, str]:
        params = self._chat_params(system, user, history, stream=True)

        async def call(p: dict[str, Any]) -> tuple[str, str]:
            stream = await self.client.chat.completions.create(**p)
            parts: list[str] = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
            return "".join(parts), f"<streamed {len(parts)} chunks>"

        return await self._with_fallback(call, params)

    async def _call_responses(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> tuple[str, str]:
        params = self._responses_params(system, user, history, stream=False)

        async def call(p: dict[str, Any]) -> tuple[str, str]:
            response = await self.client.responses.create(**p)
            return _extract_responses_text(response), _dump(response)

        return await self._with_fallback(call, params)

    async def _call_responses_stream(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> tuple[str, str]:
        params = self._responses_params(system, user, history, stream=True)

        async def call(p: dict[str, Any]) -> tuple[str, str]:
            stream = await self.client.responses.create(**p)
            parts: list[str] = []
            dump = "<streaming>"
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "response.output_text.delta":
                    parts.append(getattr(event, "delta", "") or "")
                elif event_type == "response.completed":
                    final = getattr(event, "response", None)
                    if final is not None:
                        dump = _dump(final)
            return "".join(parts), dump

        return await self._with_fallback(call, params)

    async def _with_fallback(
        self,
        call: Callable[[dict[str, Any]], Awaitable[tuple[str, str]]],
        params: dict[str, Any],
    ) -> tuple[str, str]:
        """执行调用；推理参数导致失败时降级重试一次。"""
        try:
            return await call(params)
        except Exception as error:
            if self.use_reasoning and is_retryable_error(error):
                retry_params = dict(params)
                retry_params.pop("extra_body", None)
                retry_params.pop("reasoning", None)
                retry_params["temperature"] = self.temperature
                return await call(retry_params)
            if isinstance(error, PermissionDeniedError):
                raise RuntimeError(
                    f"AI 请求被拒绝: {error}。"
                    "可能原因：内容触发安全过滤 / API Key 权限不足 / 代理服务配置问题。"
                ) from error
            raise

    def _report_empty(self, raw: str, dump: str) -> None:
        """把空响应详情打到 stderr（服务端日志可见）。"""
        lines = [
            "",
            "=" * 80,
            "[EMPTY RESPONSE] 模型未返回可用正文",
            f"[EMPTY RESPONSE] format={self.api_format} model={self.model}",
            f"[EMPTY RESPONSE] stream={self.stream} use_reasoning={self.use_reasoning}",
            f"[EMPTY RESPONSE] raw_text={raw!r}",
            f"[EMPTY RESPONSE] response_dump={dump}",
            "=" * 80,
            "",
        ]
        print("\n".join(lines), file=sys.stderr, flush=True)
