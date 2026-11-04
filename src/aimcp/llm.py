"""OpenAI 兼容大模型客户端封装。

支持两种接口格式：
- ``chat``：``/v1/chat/completions``（默认）
- ``responses``：``/v1/responses``（OpenAI Responses API）

两种格式都支持 ``reasoning_effort``（思考强度）；与 ``temperature`` 互斥，
调用失败且错误可重试时，会去掉推理参数、改用 ``temperature`` 再试一次。
"""

from __future__ import annotations

import re
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


def is_retryable_error(error: Exception) -> bool:
    """判断错误能否通过降级重试（去掉 reasoning 参数）解决。"""
    if isinstance(error, NON_RETRYABLE_ERRORS):
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


class LLMClient:
    """封装单次对话补全调用，支持 chat / responses 两种格式。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        api_format: str = "chat",
        temperature: float = 0.7,
        timeout_s: float = 60,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.api_format = api_format
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
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
        """发起一次非流式补全，返回去除思考标签后的正文。"""
        if self.api_format == "responses":
            raw = await self._call_responses(system, user, history)
        else:
            raw = await self._call_chat(system, user, history)
        return remove_thinking_tags(raw)

    async def _call_chat(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.use_reasoning:
            # 用 extra_body 透传，兼容旧版 SDK 与第三方网关
            params["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        else:
            params["temperature"] = self.temperature

        async def call(p: dict[str, Any]) -> str:
            response = await self.client.chat.completions.create(**p)
            return response.choices[0].message.content or ""

        return await self._with_fallback(call, params)

    async def _call_responses(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> str:
        items: list[dict[str, Any]] = list(history or [])
        items.append({"role": "user", "content": user})

        params: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": items,
            "stream": False,
        }
        if self.use_reasoning:
            params["reasoning"] = {"effort": self.reasoning_effort}
        else:
            params["temperature"] = self.temperature

        async def call(p: dict[str, Any]) -> str:
            response = await self.client.responses.create(**p)
            return response.output_text or ""

        return await self._with_fallback(call, params)

    async def _with_fallback(
        self,
        call: Callable[[dict[str, Any]], Awaitable[str]],
        params: dict[str, Any],
    ) -> str:
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
