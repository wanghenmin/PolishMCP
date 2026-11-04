"""诊断脚本：直接打印后端模型对同一段文本的原始响应。

用于排查「工具返回空字符串」这类问题。在服务器项目目录执行：

    uv run python scripts/diagnose_llm.py

依次探测：
1. chat（/v1/chat/completions）
2. responses 非流式（SDK 解析结果）
3. responses 非流式（原始 HTTP 报文，判断网关到底发了什么）
4. responses 流式（累积 delta，判断流式路径是否能拿到正文）
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from aimcp.config import load_settings

TEXT = (
    "该方案体现了较强的可扩展性，能够有效支撑大规模并发场景下的稳定运行，"
    "为后续业务扩展奠定了坚实基础。"
)
INSTRUCTIONS = "只输出改写后的文本，不要解释。"


def _client(settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL.rstrip("/"),
        timeout=settings.LLM_TIMEOUT_S,
        max_retries=0,
    )


def _base_params(settings, stream: bool) -> dict:
    params: dict = {
        "model": settings.LLM_MODEL,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": TEXT}],
        "stream": stream,
    }
    if settings.reasoning_effort:
        params["reasoning"] = {"effort": settings.reasoning_effort}
    else:
        params["temperature"] = settings.LLM_TEMPERATURE
    return params


async def probe_chat(settings) -> None:
    print("\n=== 1. chat: /v1/chat/completions ===", flush=True)
    try:
        resp = await _client(settings).chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": TEXT}],
            stream=False,
        )
        message = resp.choices[0].message
        print("finish_reason:", resp.choices[0].finish_reason, flush=True)
        print("content:", repr(message.content), flush=True)
    except Exception as error:  # noqa: BLE001 - 诊断脚本需要打印所有异常
        print("ERROR:", type(error).__name__, error, flush=True)


async def probe_responses_nonstream(settings) -> None:
    print("\n=== 2. responses 非流式（SDK 解析） ===", flush=True)
    try:
        resp = await _client(settings).responses.create(**_base_params(settings, False))
        print("status:", getattr(resp, "status", None), flush=True)
        print("output_text:", repr(resp.output_text), flush=True)
        print("output_types:", [getattr(i, "type", None) for i in (resp.output or [])], flush=True)
        print("usage:", getattr(resp, "usage", None), flush=True)
    except Exception as error:  # noqa: BLE001
        print("ERROR:", type(error).__name__, error, flush=True)


async def probe_responses_raw(settings) -> None:
    print("\n=== 3. responses 非流式（原始 HTTP 报文） ===", flush=True)
    try:
        raw = await _client(settings).responses.with_raw_response.create(
            **_base_params(settings, False)
        )
        print("http status:", raw.status_code, flush=True)
        print("raw body:", raw.text[:3000], flush=True)
    except Exception as error:  # noqa: BLE001
        print("ERROR:", type(error).__name__, error, flush=True)


async def probe_responses_stream(settings) -> None:
    print("\n=== 4. responses 流式（累积 delta） ===", flush=True)
    try:
        stream = await _client(settings).responses.create(**_base_params(settings, True))
        collected: list[str] = []
        types: list[str] = []
        async for event in stream:
            event_type = getattr(event, "type", None)
            types.append(event_type)
            if event_type == "response.output_text.delta":
                collected.append(getattr(event, "delta", "") or "")
        print("event types:", types, flush=True)
        print("collected text:", repr("".join(collected)), flush=True)
    except Exception as error:  # noqa: BLE001
        print("ERROR:", type(error).__name__, error, flush=True)


async def main() -> None:
    settings = load_settings()
    print("api_format:", settings.normalize_api_format(), flush=True)
    print("base_url:", settings.LLM_BASE_URL, flush=True)
    print("models -> polish:", settings.resolve_model("polish"), flush=True)
    print("reasoning_effort:", settings.reasoning_effort, flush=True)

    await probe_chat(settings)
    await probe_responses_nonstream(settings)
    await probe_responses_raw(settings)
    await probe_responses_stream(settings)


if __name__ == "__main__":
    asyncio.run(main())
