"""诊断脚本：直接打印后端模型对同一段文本的原始响应。

用于排查「工具返回空字符串」这类问题。在服务器项目目录执行：

    uv run python scripts/diagnose_llm.py

会分别用 chat（/v1/chat/completions）和 responses（/v1/responses）两种格式
各调用一次，打印状态码、正文、finish_reason/status 以及原始响应 JSON。
"""

from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI

from aimcp.config import load_settings

TEXT = (
    "该方案体现了较强的可扩展性，能够有效支撑大规模并发场景下的稳定运行，"
    "为后续业务扩展奠定了坚实基础。"
)


def _client(settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL.rstrip("/"),
        timeout=settings.LLM_TIMEOUT_S,
        max_retries=0,
    )


async def probe_chat(settings) -> None:
    print("\n=== chat: /v1/chat/completions ===", flush=True)
    params = {
        "model": settings.LLM_MODEL,
        "messages": [{"role": "user", "content": TEXT}],
        "stream": False,
    }
    if settings.reasoning_effort:
        params["extra_body"] = {"reasoning_effort": settings.reasoning_effort}
    else:
        params["temperature"] = settings.LLM_TEMPERATURE

    print("params:", json.dumps(params, ensure_ascii=False)[:400], flush=True)
    try:
        resp = await _client(settings).chat.completions.create(**params)
        choice = resp.choices[0]
        message = choice.message
        print("finish_reason:", choice.finish_reason, flush=True)
        print("content:", repr(message.content), flush=True)
        print("reasoning_content:", repr(getattr(message, "reasoning_content", None)), flush=True)
        print("usage:", resp.usage, flush=True)
        print("raw:", resp.model_dump_json()[:2000], flush=True)
    except Exception as error:  # noqa: BLE001 - 诊断脚本需要打印所有异常
        print("ERROR:", type(error).__name__, error, flush=True)


async def probe_responses(settings) -> None:
    print("\n=== responses: /v1/responses ===", flush=True)
    params = {
        "model": settings.LLM_MODEL,
        "instructions": "只输出改写后的文本，不要解释。",
        "input": [{"role": "user", "content": TEXT}],
        "stream": False,
    }
    if settings.reasoning_effort:
        params["reasoning"] = {"effort": settings.reasoning_effort}
    else:
        params["temperature"] = settings.LLM_TEMPERATURE

    print("params:", json.dumps(params, ensure_ascii=False)[:400], flush=True)
    try:
        resp = await _client(settings).responses.create(**params)
        print("status:", getattr(resp, "status", None), flush=True)
        print("incomplete_details:", getattr(resp, "incomplete_details", None), flush=True)
        output = getattr(resp, "output", None) or []
        print("output_text:", repr(resp.output_text), flush=True)
        print("output_types:", [getattr(i, "type", None) for i in output], flush=True)
        print("usage:", getattr(resp, "usage", None), flush=True)
        print("raw:", resp.model_dump_json()[:2000], flush=True)
    except Exception as error:  # noqa: BLE001 - 诊断脚本需要打印所有异常
        print("ERROR:", type(error).__name__, error, flush=True)


async def main() -> None:
    settings = load_settings()
    print("api_format:", settings.normalize_api_format(), flush=True)
    print("base_url:", settings.LLM_BASE_URL, flush=True)
    print("models -> polish:", settings.resolve_model("polish"), flush=True)
    print("reasoning_effort:", settings.reasoning_effort, flush=True)

    await probe_chat(settings)
    await probe_responses(settings)


if __name__ == "__main__":
    asyncio.run(main())
