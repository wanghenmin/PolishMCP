"""处理管线：分段 → 逐段改写（携带前文风格上下文）→ 拼接。"""

from __future__ import annotations

from .config import Settings
from .llm import LLMClient
from .prompts import (
    default_enhance_prompt,
    default_polish_prompt,
    emotion_polish_prompt,
    paragraph_task_suffix,
)
from .text import count_text_length, split_text_into_segments

POLISH_TASK = "学术润色"
ENHANCE_TASK = "去痕增强"
EMOTION_TASK = "感情文章润色"


class Processor:
    """按模式对文本执行润色 / 增强，长文本自动分段顺序处理。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._clients: dict[str, LLMClient] = {}

    def _client(self, stage: str) -> LLMClient:
        if stage not in self._clients:
            self._clients[stage] = LLMClient(
                model=self.settings.resolve_model(stage),
                api_key=self.settings.LLM_API_KEY,
                base_url=self.settings.LLM_BASE_URL,
                api_format=self.settings.normalize_api_format(),
                temperature=self.settings.LLM_TEMPERATURE,
                timeout_s=self.settings.LLM_TIMEOUT_S,
                reasoning_effort=self.settings.reasoning_effort,
            )
        return self._clients[stage]

    async def _run_stage(self, text: str, stage: str, prompt: str, task: str) -> str:
        segments = split_text_into_segments(text, self.settings.SEGMENT_MAX_CHARS)
        client = self._client(stage)
        system = prompt + paragraph_task_suffix(task)

        history: list[dict[str, str]] = []
        outputs: list[str] = []

        for segment in segments:
            # 标题或过短段落原样透传，不送模型
            if count_text_length(segment) < self.settings.SEGMENT_SKIP_THRESHOLD:
                outputs.append(segment)
                continue

            rewritten = await client.complete(system, segment, history)
            outputs.append(rewritten)
            history = (
                history + [{"role": "assistant", "content": rewritten}]
            )[-self.settings.HISTORY_MAX_SEGMENTS :]

        return "\n\n".join(outputs)

    async def polish(self, text: str) -> str:
        return await self._run_stage(text, "polish", default_polish_prompt(), POLISH_TASK)

    async def enhance(self, text: str) -> str:
        return await self._run_stage(text, "enhance", default_enhance_prompt(), ENHANCE_TASK)

    async def emotion_polish(self, text: str) -> str:
        return await self._run_stage(
            text, "emotion_polish", emotion_polish_prompt(), EMOTION_TASK
        )

    async def polish_and_enhance(self, text: str) -> str:
        """先润色再增强，两遍串联。"""
        polished = await self.polish(text)
        return await self.enhance(polished)
