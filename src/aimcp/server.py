"""FastMCP 装配：4 个工具 + stdio / remote 双模式。"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .config import Settings
from .processor import Processor

INSTRUCTIONS = (
    "学术写作与去 AI 痕迹助手。对文本做润色、去痕增强或感情文润色，"
    "降低 AI 生成特征，同时保留原意、术语与段落结构。"
    "长文本会自动分段处理。"
)


def create_server(settings: Settings) -> FastMCP:
    """创建带 4 个工具的 FastMCP 实例。"""
    processor = Processor(settings)

    mcp = FastMCP(
        "aimcp",
        instructions=INSTRUCTIONS,
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
    )

    @mcp.tool(
        name="polish_text",
        description=(
            "对学术/技术文本做第一遍润色，去掉 AI 腔（拔高、套话、假分析、同义词轮换等），"
            "改为清楚、具体、克制的表达，保留原意、术语与段落结构。"
            "参数 text 为待润色的文本；返回润色后的文本。"
        ),
    )
    async def polish_text(text: str) -> str:
        return await processor.polish(text)

    @mcp.tool(
        name="enhance_text",
        description=(
            "第二遍去痕增强：在第一遍润色结果上进一步消除残留的 AI 特征"
            "（句长过于均匀、同义词轮换、空泛句尾、展望拔高）。"
            "参数 text 为待增强的文本；返回增强后的文本。"
        ),
    )
    async def enhance_text(text: str) -> str:
        return await processor.enhance(text)

    @mcp.tool(
        name="polish_and_enhance",
        description=(
            "完整两阶段处理：先润色(polish)再去痕增强(enhance)，适合希望一次到位的学术文本。"
            "参数 text 为待处理的文本；返回最终文本。"
        ),
    )
    async def polish_and_enhance(text: str) -> str:
        return await processor.polish_and_enhance(text)

    @mcp.tool(
        name="emotion_polish_text",
        description=(
            "对感情向、评论向或个人叙述类文章去 AI 腔，保留原文人称、立场与叙事顺序。"
            "参数 text 为待润色的文本；返回润色后的文本。"
        ),
    )
    async def emotion_polish_text(text: str) -> str:
        return await processor.emotion_polish(text)

    return mcp
