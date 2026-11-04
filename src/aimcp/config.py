"""配置：通过环境变量 / .env 读取。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # LLM 后端（OpenAI 兼容）
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "http://127.0.0.1:8000/v1"
    LLM_MODEL: str = "gemini-2.5-pro"
    POLISH_MODEL: str = ""
    ENHANCE_MODEL: str = ""
    EMOTION_MODEL: str = ""
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT_S: float = 60
    # 接口格式：chat（/v1/chat/completions）| responses（/v1/responses）
    LLM_API_FORMAT: str = "chat"
    # 流式输出：部分网关非流式不回填正文，需开启流式
    LLM_STREAM: bool = False
    # 思考模式：开启后走 reasoning_effort（与 temperature 互斥）
    THINKING_MODE_ENABLED: bool = False
    THINKING_MODE_EFFORT: str = "high"

    # 分段处理
    SEGMENT_MAX_CHARS: int = 800
    SEGMENT_SKIP_THRESHOLD: int = 15
    HISTORY_MAX_SEGMENTS: int = 5

    # 传输
    MCP_TRANSPORT: str = "stdio"
    MCP_HOST: str = "127.0.0.1"
    MCP_PORT: int = 8000

    def resolve_model(self, stage: str) -> str:
        """按阶段取模型名，未单独配置时回退到 LLM_MODEL。"""
        override = {
            "polish": self.POLISH_MODEL,
            "enhance": self.ENHANCE_MODEL,
            "emotion_polish": self.EMOTION_MODEL,
        }.get(stage)
        return override or self.LLM_MODEL

    def normalize_transport(self) -> str:
        """归一化传输名，兼容 http / streamable_http 等写法。"""
        transport = self.MCP_TRANSPORT.strip().lower()
        if transport in {"http", "streamable_http", "streamablehttp"}:
            return "streamable-http"
        return transport

    def normalize_api_format(self) -> str:
        """归一化接口格式，兼容 responses_api / response 等写法。"""
        api_format = self.LLM_API_FORMAT.strip().lower()
        if api_format in {"responses", "response", "responses_api"}:
            return "responses"
        return "chat"

    @property
    def reasoning_effort(self) -> str | None:
        """思考模式下返回思考强度，否则返回 None。"""
        return self.THINKING_MODE_EFFORT if self.THINKING_MODE_ENABLED else None


def load_settings() -> Settings:
    """加载配置并做启动校验。"""
    settings = Settings()
    if not settings.LLM_API_KEY:
        raise SystemExit("未配置 LLM_API_KEY，请在 .env 或宿主环境变量中设置")
    if settings.LLM_API_FORMAT.strip().lower() not in {
        "chat",
        "responses",
        "response",
        "responses_api",
    }:
        raise SystemExit(
            f"不支持的 LLM_API_FORMAT: {settings.LLM_API_FORMAT!r}（可选 chat / responses）"
        )
    return settings
