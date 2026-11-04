"""MCP Server 入口：按 MCP_TRANSPORT 选择 stdio / remote 模式运行。"""

from __future__ import annotations

from .config import load_settings
from .server import create_server

_SUPPORTED_TRANSPORTS = {"stdio", "streamable-http", "sse"}


def main() -> None:
    settings = load_settings()
    transport = settings.normalize_transport()

    if transport not in _SUPPORTED_TRANSPORTS:
        raise SystemExit(
            f"不支持的 MCP_TRANSPORT: {settings.MCP_TRANSPORT!r}"
            "（可选 stdio / streamable-http / sse）"
        )

    server = create_server(settings)
    server.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
