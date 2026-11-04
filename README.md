# AImcp

学术写作与去 AI 痕迹的 MCP Server。把 BypassAIGC 的核心逻辑提取成一个独立、轻量的服务，
通过 4 个工具供 Claude Desktop / Cursor / 任意 MCP 宿主调用，支持 **stdio** 与 **remote**（streamable-http）两种模式。

> 核心逻辑完全由提示词驱动：去痕规则 + 输出约束 + 任务提示词，见 `src/aimcp/prompts.py`。

## 工具

| 工具 | 说明 |
|------|------|
| `polish_text` | 学术/技术文本第一遍润色，去 AI 腔 |
| `enhance_text` | 第二遍去痕增强（治均匀句长、同义词轮换、套话） |
| `polish_and_enhance` | 先润色再增强，两遍串联 |
| `emotion_polish_text` | 感情向 / 个人叙述文章去 AI 腔 |

所有工具的入参都是 `text: str`，返回改写后的文本。长文本会自动分段、逐段携带前文风格上下文处理，再拼接。

## 快速开始

```bash
cd /home/dev/AImcp
uv sync
cp .env.example .env      # 填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
```

## 配置

全部通过环境变量或 `.env` 设置，常用项：

| 变量 | 说明 | 默认 |
|------|------|------|
| `LLM_API_KEY` | 大模型密钥（必填） | — |
| `LLM_BASE_URL` | OpenAI 兼容基址，带 `/v1` | `http://127.0.0.1:8000/v1` |
| `LLM_MODEL` | 默认模型 | `gemini-2.5-pro` |
| `POLISH_MODEL` / `ENHANCE_MODEL` / `EMOTION_MODEL` | 分阶段覆盖模型（可选） | 回退 `LLM_MODEL` |
| `LLM_API_FORMAT` | 接口格式：`chat` 或 `responses` | `chat` |
| `THINKING_MODE_ENABLED` | 启用思考模式（走 `reasoning_effort`，与 temperature 互斥） | `false` |
| `THINKING_MODE_EFFORT` | 思考强度 | `high` |
| `SEGMENT_MAX_CHARS` | 单段最大长度（汉字/字母数） | `800` |
| `SEGMENT_SKIP_THRESHOLD` | 短于该值视为标题、原样透传 | `15` |
| `HISTORY_MAX_SEGMENTS` | 每段携带的前文段数 | `5` |
| `MCP_TRANSPORT` | `stdio` / `streamable-http` / `sse` | `stdio` |
| `MCP_HOST` / `MCP_PORT` | remote 模式监听地址 | `127.0.0.1` / `8000` |

### 接口格式（chat / responses）

`LLM_API_FORMAT` 决定调用哪种 OpenAI 接口：

| 值 | 端点 | 说明 |
|----|------|------|
| `chat` | `/v1/chat/completions` | 兼容性最好，几乎所有网关都支持（默认） |
| `responses` | `/v1/responses` | OpenAI Responses API，直连 OpenAI 或已支持该接口的网关 |

映射关系：`system` → `instructions`，历史与当前段落 → `input`，`max_tokens` → `max_output_tokens`，
`reasoning_effort` → `reasoning={"effort": ...}`。

开启 `THINKING_MODE_ENABLED` 后，优先用推理参数（与 `temperature` 互斥）；
若调用失败且错误可重试，会自动降级为 `temperature` 重试一次。

## 运行

### stdio（本地宿主）

```bash
uv run python -m aimcp
```

宿主配置：

```json
{
  "mcpServers": {
    "aimcp": {
      "command": "uv",
      "args": ["--directory", "/home/dev/AImcp", "run", "python", "-m", "aimcp"],
      "env": {
        "LLM_API_KEY": "sk-...",
        "LLM_BASE_URL": "http://HOST:PORT/v1",
        "LLM_MODEL": "gemini-2.5-pro"
      }
    }
  }
}
```

### remote（streamable-http）

```bash
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000 uv run python -m aimcp
```

服务地址为 `http://HOST:8000/mcp`，宿主侧：

```json
{
  "mcpServers": {
    "aimcp": { "url": "http://HOST:8000/mcp" }
  }
}
```

## 开发

```bash
uv run pytest        # 纯逻辑测试
uv run ruff check .
```

## 目录结构

```
src/aimcp/
├── __main__.py     # 入口：选择传输模式
├── config.py       # 配置
├── prompts.py      # 去痕规则与任务提示词（核心逻辑）
├── text.py         # 长度统计与分段
├── llm.py          # OpenAI 兼容客户端
├── processor.py    # 分段 + 上下文管线
└── server.py       # FastMCP 与 4 个工具
```
