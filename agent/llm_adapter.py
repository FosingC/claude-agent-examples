"""
LLM 适配器：用 Python 标准库调用 Kimi / Qwen 的 OpenAI 兼容接口，
无需安装 openai 包。
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 数据类：模拟 Anthropic SDK 的消息结构
# ---------------------------------------------------------------------------

@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class _Message:
    content: list
    stop_reason: str | None = None
    usage: _Usage | None = None


# ---------------------------------------------------------------------------
# 工具 / 消息格式转换
# ---------------------------------------------------------------------------

def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return result


def _openai_messages_from_anthropic(
    system: str | None,
    messages: list[dict],
) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # user 的 tool_results
        if role == "user" and isinstance(content, list):
            tool_msgs = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(block.get("content", "")),
                    })
                else:
                    text = block.get("text", "") if isinstance(block, dict) else str(block)
                    tool_msgs.append({"role": "user", "content": text})
            out.extend(tool_msgs)
            continue

        # assistant 的 content 可能是 _TextBlock / _ToolUseBlock 列表
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                if btype == "text":
                    t = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
                    text_parts.append(str(t))
                elif btype == "tool_use":
                    tid = getattr(block, "id", None) or (block.get("id") if isinstance(block, dict) else "")
                    name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else "")
                    inp = getattr(block, "input", None) or (block.get("input") if isinstance(block, dict) else {})
                    tool_calls.append({
                        "id": str(tid),
                        "type": "function",
                        "function": {
                            "name": str(name),
                            "arguments": json.dumps(inp, ensure_ascii=False),
                        },
                    })
            assistant_msg: dict = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "\n".join(text_parts)
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
            continue

        out.append({"role": role, "content": content})
    return out


def _openai_finish_to_stop(finish_reason: str | None) -> str | None:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    return mapping.get(finish_reason, finish_reason)


def _openai_response_to_message(body: dict) -> _Message:
    choice = body["choices"][0]
    content_blocks: list[Any] = []

    text = choice["message"].get("content") or ""
    if text:
        content_blocks.append(_TextBlock(text=text))

    for tc in (choice["message"].get("tool_calls") or []):
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        content_blocks.append(_ToolUseBlock(
            id=tc["id"],
            name=tc["function"]["name"],
            input=args,
        ))

    usage = None
    u = body.get("usage")
    if u:
        usage = _Usage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
        )

    return _Message(
        content=content_blocks,
        stop_reason=_openai_finish_to_stop(choice.get("finish_reason")),
        usage=usage,
    )


# ---------------------------------------------------------------------------
# HTTP 请求（标准库）
# ---------------------------------------------------------------------------

def _chat_completion(
    base_url: str,
    api_key: str,
    payload: dict,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 对外 Client 类
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = (api_key or os.environ.get("LLM_API_KEY") or "").strip()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").strip()

        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY 未设置。请在 .env 文件中填写你的 API Key，例如：\n"
                "  LLM_API_KEY=your_kimi_api_key\n"
                "  LLM_BASE_URL=https://api.moonshot.cn/v1"
            )

    class Messages:
        def __init__(self, client: LLMClient):
            self._client = client

        def create(
            self,
            *,
            model: str,
            max_tokens: int | None = None,
            system: str | None = None,
            tools: list[dict] | None = None,
            messages: list[dict] | None = None,
            **kwargs,
        ) -> _Message:
            openai_messages = _openai_messages_from_anthropic(system, messages or [])
            openai_tools = _anthropic_tools_to_openai(tools) if tools else None

            payload: dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if openai_tools:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"

            body = _chat_completion(
                self._client.base_url,
                self._client.api_key,
                payload,
            )
            return _openai_response_to_message(body)

    @property
    def messages(self):
        return self.Messages(self)


# 兼容旧 import
Anthropic = LLMClient
