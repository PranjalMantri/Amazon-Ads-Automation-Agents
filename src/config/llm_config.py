from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

CLAUDE_HAIKU_MODEL = "claude-3-haiku-20240307"
LLM_LOG_FILE = "llm_logs.jsonl"

def _get_anthropic_api_key() -> str:
    """Retrieve API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it in .env before running."
        )
    return api_key

def log_llm_event(event_type: str, data: Any, model_name: str, request_id: str | None = None):
    """Append an LLM invocation event (input or output) to *LLM_LOG_FILE*."""
    timestamp = datetime.datetime.now().isoformat()

    try:
        if event_type == "input":
            if hasattr(data, "messages"):
                content = [m.content for m in data.messages]
            elif isinstance(data, list):
                content = [str(m) for m in data]
            elif isinstance(data, str):
                content = data
            else:
                content = str(data)
        else:
            output_data: dict[str, Any] = {}
            if hasattr(data, "content"):
                output_data["content"] = data.content
            if hasattr(data, "tool_calls") and data.tool_calls:
                output_data["tool_calls"] = data.tool_calls
            if hasattr(data, "response_metadata"):
                output_data["usage"] = (
                    data.response_metadata.get("token_usage")
                    or data.response_metadata.get("usage")
                )
            content = output_data or str(data)

        entry = {
            "timestamp": timestamp,
            "request_id": request_id,
            "model": model_name,
            "type": event_type,
            "data": content,
        }

        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    except Exception as exc:
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": timestamp,
                "request_id": request_id,
                "model": model_name,
                "type": "error",
                "error": str(exc),
                "raw_data_str": str(data),
            }) + "\n")


class LoggingChatAnthropic(ChatAnthropic):
    """ChatAnthropic subclass that logs every invocation to *LLM_LOG_FILE*."""

    def invoke(self, input, config=None, **kwargs):
        request_id = str(uuid.uuid4())
        log_llm_event("input", input, self.model, request_id)
        result = super().invoke(input, config=config, **kwargs)
        log_llm_event("output", result, self.model, request_id)
        return result

    async def ainvoke(self, input, config=None, **kwargs):
        request_id = str(uuid.uuid4())
        log_llm_event("input", input, self.model, request_id)
        result = await super().ainvoke(input, config=config, **kwargs)
        log_llm_event("output", result, self.model, request_id)
        return result


def _base_llm(
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = CLAUDE_HAIKU_MODEL,
    timeout: Optional[float] = None,
) -> ChatAnthropic:
    """Create a configured ``LoggingChatAnthropic`` instance."""
    return LoggingChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=_get_anthropic_api_key(),
    )


def get_metrics_llm() -> ChatAnthropic:
    return _base_llm(temperature=0.0, max_tokens=1024, model=CLAUDE_HAIKU_MODEL)


def get_insights_llm() -> ChatAnthropic:
    return _base_llm(temperature=0.4, max_tokens=4096, model=CLAUDE_HAIKU_MODEL)


__all__ = [
    "CLAUDE_HAIKU_MODEL",
    "get_metrics_llm",
    "get_insights_llm",
]

