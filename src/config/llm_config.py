from __future__ import annotations

import os
import json
import datetime
from typing import Optional, Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

CLAUDE_HAIKU_MODEL = "claude-3-haiku-20240307"

def _get_anthropic_api_key() -> str:
    """Retrieve API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it in .env before running."
        )
    return api_key

def log_llm_event(event_type: str, data: Any, model_name: str, request_id: str = None):
    """Log LLM event (input or output) to a file."""
    log_file = "llm_logs.jsonl"
    timestamp = datetime.datetime.now().isoformat()
    
    try:
        content = None
        if event_type == "input":
            # Convert input to a serializable format
            if hasattr(data, "messages"):
                content = [m.content for m in data.messages]
            elif isinstance(data, list):
                content = [str(m) for m in data]
            elif isinstance(data, str):
                content = data
            else:
                content = str(data)
        else: # output
            output_data = {}
            if hasattr(data, "content"):
                output_data["content"] = data.content
            
            if hasattr(data, "tool_calls") and data.tool_calls:
                # specific handling for tool calls which might need serialization
                output_data["tool_calls"] = data.tool_calls
            
            if hasattr(data, "response_metadata"):
                output_data["usage"] = data.response_metadata.get("token_usage") or data.response_metadata.get("usage")

            if not output_data:
                 content = str(data)
            else:
                 content = output_data

        entry = {
            "timestamp": timestamp,
            "request_id": request_id,
            "model": model_name,
            "type": event_type,
            "data": content
        }
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            
    except Exception as e:
        # Fallback logging
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": timestamp,
                "request_id": request_id,
                "model": model_name,
                "type": "error",
                "error": str(e),
                "raw_data_str": str(data)
            }) + "\n")


class LoggingChatAnthropic(ChatAnthropic):
    """Wrapper to log inputs and outputs before calling the actual model."""
    
    def invoke(self, input, config=None, **kwargs):
        import uuid
        request_id = str(uuid.uuid4())
        log_llm_event("input", input, self.model, request_id)
        result = super().invoke(input, config=config, **kwargs)
        log_llm_event("output", result, self.model, request_id)
        return result

    async def ainvoke(self, input, config=None, **kwargs):
        import uuid
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
    """Create a configured ChatAnthropic instance."""
    # Use the logging wrapper
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

