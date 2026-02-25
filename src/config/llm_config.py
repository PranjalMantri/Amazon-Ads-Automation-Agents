from __future__ import annotations

import os
from typing import Optional

from langchain_anthropic import ChatAnthropic


CLAUDE_HAIKU_MODEL = "claude-3-haiku-20240307"


def _get_anthropic_api_key() -> str:
    """Retrieve API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before running the Amazon Ads agents."
        )
    return api_key


def _base_llm(
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = CLAUDE_HAIKU_MODEL,
    timeout: Optional[float] = None,
) -> ChatAnthropic:
    """Create a configured ChatAnthropic instance."""
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        api_key=_get_anthropic_api_key(),
    )


def get_metrics_llm() -> ChatAnthropic:
    """Metrics agent uses deterministic outputs (temp=0)."""
    return _base_llm(temperature=0.0, max_tokens=1024)


def get_insights_llm() -> ChatAnthropic:
    """Insights agent uses creative outputs (temp=0.2)."""
    return _base_llm(temperature=0.2, max_tokens=2048)


__all__ = [
    "CLAUDE_HAIKU_MODEL",
    "get_metrics_llm",
    "get_insights_llm",
]

