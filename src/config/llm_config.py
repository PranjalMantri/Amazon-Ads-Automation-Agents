from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_MODEL = "gemini-3-flash-preview"

def _get_google_api_key() -> str:
    """Retrieve API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set. "
            "Set it in .env before running."
        )
    return api_key


def _base_llm(
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = GEMINI_MODEL,
    timeout: Optional[float] = None,
) -> ChatGoogleGenerativeAI:
    """Create a configured ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        api_key=_get_google_api_key(),
    )


def get_metrics_llm() -> ChatGoogleGenerativeAI:
    return _base_llm(temperature=0.0, max_tokens=1024)


def get_insights_llm() -> ChatGoogleGenerativeAI:
    return _base_llm(temperature=0.4, max_tokens=2048)


__all__ = [
    "CLAUDE_HAIKU_MODEL",
    "get_metrics_llm",
    "get_insights_llm",
]

