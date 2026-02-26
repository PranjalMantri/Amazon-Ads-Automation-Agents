"""
Agent definitions for the Amazon Ads multi-agent system.

This package exposes:
- Metrics computation agent (pure Python, zero LLM calls)
- Insights reasoning agent (1 LLM call)
- Supervisor orchestration utilities (deterministic routing)
"""

from .metrics_agent import run_metrics_agent

__all__ = [
    "run_metrics_agent",
]

