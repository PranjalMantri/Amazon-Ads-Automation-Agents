"""
Agent definitions for the Amazon Ads multi-agent system.

This package exposes:
- Metrics computation agent
- Insights reasoning agent
- Supervisor orchestration utilities
"""

from .metrics_agent import METRICS_AGENT_SYSTEM_PROMPT, get_metrics_agent_tools, run_metrics_agent

__all__ = [
    "METRICS_AGENT_SYSTEM_PROMPT",
    "get_metrics_agent_tools",
    "run_metrics_agent",
]

