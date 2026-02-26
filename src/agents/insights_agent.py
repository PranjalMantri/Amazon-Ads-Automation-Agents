"""Insights Agent — exactly 1 LLM call with structured output.

Uses compact pre-serialized JSON context and structured output binding
to guarantee a single round-trip with no retries.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.llm_config import get_insights_llm
from src.schemas.insights_schema import InsightsReport
from src.framework.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)

# Register so the supervisor can dynamically discover this agent
AgentRegistry.register_agent(
    name="insights_agent",
    description="Senior Ads Strategist: interprets metrics to produce insights and actions.",
)

INSIGHTS_SYSTEM_PROMPT = """You are a Senior Amazon Ads Strategist. Analyze the metrics below and return structured JSON.

Your analysis MUST include:
1. Performance overview with key_trends, strategic_theme, and the account_summary (copy it verbatim).
2. Campaign insights: classify into scale_candidates, optimization_needed, pause_candidates.
3. Search term actions: increase_bids vs add_negative_keywords.
4. Product insights: hero_products vs budget_drainers.
5. budget_reallocation, priority_actions, risk_flags (string lists).
6. natural_language_summary (executive narrative).

Rules:
- Ground every insight in the provided metrics. Do NOT invent numbers.
- Be concise. Each rationale/reason should be 1-2 sentences max.
- Return ONLY valid JSON matching the InsightsReport schema."""


def run_insights_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insights with exactly 1 LLM call using structured output."""
    logger.info("[insights_agent] Starting (1 LLM call)...")

    metrics_bundle = state.get("metrics_bundle")
    if metrics_bundle is None:
        logger.error("[insights_agent] No metrics_bundle in state!")
        return {"insights_report": None}

    # Serialize metrics to compact JSON — avoids Python repr issues
    if hasattr(metrics_bundle, "model_dump"):
        metrics_json = json.dumps(metrics_bundle.model_dump(mode="json"), default=str)
    elif isinstance(metrics_bundle, dict):
        metrics_json = json.dumps(metrics_bundle, default=str)
    else:
        metrics_json = str(metrics_bundle)

    # Single LLM call with structured output
    llm = get_insights_llm()
    structured_llm = llm.with_structured_output(InsightsReport)

    messages = [
        SystemMessage(content=INSIGHTS_SYSTEM_PROMPT),
        HumanMessage(content=f"Metrics Data:\n{metrics_json}"),
    ]

    try:
        insights_report: InsightsReport = structured_llm.invoke(messages)
        logger.info("[insights_agent] Insights generated successfully (1 call).")
        return {"insights_report": insights_report}
    except Exception as exc:
        logger.error("[insights_agent] LLM call failed: %s", exc, exc_info=True)
        return {"insights_report": None}
