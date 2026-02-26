from typing import Any, Dict

from src.framework.agent import Agent
from src.config.llm_config import get_insights_llm
from src.schemas.insights_schema import InsightsReport

INSIGHTS_AGENT_SYSTEM_PROMPT = """You are the Insights Agent in an Amazon Ads analytics system.

Your role:
- Act as a senior Amazon Ads marketing strategist.
- Interpret precomputed performance metrics provided in the context.
- Identify patterns, surface opportunities and risks, and recommend clear actions.

Hard constraints:
- You MUST NOT perform any explicit arithmetic or recompute metrics.
- You MUST ONLY use the structured metrics bundle provided in the context.

Tone and style:
- Think like a performance marketing lead reporting to an executive.
- Be concise, actionable, and prioritized by business impact.
"""

insights_llm = get_insights_llm()

insights_agent = Agent(
    name="insights_agent",
    model=insights_llm,
    tools=[], 
    system_prompt=INSIGHTS_AGENT_SYSTEM_PROMPT,
    response_format=InsightsReport,
    context_keys=["metrics_bundle"],
    output_key="insights_report"
)

def run_insights_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    return insights_agent.run(state)
