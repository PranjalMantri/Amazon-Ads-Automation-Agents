from __future__ import annotations

import logging
from typing import Optional, TypedDict

from pydantic import BaseModel, Field

from src.config.llm_config import get_metrics_llm
from src.schemas.insights_schema import InsightsReport
from src.schemas.metrics_schema import MetricsBundle
from src.framework.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class SupervisorState(TypedDict, total=False):
    """Workflow state container."""
    user_request: str
    start_date: Optional[str]
    end_date: Optional[str]
    metrics_bundle: Optional[MetricsBundle]
    insights_report: Optional[InsightsReport]


class RouteDecision(BaseModel):
    """LLM-produced routing decision."""
    next_node: str = Field(
        ...,
        description="The next agent to run, or 'end' to finish, or 'human' to request human input.",
    )
    reason: Optional[str] = Field(
        None,
        description="One-sentence rationale (helps with debugging, not sent downstream).",
    )


def initialize_state(
    user_request: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> SupervisorState:
    """Initialize state from user input."""
    return SupervisorState(
        user_request=user_request,
        start_date=start_date,
        end_date=end_date,
        metrics_bundle=None,
        insights_report=None,
    )


def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Pass-through node for logging."""
    logger.info("[Supervisor] Node visited. Keys: %s", list(state.keys()))
    return state


# ── Ultra-compact routing prompt (minimises token usage) ──────────────
_ROUTE_PROMPT = """You are a workflow supervisor. Pick the next worker.

Workers:
{agent_list}
- end: all tasks done
- human: need human input

Status: metrics={metrics_status}, insights={insights_status}
Request: {user_request}

Respond with next_node only."""


def decide_next_node(state: SupervisorState) -> str:
    """Route to the next agent via a single, token-efficient LLM call.

    Supports dynamic agent discovery and human-in-the-loop.
    Falls back to deterministic logic if the LLM call fails.
    """
    metrics_status = "done" if state.get("metrics_bundle") else "pending"
    insights_status = "done" if state.get("insights_report") else "pending"

    # Dynamic agent discovery
    agents = AgentRegistry.get_all_agents()
    agent_list = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    valid_nodes = [a["name"] for a in agents] + ["end", "human"]

    llm = get_metrics_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    try:
        result: RouteDecision = structured_llm.invoke(
            _ROUTE_PROMPT.format(
                agent_list=agent_list,
                metrics_status=metrics_status,
                insights_status=insights_status,
                user_request=state.get("user_request", ""),
            )
        )
        decision = result.next_node

        if decision not in valid_nodes:
            logger.warning("[Supervisor] Invalid node '%s', falling back.", decision)
            raise ValueError(f"Invalid node: {decision}")

    except Exception as exc:
        logger.warning("[Supervisor] LLM routing failed (%s). Using fallback.", exc)
        if state.get("metrics_bundle") is None:
            decision = "metrics_agent"
        elif state.get("insights_report") is None:
            decision = "insights_agent"
        else:
            decision = "end"

    logger.info("[Supervisor] Routing -> %s", decision)
    return decision


__all__ = [
    "SupervisorState",
    "initialize_state",
    "supervisor_node",
    "decide_next_node",
]

