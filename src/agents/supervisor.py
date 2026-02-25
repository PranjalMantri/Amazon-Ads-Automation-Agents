from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.schemas.insights_schema import InsightsReport
from src.schemas.metrics_schema import MetricsBundle


class SupervisorState(TypedDict, total=False):
    """Workflow state container."""
    user_request: str
    start_date: Optional[str]
    end_date: Optional[str]
    metrics_bundle: Optional[MetricsBundle]
    insights_report: Optional[InsightsReport]


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
    """Pass-through node for logging/routing."""
    print("[Supervisor] Supervisor node visited.")
    print(
        "[Supervisor] Current state keys: "
        f"{list(state.keys())}"
    )
    return state


def decide_next_node(state: SupervisorState) -> str:
    """Route to next agent or end."""
    if state.get("metrics_bundle") is None:
        decision = "metrics_agent"
    elif state.get("insights_report") is None:
        decision = "insights_agent"
    else:
        decision = "end"

    print(f"[Supervisor] Routing decision: {decision}")
    return decision


__all__ = [
    "SupervisorState",
    "initialize_state",
    "supervisor_node",
    "decide_next_node",
]

