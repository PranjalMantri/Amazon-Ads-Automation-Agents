from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.insights_agent import run_insights_agent
from src.agents.metrics_agent import run_metrics_agent
from src.agents.supervisor import (
    SupervisorState,
    decide_next_node,
    supervisor_node,
)


def build_workflow():
    """Build LangGraph supervisor workflow."""
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("metrics_agent", run_metrics_agent)
    graph.add_node("insights_agent", run_insights_agent)

    # Entry point
    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        decide_next_node,
        {
            "metrics_agent": "metrics_agent",
            "insights_agent": "insights_agent",
            "end": END,
        },
    )

    graph.add_edge("metrics_agent", "supervisor")
    graph.add_edge("insights_agent", "supervisor")

    return graph.compile()


__all__ = ["build_workflow"]

