from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.insights_agent import run_insights_agent
from src.agents.metrics_agent import run_metrics_agent
from src.agents.supervisor import (
    SupervisorState,
    decide_next_node,
    supervisor_node,
)


def _human_node(state: SupervisorState) -> SupervisorState:
    """Placeholder for human-in-the-loop interaction.

    When the supervisor routes here, execution pauses for human input.
    Extend this to integrate with your preferred UI / messaging system.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[human] Human-in-the-loop requested. Returning to supervisor.")
    # In production, you'd wait for human input and update state here.
    return state


def build_workflow():
    """Build LangGraph supervisor workflow with dynamic LLM routing."""
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("metrics_agent", run_metrics_agent)
    graph.add_node("insights_agent", run_insights_agent)
    graph.add_node("human", _human_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Dynamic routing decided by LLM — supports any registered agent + human
    graph.add_conditional_edges(
        "supervisor",
        decide_next_node,
        {
            "metrics_agent": "metrics_agent",
            "insights_agent": "insights_agent",
            "human": "human",
            "end": END,
        },
    )

    # All agents return to supervisor for next routing decision
    graph.add_edge("metrics_agent", "supervisor")
    graph.add_edge("insights_agent", "supervisor")
    graph.add_edge("human", "supervisor")

    return graph.compile()


__all__ = ["build_workflow"]

