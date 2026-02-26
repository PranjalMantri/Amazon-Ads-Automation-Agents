from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config.llm_config import get_metrics_llm
from src.schemas.insights_schema import InsightsReport
from src.schemas.metrics_schema import MetricsBundle
from src.framework.agent_registry import AgentRegistry


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
    """Route to next agent or end using LLM router."""
    
    metrics_status = "Calculated" if state.get("metrics_bundle") else "Not calculated"
    insights_status = "Generated" if state.get("insights_report") else "Not generated"

    # Fetch registered agents dynamically
    agents = AgentRegistry.get_all_agents()
    agent_descriptions = "\n".join([f"- {agent['name']}: {agent['description']}" for agent in agents])
    
    # Extract agent names for validation, filtering out non-agent keys if any
    valid_agent_names = [agent['name'] for agent in agents]
    # Add 'end' to the valid options
    valid_options = valid_agent_names + ["end"]

    system = """You are a supervisor routing a user request.
    Your goal is to complete the user request by delegating to specialized workers.
    
    Available Workers:
    {agent_descriptions}
    - end: Use this if the user request is satisfied or both metrics and insights are done.
    
    Context:
    Metrics Status: {metrics_status}
    Insights Status: {insights_status}
    
    User Request: {user_request}
    
    Provide the name of the next worker to route to.
    """
    
    prompt = ChatPromptTemplate.from_template(system)
    llm = get_metrics_llm()
    
    class DynamicRouteQuery(BaseModel):
        next_node: str = Field(..., description=f"The next node to run. Must be one of: {', '.join(valid_options)}")

    structured_llm = llm.with_structured_output(DynamicRouteQuery)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "metrics_status": metrics_status, 
            "insights_status": insights_status, 
            "user_request": state.get("user_request", ""),
            "agent_descriptions": agent_descriptions
        })
        decision = result.next_node
        
        # Validation
        if decision not in valid_options:
             print(f"[Supervisor] LLM returned invalid node '{decision}'. defaulting...")
             raise ValueError("Invalid node selected")
             
    except Exception as e:
        # Fallback if LLM fails
        print(f"[Supervisor] LLM routing failed: {e}. Falling back to logic.")
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

