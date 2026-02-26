from typing import Any, Dict

from src.framework.agent import Agent
from src.config.llm_config import get_metrics_llm
from src.schemas.metrics_schema import MetricsBundle
from src.tools.data_loader_tools import (
    list_available_datasets,
    get_dataset_schema,
    get_dataset_sample,
)
from src.tools.metrics_tools import (
    compute_account_summary,
    compute_campaign_metrics,
    compute_product_metrics,
    compute_search_term_metrics,
)

METRICS_AGENT_SYSTEM_PROMPT = """You are the Metrics Computation Agent in an Amazon Ads analytics system.

Your goal is to compute performance metrics from available datasets.

Process:
1. List available datasets.
2. Call `compute_campaign_metrics`, `compute_search_term_metrics`, and `compute_product_metrics` tools.
   - Use the dataset names you discovered (usually 'sponsored_display', 'sponsored_brands', etc.).
3. The tools return lists of metrics.
4. Call `compute_account_summary` using the results from the previous step.
5. Review the gathered data.
6. Call the final submission tool with the complete `MetricsBundle`.

Ensure you populate the `report_metadata` with the start and end dates provided in the context.
"""

def get_metrics_agent_tools() -> list:
    return [
        list_available_datasets,
        get_dataset_schema,
        get_dataset_sample,
        compute_campaign_metrics,
        compute_search_term_metrics,
        compute_product_metrics,
        compute_account_summary,
    ]

metrics_llm = get_metrics_llm()
metrics_tools = get_metrics_agent_tools()

metrics_agent = Agent(
    name="metrics_agent",
    model=metrics_llm,
    tools=metrics_tools,
    system_prompt=METRICS_AGENT_SYSTEM_PROMPT,
    response_format=MetricsBundle,
    context_keys=["start_date", "end_date"],
    output_key="metrics_bundle"
)

def run_metrics_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    return metrics_agent.run(state)
