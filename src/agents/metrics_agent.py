from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Sequence
from src.schemas.metrics_schema import (
    CampaignMetrics,
    MetricsBundle,
    ProductMetrics,
    ReportMetadata,
    SearchTermMetrics,
    AccountSummary,
)
from src.tools.data_loader_tools import (
    load_sd_product_data,
    load_sb_search_term_data,
)
from src.tools.metrics_tools import (
    compute_account_summary,
    compute_campaign_metrics,
    compute_product_metrics,
    compute_search_term_metrics,
)


METRICS_AGENT_SYSTEM_PROMPT = """
You are the Metrics Computation Agent in an Amazon Ads analytics system.

Your responsibilities:
- Fetch raw data via the provided tools ONLY.
- Use deterministic Python tools to compute ALL performance metrics.
- Assemble a structured metrics bundle that conforms to the `MetricsBundle` schema.

Hard constraints:
- You MUST NOT perform any arithmetic or metric calculations yourself.
- You MUST NOT provide natural-language explanations, interpretations, or recommendations.
- You MUST NOT access raw Excel files directly; only use tools.
- All metrics (spend, sales, orders, impressions, clicks, ctr, cvr, cpc, acos, roas)
  come exclusively from Python tools.

Output requirements:
- Produce a JSON object matching the following shape:
  {
    "report_metadata": {
      "start_date": "YYYY-MM-DD or null",
      "end_date": "YYYY-MM-DD or null",
      "generated_at": "ISO timestamp"
    },
    "account_summary": { ... },
    "campaign_metrics": [ ... ],
    "search_term_metrics": [ ... ],
    "product_metrics": [ ... ]
  }

You operate as a deterministic computation layer, not as a reasoning or reporting agent.
""".strip()


def get_metrics_agent_tools() -> Sequence[Any]:
    """
    Return the full set of tools that the Metrics Agent is allowed to use.

    This is useful for wiring the agent into LangGraph or LangChain executors.
    """
    return [
        load_sd_product_data,
        load_sb_search_term_data,
        compute_campaign_metrics,
        compute_search_term_metrics,
        compute_product_metrics,
        compute_account_summary,
    ]


def _coerce_optional_date(value: Optional[Any]) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    # Accept ISO-like strings from upstream callers.
    return date.fromisoformat(str(value))


def run_metrics_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic orchestration for metrics computation.

    Although a Claude Haiku LLM instance is configured for this agent, it is
    deliberately NOT used for any numerical work. Instead, this function calls
    the underlying Python tools directly, guaranteeing that all metrics are
    computed deterministically in code.

    Expected input state keys:
        - "start_date": Optional[date or ISO string]
        - "end_date": Optional[date or ISO string]

    Output (state is mutated and returned):
        - "metrics_bundle": MetricsBundle instance containing all metrics.
    """
    print("[MetricsAgent] Starting metrics computation.")
    print(f"[MetricsAgent] Input state: start_date={state.get('start_date')}, end_date={state.get('end_date')}")

    # Load raw data via tools (StructuredTool instances).
    sd_rows: List[Dict[str, Any]] = load_sd_product_data.invoke({})
    print(f"[MetricsAgent] Loaded SD product rows: {len(sd_rows)}")
    sb_rows: List[Dict[str, Any]] = load_sb_search_term_data.invoke({})
    print(f"[MetricsAgent] Loaded SB search term rows: {len(sb_rows)}")

    # Compute granular metrics using deterministic Python tools.
    campaign_metrics_raw: List[Dict[str, Any]] = compute_campaign_metrics.invoke(
        {"sd_data": sd_rows, "sb_data": sb_rows}
    )
    print(f"[MetricsAgent] Computed campaign metrics: {len(campaign_metrics_raw)}")
    

    search_term_metrics_raw: List[Dict[str, Any]] = compute_search_term_metrics.invoke(
        {"sb_data": sb_rows}
    )
    print(
        f"[MetricsAgent] Computed search term metrics: "
        f"{len(search_term_metrics_raw)}"
    )

    product_metrics_raw: List[Dict[str, Any]] = compute_product_metrics.invoke(
        {"sd_data": sd_rows}
    )
    print(f"[MetricsAgent] Computed product metrics: {len(product_metrics_raw)}")

    # Compute account-level summary.
    account_summary_raw: Dict[str, Any] = compute_account_summary.invoke(
        {
            "campaign_metrics": campaign_metrics_raw,
            "search_term_metrics": search_term_metrics_raw,
            "product_metrics": product_metrics_raw,
        }
    )
    print("[MetricsAgent] Computed account summary:")
    print(f"  Spend: {account_summary_raw.get('spend', 0.0):.2f}")
    print(f"  Sales: {account_summary_raw.get('sales', 0.0):.2f}")
    print(f"  ACOS:  {account_summary_raw.get('acos', 0.0):.2f}")
    print(f"  ROAS:  {account_summary_raw.get('roas', 0.0):.2f}")

    # Convert raw dictionaries into strongly-typed Pydantic models.
    campaign_models = [CampaignMetrics(**m) for m in campaign_metrics_raw]
    search_term_models = [SearchTermMetrics(**m) for m in search_term_metrics_raw]
    product_models = [ProductMetrics(**m) for m in product_metrics_raw]
    account_summary = AccountSummary(**account_summary_raw)

    start_date = _coerce_optional_date(state.get("start_date"))
    end_date = _coerce_optional_date(state.get("end_date"))

    report_metadata = ReportMetadata(
        start_date=start_date,
        end_date=end_date,
    )

    bundle = MetricsBundle(
        report_metadata=report_metadata,
        account_summary=account_summary,
        campaign_metrics=campaign_models,
        search_term_metrics=search_term_models,
        product_metrics=product_models,
    )

    state = dict(state)
    state["metrics_bundle"] = bundle
    print("[MetricsAgent] Metrics bundle attached to state.")
    return state


__all__ = [
    "METRICS_AGENT_SYSTEM_PROMPT",
    "get_metrics_agent_tools",
    "run_metrics_agent",
]

