from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from src.config.llm_config import get_insights_llm
from src.schemas.insights_schema import (
    CampaignInsight,
    CampaignInsightsSection,
    InsightsReport,
    PerformanceOverview,
    ProductInsight,
    ProductInsightsSection,
    SearchTermAction,
    SearchTermActionsSection,
)
from src.schemas.metrics_schema import MetricsBundle


INSIGHTS_AGENT_SYSTEM_PROMPT = """
You are the Insights Agent in an Amazon Ads analytics system.

Your role:
- Act as a senior Amazon Ads marketing strategist.
- Interpret precomputed performance metrics.
- Identify patterns, surface opportunities and risks, and recommend clear actions.

Hard constraints:
- You MUST NOT perform any explicit arithmetic or recompute metrics.
- You MUST NOT access or request raw Excel data or row-level logs.
- You MUST ONLY use the structured metrics bundle passed to you as input.
- You MUST keep all reasoning grounded strictly in the provided metrics values.

Input:
- A single `metrics_bundle` object that conforms to the `MetricsBundle` schema.
  This includes:
  - account_summary
  - campaign_metrics[]
  - search_term_metrics[]
  - product_metrics[]

Output:
- You MUST produce a JSON object that strictly matches the following structure:
  {
    "performance_overview": { ... },
    "campaign_insights": {
      "scale_candidates": [],
      "optimization_needed": [],
      "pause_candidates": []
    },
    "search_term_actions": {
      "increase_bids": [],
      "add_negative_keywords": []
    },
    "product_insights": {
      "hero_products": [],
      "budget_drainers": []
    },
    "budget_reallocation": [],
    "priority_actions": [],
    "risk_flags": [],
    "natural_language_summary": ""
  }

Tone and style:
- Think like a performance marketing lead reporting to an executive.
- Be concise, actionable, and prioritised by business impact.
- Keep the natural language summary clear and non-technical.
""".strip()


def _mock_insights_report(metrics_bundle: MetricsBundle) -> InsightsReport:
    """
    Create a deterministic mock InsightsReport used when no LLM is available.

    This allows end-to-end flow testing without an Anthropic API key.
    """
    print("[InsightsAgent] Using MOCK insights report (no API key).")

    account = metrics_bundle.account_summary

    performance_overview = PerformanceOverview(
        account_summary=account,
        key_trends=[
            "Mocked overview: metrics computed successfully.",
            "Replace mock mode with a real Claude API key for production insights.",
        ],
        strategic_theme="Mock evaluation run",
    )

    # For the mock, just surface up to 3 campaigns/products/search terms as examples.
    campaigns = metrics_bundle.campaign_metrics[:3]
    products = metrics_bundle.product_metrics[:3]
    search_terms = metrics_bundle.search_term_metrics[:3]

    campaign_insights = CampaignInsightsSection(
        scale_candidates=[
            CampaignInsight(
                campaign_id=c.campaign_id,
                campaign_name=c.campaign_name,
                rationale="Mock: candidate for scaling based on placeholder logic.",
                referenced_metrics=c,
            )
            for c in campaigns
        ]
    )

    search_term_actions = SearchTermActionsSection(
        increase_bids=[
            SearchTermAction(
                search_term=s.search_term,
                campaign_id=s.campaign_id,
                campaign_name=s.campaign_name,
                action_reason="Mock: consider increasing bids for testing purposes.",
                referenced_metrics=s,
            )
            for s in search_terms
        ]
    )

    product_insights = ProductInsightsSection(
        hero_products=[
            ProductInsight(
                asin=p.asin,
                sku=p.sku,
                product_name=p.product_name,
                insight_reason="Mock: flagged as a hero product for demonstration.",
                referenced_metrics=p,
            )
            for p in products
        ]
    )

    summary_lines = [
        "This is a MOCK executive summary generated without an LLM.",
        "The metrics pipeline and supervisor orchestration executed successfully.",
        "Configure ANTTHROPIC_API_KEY to enable real Claude-powered insights.",
    ]

    return InsightsReport(
        performance_overview=performance_overview,
        campaign_insights=campaign_insights,
        search_term_actions=search_term_actions,
        product_insights=product_insights,
        budget_reallocation=[
            "Mock suggestion: re-evaluate budget allocation once real insights are enabled."
        ],
        priority_actions=[
            "Verify metric correctness.",
            "Connect to live Amazon Ads API.",
            "Enable Claude API key for production insights.",
        ],
        risk_flags=["Running in mock mode: insights are placeholders only."],
        natural_language_summary="\n".join(summary_lines),
    )


def _summarize_metrics_bundle(bundle: MetricsBundle) -> Dict[str, Any]:
    """
    Summarize and truncate the metrics bundle to fit within LLM context limits.
    
    Retains:
    - All account summary and metadata.
    - Top 50 campaigns by spend.
    - Top 50 products by sales.
    - Top 50 search terms by spend (cost drivers).
    - Top 50 search terms by sales (revenue drivers).
    """
    print("[InsightsAgent] Summarizing metrics bundle for LLM context...")
    
    data = bundle.model_dump(mode="json")
    
    # 1. Filter Campaigns (keep top 50 by spend)
    campaigns = data.get("campaign_metrics", [])
    campaigns.sort(key=lambda x: x.get("spend", 0.0), reverse=True)
    top_campaigns = campaigns[:50]
    
    # 2. Filter Products (keep top 50 by sales)
    products = data.get("product_metrics", [])
    products.sort(key=lambda x: x.get("sales", 0.0), reverse=True)
    top_products = products[:50]
    
    # 3. Filter Search Terms (mix of top spenders and top sellers)
    search_terms = data.get("search_term_metrics", [])
    
    # Top spenders usually reveal waste/optimisation needs
    by_spend = sorted(search_terms, key=lambda x: x.get("spend", 0.0), reverse=True)[:50]
    # Top sellers reveal scaling opportunities
    by_sales = sorted(search_terms, key=lambda x: x.get("sales", 0.0), reverse=True)[:50]
    
    # Deduplicate by search_term + campaign_id to create a unified list
    seen = set()
    unique_search_terms = []
    
    for item in by_spend + by_sales:
        key = (item.get("search_term"), item.get("campaign_id"))
        if key not in seen:
            seen.add(key)
            unique_search_terms.append(item)
            
    print(f"[InsightsAgent] Reduced payload stats:")
    print(f"  - Campaigns: {len(campaigns)} -> {len(top_campaigns)}")
    print(f"  - Products: {len(products)} -> {len(top_products)}")
    print(f"  - Search Terms: {len(search_terms)} -> {len(unique_search_terms)}")

    return {
        "report_metadata": data.get("report_metadata"),
        "account_summary": data.get("account_summary"),
        "campaign_metrics": top_campaigns,
        "product_metrics": top_products,
        "search_term_metrics": unique_search_terms,
        "_note": "Data truncated to top performing/spending entities for analysis context."
    }


def run_insights_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Insights Agent over a precomputed MetricsBundle.

    Expected input state keys:
        - "metrics_bundle": MetricsBundle instance (required)

    Output (state is mutated and returned):
        - "insights_report": InsightsReport instance containing all insights.
    """
    print("[InsightsAgent] Starting insights generation.")

    metrics_bundle = state.get("metrics_bundle")
    if metrics_bundle is None:
        raise ValueError(
            "metrics_bundle is missing from state. The Metrics Agent must run "
            "before the Insights Agent."
        )

    if isinstance(metrics_bundle, dict):
        metrics_bundle = MetricsBundle(**metrics_bundle)
    elif not isinstance(metrics_bundle, MetricsBundle):
        raise TypeError(
            "metrics_bundle must be a MetricsBundle or a compatible dict, "
            f"got {type(metrics_bundle)!r}"
        )

    print("[InsightsAgent] Received MetricsBundle as input.")
    print(f"  Start Date: {metrics_bundle.report_metadata.start_date}")
    print(f"  End Date:   {metrics_bundle.report_metadata.end_date}")
    acct = metrics_bundle.account_summary
    if acct:
        print(f"  Account Spend: {acct.spend:.2f} | Sales: {acct.sales:.2f}")

    try:
        llm = get_insights_llm()
        print("[InsightsAgent] Using real Claude LLM for insights.")

        # Enforce structured output to match InsightsReport exactly.
        structured_llm = llm.with_structured_output(InsightsReport)
        
        # Prepare summarized context for the LLM
        summarized_metrics = _summarize_metrics_bundle(metrics_bundle)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INSIGHTS_AGENT_SYSTEM_PROMPT),
                (
                    "user",
                    "Here is the precomputed metrics bundle in JSON form. "
                    "Use ONLY this information to generate insights and recommendations.\n\n"
                    "{metrics_json}",
                ),
            ]
        )

        chain = prompt | structured_llm

        insights_report: InsightsReport = chain.invoke(
            {"metrics_json": summarized_metrics}
        )
    except RuntimeError:
        # Fallback to a deterministic mock report when no API key is configured.
        insights_report = _mock_insights_report(metrics_bundle)

    new_state = dict(state)
    new_state["insights_report"] = insights_report
    print("[InsightsAgent] Insights report attached to state.")
    return new_state


__all__ = [
    "INSIGHTS_AGENT_SYSTEM_PROMPT",
    "run_insights_agent",
]

