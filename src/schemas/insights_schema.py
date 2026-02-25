from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .metrics_schema import (
    AccountSummary,
    CampaignMetrics,
    ProductMetrics,
    SearchTermMetrics,
)


class PerformanceOverview(BaseModel):
    """
    High-level qualitative and quantitative description of performance.

    This can embed selected numeric fields from the metrics bundle but should not
    introduce new calculations beyond what was already computed.
    """

    account_summary: AccountSummary
    key_trends: List[str] = Field(
        default_factory=list,
        description="Bullet points describing the most important performance trends.",
    )
    strategic_theme: Optional[str] = Field(
        default=None,
        description="Short phrase capturing the overall performance theme.",
    )


class CampaignInsight(BaseModel):
    """Classification and commentary for a single campaign."""

    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    rationale: str = Field(
        ...,
        description="Why this campaign is in this bucket, grounded in provided metrics.",
    )
    referenced_metrics: Optional[CampaignMetrics] = None


class SearchTermAction(BaseModel):
    """Recommended action for a specific search term."""

    search_term: str
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    action_reason: str = Field(
        ...,
        description="Why this action is recommended, grounded in provided metrics.",
    )
    referenced_metrics: Optional[SearchTermMetrics] = None


class ProductInsight(BaseModel):
    """Insight for a specific product / ASIN."""

    asin: Optional[str] = None
    sku: Optional[str] = None
    product_name: Optional[str] = None
    insight_reason: str = Field(
        ...,
        description="Explanation of why this product falls into this category.",
    )
    referenced_metrics: Optional[ProductMetrics] = None


class CampaignInsightsSection(BaseModel):
    scale_candidates: List[CampaignInsight] = Field(default_factory=list)
    optimization_needed: List[CampaignInsight] = Field(default_factory=list)
    pause_candidates: List[CampaignInsight] = Field(default_factory=list)


class SearchTermActionsSection(BaseModel):
    increase_bids: List[SearchTermAction] = Field(default_factory=list)
    add_negative_keywords: List[SearchTermAction] = Field(default_factory=list)


class ProductInsightsSection(BaseModel):
    hero_products: List[ProductInsight] = Field(default_factory=list)
    budget_drainers: List[ProductInsight] = Field(default_factory=list)


class InsightsReport(BaseModel):
    """
    Final structured output of the Insights Agent.

    This is what is ultimately returned to the caller alongside the raw metrics bundle.
    """

    performance_overview: PerformanceOverview
    campaign_insights: CampaignInsightsSection
    search_term_actions: SearchTermActionsSection
    product_insights: ProductInsightsSection
    budget_reallocation: List[str] = Field(
        default_factory=list,
        description="Recommendations on how to shift budget across entities.",
    )
    priority_actions: List[str] = Field(
        default_factory=list,
        description="Ordered list of the most important next steps.",
    )
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Any notable risks, uncertainties, or data quality concerns.",
    )
    natural_language_summary: str = Field(
        ...,
        description="Executive-style narrative summary of performance and actions.",
    )

    class Config:
        extra = "forbid"

