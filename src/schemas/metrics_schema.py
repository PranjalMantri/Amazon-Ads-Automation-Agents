from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Any, Optional, List

from pydantic import BaseModel, Field, validator


class CampaignType(str, Enum):
    SD = "SD"
    SB = "SB"


class BasePerformanceMetrics(BaseModel):
    """Core numeric performance metrics computed deterministically in Python."""

    spend: float = Field(0.0, ge=0)
    sales: float = Field(0.0, ge=0)
    orders: int = Field(0, ge=0)
    impressions: int = Field(0, ge=0)
    clicks: int = Field(0, ge=0)

    ctr: float = Field(0.0, ge=0)  # clicks / impressions
    cvr: float = Field(0.0, ge=0)  # orders / clicks
    cpc: float = Field(0.0, ge=0)  # spend / clicks
    acos: float = Field(0.0, ge=0)  # spend / sales
    roas: float = Field(0.0, ge=0)  # sales / spend

    class Config:
        extra = "forbid"


class CampaignMetrics(BasePerformanceMetrics):
    """Aggregated metrics at the campaign level."""

    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    campaign_type: Optional[CampaignType] = None
    portfolio_name: Optional[str] = None


class SearchTermMetrics(BasePerformanceMetrics):
    """Aggregated metrics at the search term level."""

    search_term: str
    match_type: Optional[str] = None
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    ad_group_name: Optional[str] = None


class ProductMetrics(BasePerformanceMetrics):
    """Aggregated metrics at the product / ASIN level."""

    asin: Optional[str] = None
    sku: Optional[str] = None
    product_name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None


class AccountSummary(BasePerformanceMetrics):
    """Overall account-level summary across all entities."""

    total_campaigns: int = Field(0, ge=0)
    total_products: int = Field(0, ge=0)
    total_search_terms: int = Field(0, ge=0)

    start_date: Optional[date] = None
    end_date: Optional[date] = None


class ReportMetadata(BaseModel):
    """Metadata describing the time range and generation time of the report."""

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CategoryPerformance(BaseModel):
    """Aggregated performance for a specific category (Top/Bottom by Metric)."""
    category_name: str
    items: List[Any]  # Can hold CampaignMetrics, SearchTermMetrics, etc.

class MetricsBundle(BaseModel):
    """
    Top-level metrics bundle passed from the Metrics Agent to the Insights Agent.
    
    Contains aggregated account summary and filtered lists of top/bottom performers
    to minimize token usage while retaining key insights.
    """

    report_metadata: ReportMetadata
    account_summary: AccountSummary
    
    # Specific slices of interest
    top_campaigns_by_spend: List[CampaignMetrics] = Field(default_factory=list)
    top_campaigns_by_roas: List[CampaignMetrics] = Field(default_factory=list)
    bottom_campaigns_by_roas: List[CampaignMetrics] = Field(default_factory=list)

    top_search_terms_by_spend: List[SearchTermMetrics] = Field(default_factory=list)
    top_search_terms_by_roas: List[SearchTermMetrics] = Field(default_factory=list)
    
    top_products_by_spend: List[ProductMetrics] = Field(default_factory=list)
    top_products_by_roas: List[ProductMetrics] = Field(default_factory=list)
    bottom_products_by_roas: List[ProductMetrics] = Field(default_factory=list)

    class Config:
        extra = "allow" # Allow extra fields if agents want to pass more slices

