"""
Schemas for metrics computation and business insights.
"""

from .metrics_schema import (
    AccountSummary,
    BasePerformanceMetrics,
    CampaignMetrics,
    MetricsBundle,
    ProductMetrics,
    ReportMetadata,
    SearchTermMetrics,
)
from .insights_schema import (
    CampaignInsight,
    InsightsReport,
    ProductInsight,
    SearchTermAction,
)

__all__ = [
    "BasePerformanceMetrics",
    "CampaignMetrics",
    "SearchTermMetrics",
    "ProductMetrics",
    "AccountSummary",
    "ReportMetadata",
    "MetricsBundle",
    "CampaignInsight",
    "SearchTermAction",
    "ProductInsight",
    "InsightsReport",
]

