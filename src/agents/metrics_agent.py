"""Metrics Agent — deterministic computation with no LLM calls.

All metrics are computed via pandas aggregation and validated with Pydantic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.schemas.metrics_schema import MetricsBundle
from src.tools.data_loader_tools import DATASET_MAPPING
from src.tools.metrics_tools import get_holistic_performance_report_data
from src.framework.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)

# Register so the supervisor can dynamically discover this agent
AgentRegistry.register_agent(
    name="metrics_agent",
    description="Computes ad performance metrics from raw data (no LLM needed).",
)


def run_metrics_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all metrics and return a validated ``MetricsBundle``."""
    logger.info("[metrics_agent] Starting metrics computation...")

    dataset_names = list(DATASET_MAPPING.keys())
    raw_report = get_holistic_performance_report_data(dataset_names)

    try:
        metrics_bundle = MetricsBundle(**raw_report)
    except Exception as exc:
        logger.error("[metrics_agent] Pydantic validation failed: %s", exc)
        metrics_bundle = _attempt_repair(raw_report)

    logger.info("[metrics_agent] Metrics computed & validated successfully.")
    return {"metrics_bundle": metrics_bundle}


def _attempt_repair(raw: Dict[str, Any]) -> MetricsBundle:
    """Best-effort repair of common serialization issues (enums, datetimes)."""
    from datetime import datetime

    def _fix(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "value"):
            return obj.value
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fix(i) for i in obj]
        return obj

    cleaned = _fix(raw)
    return MetricsBundle(**cleaned)
