from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from src.graph import build_workflow
from src.agents.supervisor import initialize_state, SupervisorState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Amazon Ads multi-agent analytics workflow."
    )
    parser.add_argument(
        "--request",
        type=str,
        default="Generate an Amazon Ads performance report.",
        help="High-level description of the analysis request.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date for the report (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date for the report (YYYY-MM-DD).",
    )
    return parser.parse_args()


def pretty_print_results(final_state: SupervisorState) -> None:
    metrics_bundle = final_state.get("metrics_bundle")
    insights_report = final_state.get("insights_report")

    print("\n=== Amazon Ads Metrics Bundle (structured) ===")
    if metrics_bundle is None:
        print("No metrics bundle produced.")
    else:
        if hasattr(metrics_bundle, "model_dump"):
            metrics_payload: Any = metrics_bundle.model_dump(mode="json")
        else:
            metrics_payload = metrics_bundle
            
        # Store Account Summary to file
        with open("metrics_output.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)
        
        print("Full metrics bundle produced successfully.")

    print("\n=== Amazon Ads Insights Report (structured) ===")
    if insights_report is None:
        print("No insights report produced.")
    else:
        if hasattr(insights_report, "model_dump"):
            insights_payload: Any = insights_report.model_dump(mode="json")
        else:
            insights_payload = insights_report
        print("Insights report produced successfully.")

        summary = getattr(insights_report, "natural_language_summary", None)
        if summary:
            print("\n=== Executive Summary ===")
            print(summary)


def main() -> None:
    args = parse_args()

    logger.info("Building LangGraph workflow...")
    app = build_workflow()

    initial_state: SupervisorState = initialize_state(
        user_request=args.request,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    logger.info(f"Invoking workflow with request: {args.request!r}")

    final_state: SupervisorState = app.invoke(initial_state)
    logger.info("Workflow completed. Rendering results.")
    pretty_print_results(final_state)


if __name__ == "__main__":
    main()

