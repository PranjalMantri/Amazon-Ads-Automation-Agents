from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from langchain.tools import tool

from src.schemas.metrics_schema import (
    AccountSummary,
    CampaignMetrics,
    ProductMetrics,
    SearchTermMetrics,
)
from src.tools.data_loader_tools import _load_dataframe


def _to_dataframe(records: Optional[Iterable[Dict[str, Any]]]) -> pd.DataFrame:
    """Convert a list of dictionaries to a pandas DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(list(records))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to simplify downstream access."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first column from `candidates` that exists in `df`, trying a few
    normalizations to be robust to common naming differences.
    """
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def _extract_numeric_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    """
    Identify core numeric metric columns by trying common name variants.

    Returns a 5-tuple of column names:
        (spend_col, sales_col, orders_col, impressions_col, clicks_col)

    Raises ValueError if any required column cannot be located.
    """
    spend_col = _first_existing_column(df, ["Spend", "spend"])
    sales_col = _first_existing_column(
        df,
        [
            "Sales",
            "sales",
            "Revenue",
            "revenue",
            "14 Day Total Sales (₹)",
            "14 Day Total Sales – (Click)",
        ],
    )
    orders_col = _first_existing_column(
        df,
        [
            "Orders",
            "orders",
            "Purchases",
            "14 Day Total Orders (#)",
            "14 Day Total Orders (#) – (Click)",
        ],
    )
    impressions_col = _first_existing_column(df, ["Impressions", "impressions"])
    clicks_col = _first_existing_column(df, ["Clicks", "clicks"])

    missing = [
        name
        for name, col in [
            ("spend", spend_col),
            ("sales", sales_col),
            ("orders", orders_col),
            ("impressions", impressions_col),
            ("clicks", clicks_col),
        ]
        if col is None
    ]
    if missing:
        # In robust systems, we might warn instead of crashing, but failure ensures we don't return garbage.
        raise ValueError(
            f"Required numeric columns not found: {', '.join(missing)}."
        )

    return spend_col, sales_col, orders_col, impressions_col, clicks_col


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator not in (0, 0.0) else 0.0


def _aggregate_base_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate core metrics for a grouped DataFrame and compute derived ratios.
    """
    (
        spend_col,
        sales_col,
        orders_col,
        impressions_col,
        clicks_col,
    ) = _extract_numeric_columns(group_df)

    spend = float(group_df[spend_col].fillna(0).sum())
    sales = float(group_df[sales_col].fillna(0).sum())
    orders = int(group_df[orders_col].fillna(0).sum())
    impressions = int(group_df[impressions_col].fillna(0).sum())
    clicks = int(group_df[clicks_col].fillna(0).sum())

    ctr = _safe_div(clicks, impressions)
    cvr = _safe_div(orders, clicks)
    cpc = _safe_div(spend, clicks)
    acos = _safe_div(spend, sales)
    roas = _safe_div(sales, spend)

    return {
        "spend": spend,
        "sales": sales,
        "orders": orders,
        "impressions": impressions,
        "clicks": clicks,
        "ctr": ctr,
        "cvr": cvr,
        "cpc": cpc,
        "acos": acos,
        "roas": roas,
    }


@tool("compute_campaign_metrics", return_direct=False)
def compute_campaign_metrics(
    dataset_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Deterministically compute campaign-level performance metrics.

    Args:
        dataset_names: List of dataset names to include (e.g. ['sponsored_display', 'sponsored_brands'])
    """
    frames: List[pd.DataFrame] = []
    
    for name in dataset_names:
        try:
            df = _load_dataframe(name)
            df = _normalize_columns(df)
            # Tag the source type if possible
            if "sponsored_display" in name:
                df["__campaign_type"] = "SD"
            elif "sponsored_brands" in name:
                df["__campaign_type"] = "SB"
            else:
                df["__campaign_type"] = "Other"
            frames.append(df)
        except Exception:
            # Skip invalid datasets or handle error
            continue

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

    campaign_name_col = _first_existing_column(
        df, ["Campaign Name", "campaign_name", "Campaign"]
    )
    campaign_id_col = _first_existing_column(df, ["Campaign ID", "campaign_id"])
    
    # If we can't identify campaigns, we can't group by them.
    if not campaign_name_col and not campaign_id_col:
        return []

    group_keys = []
    if campaign_id_col:
        group_keys.append(campaign_id_col)
    if campaign_name_col:
        group_keys.append(campaign_name_col)
    # group_keys.append("__campaign_type") # Optional: split by type or aggregate same campaign across types? 
    # Usually same campaign ID implies same campaign. Let's keep type for clarity if unique.
    if "__campaign_type" in df.columns:
        group_keys.append("__campaign_type")

    grouped = df.groupby(group_keys, dropna=False)

    results: List[Dict[str, Any]] = []
    for key_vals, group_df in grouped:
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(group_keys, key_vals))
        
        try:
            base = _aggregate_base_metrics(group_df)
        except ValueError:
            continue

        metrics = CampaignMetrics(
            campaign_id=str(key_map.get(campaign_id_col)) if campaign_id_col else None,
            campaign_name=str(key_map.get(campaign_name_col))
            if campaign_name_col
            else None,
            campaign_type=key_map.get("__campaign_type"),
            **base,
        )
        results.append(metrics.dict())

    return results


@tool("compute_search_term_metrics", return_direct=False)
def compute_search_term_metrics(
    dataset_name: str = "sponsored_brands",
) -> List[Dict[str, Any]]:
    """
    Deterministically compute search-term-level performance metrics.
    Defaults to 'sponsored_brands' if not specified.
    """
    try:
        df = _load_dataframe(dataset_name)
    except Exception:
        return []

    df = _normalize_columns(df)

    search_term_col = _first_existing_column(
        df, ["Search Term", "search_term", "Customer Search Term"]
    )
    if not search_term_col:
        return []

    campaign_name_col = _first_existing_column(
        df, ["Campaign Name", "campaign_name", "Campaign"]
    )
    campaign_id_col = _first_existing_column(df, ["Campaign ID", "campaign_id"])
    ad_group_name_col = _first_existing_column(df, ["Ad Group Name", "ad_group_name"])
    match_type_col = _first_existing_column(df, ["Match Type", "match_type"])

    group_keys = [search_term_col]
    if campaign_id_col:
        group_keys.append(campaign_id_col)
    if campaign_name_col:
        group_keys.append(campaign_name_col)
    if ad_group_name_col:
        group_keys.append(ad_group_name_col)
    if match_type_col:
        group_keys.append(match_type_col)

    grouped = df.groupby(group_keys, dropna=False)

    results: List[Dict[str, Any]] = []
    for key_vals, group_df in grouped:
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(group_keys, key_vals))
        
        try:
            base = _aggregate_base_metrics(group_df)
        except ValueError:
            continue

        metrics = SearchTermMetrics(
            search_term=str(key_map.get(search_term_col)),
            match_type=str(key_map.get(match_type_col))
            if match_type_col
            else None,
            campaign_id=str(key_map.get(campaign_id_col)) if campaign_id_col else None,
            campaign_name=str(key_map.get(campaign_name_col))
            if campaign_name_col
            else None,
            ad_group_name=str(key_map.get(ad_group_name_col))
            if ad_group_name_col
            else None,
            **base,
        )
        results.append(metrics.dict())

    return results


@tool("compute_product_metrics", return_direct=False)
def compute_product_metrics(
    dataset_name: str = "sponsored_display",
) -> List[Dict[str, Any]]:
    """
    Deterministically compute product-level performance metrics.
    Defaults to 'sponsored_display'.
    """
    try:
        df = _load_dataframe(dataset_name)
    except Exception:
        return []

    df = _normalize_columns(df)

    asin_col = _first_existing_column(df, ["ASIN", "asin"])
    sku_col = _first_existing_column(df, ["SKU", "sku"])
    product_name_col = _first_existing_column(
        df, ["Advertised ASIN", "Product Name", "product_name"]
    )
    brand_col = _first_existing_column(df, ["Brand", "brand"])
    category_col = _first_existing_column(df, ["Category", "category"])
    campaign_name_col = _first_existing_column(
        df, ["Campaign Name", "campaign_name", "Campaign"]
    )
    campaign_id_col = _first_existing_column(df, ["Campaign ID", "campaign_id"])

    group_keys = []
    for col in [asin_col, sku_col, product_name_col, campaign_id_col, campaign_name_col]:
        if col:
            group_keys.append(col)

    if not group_keys:
        # Fallback: aggregate across entire dataset as a single pseudo-product.
        # But allow for "all" 
        df["__all"] = 1
        grouped = df.groupby("__all", dropna=False)
    else:
        grouped = df.groupby(group_keys, dropna=False)

    results: List[Dict[str, Any]] = []
    for key_vals, group_df in grouped:
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(group_keys or ["__all"], key_vals))
        
        try:
            base = _aggregate_base_metrics(group_df)
        except ValueError:
            continue

        metrics = ProductMetrics(
            asin=str(key_map.get(asin_col)) if asin_col else None,
            sku=str(key_map.get(sku_col)) if sku_col else None,
            product_name=str(key_map.get(product_name_col))
            if product_name_col
            else None,
            brand=str(key_map.get(brand_col)) if brand_col else None,
            category=str(key_map.get(category_col)) if category_col else None,
            campaign_id=str(key_map.get(campaign_id_col)) if campaign_id_col else None,
            campaign_name=str(key_map.get(campaign_name_col))
            if campaign_name_col
            else None,
            **base,
        )
        results.append(metrics.dict())

    return results

# NOTE: Account summary still expects the calculated metrics to aggregate them.
# The agent will likely compute C/S/P metrics and then pass them to this tool.
@tool("compute_account_summary", return_direct=False)
def compute_account_summary(
    campaign_metrics: Optional[List[Dict[str, Any]]] = None,
    search_term_metrics: Optional[List[Dict[str, Any]]] = None,
    product_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Deterministically compute an account-level summary.

    This function aggregates core performance metrics from campaign-level
    results (or other levels if campaign is missing) and counts entities.
    """
    campaign_metrics = campaign_metrics or []
    search_term_metrics = search_term_metrics or []
    product_metrics = product_metrics or []

    # Aggregate core performance from campaigns only to avoid double counting.
    # If no campaign metrics, try products, then search terms.
    source_metrics = campaign_metrics
    if not source_metrics and product_metrics:
        source_metrics = product_metrics
    if not source_metrics and search_term_metrics:
        source_metrics = search_term_metrics

    spend = sum(float(m.get("spend", 0.0)) for m in source_metrics)
    sales = sum(float(m.get("sales", 0.0)) for m in source_metrics)
    orders = int(sum(float(m.get("orders", 0)) for m in source_metrics))
    impressions = int(sum(float(m.get("impressions", 0)) for m in source_metrics))
    clicks = int(sum(float(m.get("clicks", 0)) for m in source_metrics))

    ctr = _safe_div(clicks, impressions)
    cvr = _safe_div(orders, clicks)
    cpc = _safe_div(spend, clicks)
    acos = _safe_div(spend, sales)
    roas = _safe_div(sales, spend)

    total_campaigns = len({m.get('campaign_id') for m in campaign_metrics if m.get('campaign_id')})
    # If campaign_metrics list is empty, count might be 0, but total_campaigns might also be derived from products/search terms
    # but strictly speaking we count what we have.
    
    summary = AccountSummary(
        spend=spend,
        sales=sales,
        orders=orders,
        impressions=impressions,
        clicks=clicks,
        ctr=ctr,
        cvr=cvr,
        cpc=cpc,
        acos=acos,
        roas=roas,
        total_campaigns=total_campaigns if total_campaigns > 0 else len(campaign_metrics),
        total_products=len(product_metrics),
        total_search_terms=len(search_term_metrics),
        start_date=None,
        end_date=None,
    )

    return summary.dict()
