from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime

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
    """Return the first column from *candidates* that exists in *df* (case-insensitive)."""
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


def _compute_campaign_metrics(
    dataset_names: List[str],
    sort_by: str = "spend",
    ascending: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Internal logic for computing campaign-level performance metrics.
    """
    frames: List[pd.DataFrame] = []
    
    for name in dataset_names:
        try:
            df = _load_dataframe(name)
            df = _normalize_columns(df)
            if "sponsored_display" in name:
                df["__campaign_type"] = "SD"
            elif "sponsored_brands" in name:
                df["__campaign_type"] = "SB"
            else:
                df["__campaign_type"] = "Other"
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

    campaign_name_col = _first_existing_column(
        df, ["Campaign Name", "campaign_name", "Campaign"]
    )
    campaign_id_col = _first_existing_column(df, ["Campaign ID", "campaign_id"])

    if not campaign_name_col and not campaign_id_col:
        return []

    group_keys = []
    if campaign_id_col:
        group_keys.append(campaign_id_col)
    if campaign_name_col:
        group_keys.append(campaign_name_col)
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
        results.append(metrics.model_dump())

    sorted_results = sorted(
        results,
        key=lambda x: float(x.get(sort_by, 0.0) or 0.0),
        reverse=not ascending
    )
    
    return sorted_results[:limit]


@tool("compute_campaign_metrics", return_direct=False)
def compute_campaign_metrics(
    dataset_names: List[str],
    sort_by: str = "spend",
    ascending: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Deterministically compute campaign-level performance metrics.

    Args:
        dataset_names: List of dataset names to include (e.g. ['sponsored_display', 'sponsored_brands'])
        sort_by: Metric to sort by (e.g. 'spend', 'roas', 'sales'). default='spend'
        ascending: Sort order. default=False (descending)
        limit: Max number of records to return. default=50
    """
    return _compute_campaign_metrics(dataset_names, sort_by, ascending, limit)


def _compute_search_term_metrics(
    dataset_names: List[str],
    sort_by: str = "spend",
    ascending: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Compute search-term-level metrics across the given datasets."""
    frames = []
    for dataset_name in dataset_names:
        try:
            df = _load_dataframe(dataset_name)
            df = _normalize_columns(df)
            frames.append(df)
        except Exception:
            continue
            
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

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
        results.append(metrics.model_dump())

    sorted_results = sorted(
        results,
        key=lambda x: float(x.get(sort_by, 0.0) or 0.0),
        reverse=not ascending
    )

    return sorted_results[:limit]


@tool("compute_search_term_metrics", return_direct=False)
def compute_search_term_metrics(
    dataset_name: str = "sponsored_brands",
    sort_by: str = "spend",
    ascending: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Deterministically compute search-term-level performance metrics.
    Defaults to 'sponsored_brands' if not specified.
    
    Args:
        dataset_name: The dataset to analyze.
        sort_by: Metric to sort by (e.g. 'spend', 'clicks').
        limit: Max number of records to return.
    """
    return _compute_search_term_metrics([dataset_name], sort_by, ascending, limit)


def _compute_product_metrics(
    dataset_names: List[str],
    sort_by: str = "spend",
    ascending: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Compute product-level metrics across the given datasets."""
    frames = []
    for dataset_name in dataset_names:
        try:
            df = _load_dataframe(dataset_name)
            df = _normalize_columns(df)
            frames.append(df)
        except Exception:
            continue
            
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

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
    if asin_col: group_keys.append(asin_col)
    if sku_col: group_keys.append(sku_col)
    if product_name_col: group_keys.append(product_name_col)
    if campaign_id_col: group_keys.append(campaign_id_col)
    if campaign_name_col: group_keys.append(campaign_name_col)

    if not group_keys:
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
        results.append(metrics.model_dump())

    sorted_results = sorted(
        results,
        key=lambda x: float(x.get(sort_by, 0.0) or 0.0),
        reverse=not ascending
    )

    return sorted_results[:limit]


@tool("compute_product_metrics", return_direct=False)
def compute_product_metrics(
    dataset_name: str = "sponsored_display",
    sort_by: str = "spend",
    ascending: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Deterministically compute product-level performance metrics.
    Defaults to 'sponsored_display'.
    
    Args:
        dataset_name: The dataset to analyze.
        sort_by: Metric to sort by (e.g. 'spend', 'roas').
        limit: Max number of records to return.
    """
    return _compute_product_metrics([dataset_name], sort_by, ascending, limit)


def _compute_account_summary(
    dataset_names: List[str],
) -> Dict[str, Any]:
    """Compute an account-level summary across all datasets."""
    frames: List[pd.DataFrame] = []

    for name in dataset_names:
        try:
            df = _load_dataframe(name)
            df = _normalize_columns(df)
            frames.append(df)
        except Exception:
            continue
            
    if not frames:
        return AccountSummary().model_dump()
    
    full_df = pd.concat(frames, sort=False, ignore_index=True)

    try:
        base = _aggregate_base_metrics(full_df)
    except ValueError:
        return AccountSummary().model_dump()

    camp_col = _first_existing_column(full_df, ["Campaign ID", "campaign_id", "Campaign Name", "campaign_name"])
    total_campaigns = full_df[camp_col].nunique() if camp_col else 0

    asin_col = _first_existing_column(full_df, ["ASIN", "asin", "Advertised ASIN"])
    total_products = full_df[asin_col].nunique() if asin_col else 0

    term_col = _first_existing_column(full_df, ["Search Term", "search_term", "Customer Search Term"])
    total_search_terms = full_df[term_col].nunique() if term_col else 0

    summary = AccountSummary(
        total_campaigns=total_campaigns,
        total_products=total_products,
        total_search_terms=total_search_terms,
        **base
    )

    return summary.model_dump()


@tool("compute_account_summary", return_direct=False)
def compute_account_summary(
    dataset_names: List[str],
) -> Dict[str, Any]:
    """
    Deterministically compute an account-level summary from raw data files.
    
    Args:
        dataset_names: List of dataset names to include in the summary.
    """
    return _compute_account_summary(dataset_names)


@tool("get_holistic_performance_report", return_direct=False)
def get_holistic_performance_report(dataset_names: List[str]) -> Dict[str, Any]:
    """Generate a comprehensive performance report.

    Args:
        dataset_names: Dataset names to include in the computation.
    """
    return get_holistic_performance_report_data(dataset_names)


def get_holistic_performance_report_data(dataset_names: List[str]) -> Dict[str, Any]:
    """Compute the full report and return JSON-serializable ``MetricsBundle`` data."""
    
    account_summary = _compute_account_summary(dataset_names)

    top_campaigns_spend = _compute_campaign_metrics(
        dataset_names, sort_by="spend", ascending=False, limit=5
    )
    top_campaigns_roas = _compute_campaign_metrics(
        dataset_names, sort_by="roas", ascending=False, limit=5
    )
    bottom_campaigns_roas = _compute_campaign_metrics(
        dataset_names, sort_by="roas", ascending=True, limit=5
    )

    top_search_terms_spend = _compute_search_term_metrics(
        dataset_names, sort_by="spend", ascending=False, limit=5
    )
    top_search_terms_roas = _compute_search_term_metrics(
        dataset_names, sort_by="roas", ascending=False, limit=5
    )

    top_products_spend = _compute_product_metrics(
        dataset_names, sort_by="spend", ascending=False, limit=5
    )
    bottom_products_roas = _compute_product_metrics(
        dataset_names, sort_by="roas", ascending=True, limit=5
    )

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "value"):  # Enum
            return obj.value
        return obj

    return _serialize({
        "report_metadata": {
            "generated_at": datetime.utcnow(),
            "start_date": account_summary.get("start_date"),
            "end_date": account_summary.get("end_date")
        },
        "account_summary": account_summary,
        "top_campaigns_by_spend": top_campaigns_spend,
        "top_campaigns_by_roas": top_campaigns_roas,
        "bottom_campaigns_by_roas": bottom_campaigns_roas,
        "top_search_terms_by_spend": top_search_terms_spend,
        "top_search_terms_by_roas": top_search_terms_roas,
        "top_products_by_spend": top_products_spend,
        "bottom_products_by_roas": bottom_products_roas
    })
