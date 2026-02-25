from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain.tools import tool


def _get_data_dir() -> Path:
    """Get absolute path to data directory."""
    tools_dir = Path(__file__).resolve().parent
    project_root = tools_dir.parent.parent
    return project_root / "data"


def _load_excel(filename: str) -> List[Dict[str, Any]]:
    """Load Excel file as list of dicts."""
    data_dir = _get_data_dir()
    file_path = data_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected Excel file not found at: {file_path}. "
            "Ensure the `data/` directory exists at the project root and contains "
            f"`{filename}`."
        )

    df = pd.read_excel(file_path, engine="openpyxl")

    # Normalize column names slightly to reduce downstream brittleness.
    df.columns = [str(col).strip() for col in df.columns]

    return df.to_dict(orient="records")


@tool("load_sd_product_data", return_direct=False)
def load_sd_product_data() -> List[Dict[str, Any]]:
    """Load Sponsored Display product data."""
    return _load_excel("SD_AdvertisedProduct.xlsx")


@tool("load_sb_search_term_data", return_direct=False)
def load_sb_search_term_data() -> List[Dict[str, Any]]:
    """Load Sponsored Brands search term data."""
    return _load_excel("SB_SearchTerm_Daily.xlsx")