from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from langchain.tools import tool

# Global cache to avoid reloading large Excel files
_DATA_CACHE: Dict[str, pd.DataFrame] = {}

DATASET_MAPPING = {
    "sponsored_display": "SD_AdvertisedProduct.xlsx",
    "sponsored_brands": "SB_SearchTerm_Daily.xlsx",
}

def _get_data_dir() -> Path:
    """Get absolute path to data directory."""
    tools_dir = Path(__file__).resolve().parent
    project_root = tools_dir.parent.parent
    return project_root / "data"

def _load_dataframe(dataset_name: str) -> pd.DataFrame:
    """
    Internal helper to load dataframe by logical name.
    
    This function handles caching and file paths. It returns a pandas DataFrame, NOT raw data.
    """
    try:
        normalized = dataset_name.lower().replace(" ", "_").strip()
    except AttributeError:
        # Handle case where dataset_name might be None or not a string
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    if normalized not in DATASET_MAPPING:
        # Try finding partial match if needed or just error
        available = list(DATASET_MAPPING.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. available: {available}")
    
    key = normalized
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    
    filename = DATASET_MAPPING[key]
    data_dir = _get_data_dir()
    file_path = data_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Failed to load excel file {filename}: {e}")
        
    df.columns = [str(col).strip() for col in df.columns]
    _DATA_CACHE[key] = df
    return df

@tool("list_available_datasets")
def list_available_datasets() -> List[str]:
    """List the names of datasets available for analysis."""
    return list(DATASET_MAPPING.keys())

@tool("get_dataset_schema")
def get_dataset_schema(dataset_name: str) -> Dict[str, Any]:
    """
    Get the schema (column names and types) for a specified dataset.
    
    Args:
        dataset_name: The name of the dataset to inspect (e.g., 'sponsored_display').
    """
    try:
        df = _load_dataframe(dataset_name)
        return {
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "row_count": len(df)
        }
    except Exception as e:
        return {"error": str(e)}

@tool("get_dataset_sample")
def get_dataset_sample(dataset_name: str, n: int = 3) -> List[Dict[str, Any]]:
    """
    Get the first N rows of a dataset to understand the data format and values.
    
    Args:
        dataset_name: The name of the dataset.
        n: Number of rows to return (default 3).
    """
    try:
        df = _load_dataframe(dataset_name)
        # Return only first n rows
        return df.head(n).to_dict(orient="records")
    except Exception as e:
        return [{"error": str(e)}]
