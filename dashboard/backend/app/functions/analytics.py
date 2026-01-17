import pandas as pd
from typing import List, Dict, Any, Union

def get_neighborhood_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculates basic stats for a numeric column."""
    if column not in df.columns:
        return {"error": f"Column {column} not found."}
    return {
        "mean": float(df[column].mean()),
        "median": float(df[column].median()),
        "max": float(df[column].max()),
        "min": float(df[column].min())
    }

def get_top_k_neighborhoods(df: pd.DataFrame, sort_by: str, k: int = 5, ascending: bool = False) -> List[Dict[str, Any]]:
    """Returns top K neighborhoods sorted by a column."""
    if sort_by not in df.columns:
        return [{"error": f"Column {sort_by} not found."}]
    
    # Ensure name is included if available
    cols = list(df.columns)
    
    sorted_df = df.sort_values(by=sort_by, ascending=ascending).head(k)
    return sorted_df.to_dict(orient="records")

def count_filtered_neighborhoods(df: pd.DataFrame, column: str, threshold: float, operator: str = ">") -> int:
    """Counts rows matching a condition."""
    if column not in df.columns:
        return -1
        
    if operator == ">":
        return len(df[df[column] > threshold])
    elif operator == "<":
        return len(df[df[column] < threshold])
    elif operator == ">=":
        return len(df[df[column] >= threshold])
    elif operator == "<=":
        return len(df[df[column] <= threshold])
    elif operator == "==":
        return len(df[df[column] == threshold])
    else:
        return 0
