"""
Data processing functions for the soccer analysis application.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

from config.settings import COLUMN_MAPPINGS, PLAYER_THRESHOLDS

logger = logging.getLogger(__name__)

def process_player_stats(
    df: pd.DataFrame,
    positions: Optional[List[str]] = None,
    min_90s: Optional[float] = None,
    max_age: Optional[int] = None
) -> pd.DataFrame:
    """
    Process player statistics DataFrame with filtering and column renaming.

    Args:
        df: Input DataFrame from data source
        positions: List of positions to filter for
        min_90s: Minimum number of 90-minute periods played
        max_age: Maximum player age to include

    Returns:
        Processed DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to process_player_stats")
        return df

    # Create a copy to avoid modifying original
    processed_df = df.copy()

    # Filter by positions if provided
    if positions and "Pos" in processed_df.columns:
        position_filter = processed_df["Pos"].apply(
            lambda x: any(pos in x for pos in positions)
        )
        processed_df = processed_df[position_filter]

    # Handle age processing
    if "Age" in processed_df.columns:
        # Convert age column if it contains dashes (e.g., "24-104" format)
        if processed_df["Age"].dtype == 'object' and processed_df["Age"].str.contains('-').any():
            processed_df["Age"] = processed_df["Age"].str.split("-").str[0].astype(int)

        # Filter by max age if provided
        if max_age is not None:
            processed_df = processed_df[processed_df["Age"] <= max_age]

    # Filter by minimum 90s played
    if min_90s is not None and "90s" in processed_df.columns:
        processed_df = processed_df[processed_df["90s"] >= min_90s]

    return processed_df


def process_passing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process passing statistics DataFrame.

    Args:
        df: Input DataFrame with passing statistics

    Returns:
        Processed DataFrame with standardized column names
    """
    if df.empty:
        return df

    processed_df = df.copy()

    # Apply column renames from configuration
    if "passing" in COLUMN_MAPPINGS and "rename" in COLUMN_MAPPINGS["passing"]:
        columns = processed_df.columns.tolist()
        for idx, new_name in COLUMN_MAPPINGS["passing"]["rename"].items():
            if idx < len(columns):
                columns[idx] = new_name
        processed_df.columns = columns

    return processed_df.sort_values("total_Cmp%", ascending=False)


def process_shooting_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process shooting statistics DataFrame.

    Args:
        df: Input DataFrame with shooting statistics

    Returns:
        Processed DataFrame with relevant filters applied
    """
    if df.empty:
        return df

    # Apply default threshold filters
    threshold = PLAYER_THRESHOLDS.get("shooting", {}).get("Gls", 5)
    shooting_df_filtered = df[df["Gls"] >= threshold]

    return shooting_df_filtered.sort_values("npxG", ascending=False)


def process_defensive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process defensive statistics DataFrame.

    Args:
        df: Input DataFrame with defensive statistics

    Returns:
        Processed DataFrame with standardized column names and filters
    """
    if df.empty:
        return df

    processed_df = df.copy()

    # Handle duplicate column names
    cols = processed_df.columns.tolist()

    # Find duplicate 'Tkl' column and rename it
    col_to_change = "Tkl"
    if cols.count(col_to_change) > 1:
        second_index = cols.index(col_to_change, cols.index(col_to_change) + 1)
        cols[second_index] = col_to_change + "_challenge"
        processed_df.columns = cols

    # Apply threshold filters
    threshold = PLAYER_THRESHOLDS.get("defense", {}).get("Tkl%", 50.0)
    if "Tkl%" in processed_df.columns:
        processed_df = processed_df[processed_df["Tkl%"] >= threshold]

    return processed_df.sort_values(by="Tkl%", ascending=False)


def calculate_per_90_metrics(
    df: pd.DataFrame,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Calculate per-90 metrics for the given DataFrame.

    Args:
        df: Input DataFrame
        metrics: List of column names to normalize to per-90

    Returns:
        DataFrame with added per-90 columns
    """
    if df.empty or "90s" not in df.columns:
        return df

    result_df = df.copy()

    for metric in metrics:
        if metric in result_df.columns:
            result_df[f"{metric}_90"] = result_df[metric] / result_df["90s"]

    return result_df
