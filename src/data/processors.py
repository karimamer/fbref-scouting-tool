from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
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


def process_shooting_stats(df: pd.DataFrame, min_shots: Optional[int] = None) -> pd.DataFrame:
    """
    Process shooting statistics DataFrame with enhanced metrics.

    Args:
        df: Input DataFrame with shooting statistics
        min_shots: Minimum number of shots filter (overrides config if provided)

    Returns:
        Processed DataFrame with additional metrics
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to process_shooting_stats")
        return df

    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Apply default threshold from config if min_shots not provided
    threshold = min_shots if min_shots is not None else PLAYER_THRESHOLDS.get("shooting", {}).get("Gls", 5)

    # Check if we should filter by Goals or Shots
    if "Sh" in processed_df.columns and min_shots is not None:
        processed_df = processed_df[processed_df["Sh"] >= min_shots]
    elif "Gls" in processed_df.columns:
        processed_df = processed_df[processed_df["Gls"] >= threshold]

    # Standardize column names if needed
    column_mappings = {
        # Example mappings if needed
        # "G-xG": "goals_above_xG",
        # "np:G-xG": "np_goals_above_xG"
    }

    # Apply column renames
    processed_df = processed_df.rename(columns=column_mappings)

    # Handle age formatting if needed (e.g., "25-204" format)
    if "Age" in processed_df.columns and processed_df["Age"].dtype == "object":
        try:
            # Extract main age number before the dash
            processed_df["Age"] = processed_df["Age"].str.split("-").str[0].astype(int)
        except Exception as e:
            logger.warning(f"Could not process Age column: {str(e)}")

    # Calculate basic shooting metrics if not present
    # Shots per 90
    if "Sh/90" not in processed_df.columns and "Sh" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["Sh/90"] = processed_df["Sh"] / processed_df["90s"]

    # Shots on target per 90
    if "SoT/90" not in processed_df.columns and "SoT" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["SoT/90"] = processed_df["SoT"] / processed_df["90s"]

    # Shots on target percentage
    if "SoT%" not in processed_df.columns and "SoT" in processed_df.columns and "Sh" in processed_df.columns:
        processed_df["SoT%"] = (processed_df["SoT"] / processed_df["Sh"]) * 100

    # Goals per shot
    if "G/Sh" not in processed_df.columns and "Gls" in processed_df.columns and "Sh" in processed_df.columns:
        processed_df["G/Sh"] = processed_df["Gls"] / processed_df["Sh"]

    # Goals per shot on target
    if "G/SoT" not in processed_df.columns and "Gls" in processed_df.columns and "SoT" in processed_df.columns:
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            processed_df["G/SoT"] = processed_df["Gls"] / processed_df["SoT"]
            processed_df["G/SoT"] = processed_df["G/SoT"].replace([np.inf, -np.inf, np.nan], 0)

    # Goals per 90
    if "Gls/90" not in processed_df.columns and "Gls" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["Gls/90"] = processed_df["Gls"] / processed_df["90s"]

    # Non-penalty goals
    if "npG" not in processed_df.columns and "Gls" in processed_df.columns and "PK" in processed_df.columns:
        processed_df["npG"] = processed_df["Gls"] - processed_df["PK"]

    # Non-penalty goals per 90
    if "npG/90" not in processed_df.columns and "npG" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["npG/90"] = processed_df["npG"] / processed_df["90s"]

    # Expected goals per 90
    if "xG/90" not in processed_df.columns and "xG" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["xG/90"] = processed_df["xG"] / processed_df["90s"]

    # Non-penalty expected goals per 90
    if "npxG/90" not in processed_df.columns and "npxG" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["npxG/90"] = processed_df["npxG"] / processed_df["90s"]

    # Expected goals per shot
    if "xG/Sh" not in processed_df.columns and "xG" in processed_df.columns and "Sh" in processed_df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            processed_df["xG/Sh"] = processed_df["xG"] / processed_df["Sh"]
            processed_df["xG/Sh"] = processed_df["xG/Sh"].replace([np.inf, -np.inf, np.nan], 0)

    # Goals - xG (finishing skill)
    if "G-xG" not in processed_df.columns and "Gls" in processed_df.columns and "xG" in processed_df.columns:
        processed_df["G-xG"] = processed_df["Gls"] - processed_df["xG"]

    # Non-penalty goals - npxG (non-penalty finishing skill)
    if "npG-npxG" not in processed_df.columns and "npG" in processed_df.columns and "npxG" in processed_df.columns:
        processed_df["npG-npxG"] = processed_df["npG"] - processed_df["npxG"]

    # Sort by relevant metric (goals by default)
    if "Gls" in processed_df.columns:
        processed_df = processed_df.sort_values("Gls", ascending=False)
    elif "Sh" in processed_df.columns:
        processed_df = processed_df.sort_values("Sh", ascending=False)

    return processed_df

def process_combined_shooting_data(
    shooting_df: pd.DataFrame,
    shot_creation_df: Optional[pd.DataFrame] = None,
    possession_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Combine and process shooting statistics with other related metrics.

    Args:
        shooting_df: DataFrame with shooting statistics
        shot_creation_df: Optional DataFrame with shot creation statistics
        possession_df: Optional DataFrame with possession statistics

    Returns:
        Combined and processed DataFrame
    """
    if shooting_df.empty:
        logger.warning("Empty shooting DataFrame provided")
        return pd.DataFrame()

    # Process shooting stats first
    processed_df = process_shooting_stats(shooting_df)

    # Add shot creation metrics if available
    if shot_creation_df is not None and not shot_creation_df.empty:
        # Columns to merge from shot creation data
        creation_cols = ["Player", "Squad"]

        for col in ["SCA", "SCA90", "GCA", "GCA90"]:
            if col in shot_creation_df.columns:
                creation_cols.append(col)

        # Merge with shot creation data
        try:
            processed_df = processed_df.merge(
                shot_creation_df[creation_cols],
                on=["Player", "Squad"],
                how="left"
            )
        except Exception as e:
            logger.warning(f"Error merging shot creation data: {str(e)}")

    # Add possession metrics if available
    if possession_df is not None and not possession_df.empty:
        # Columns to merge from possession data
        possession_cols = ["Player", "Squad"]

        for col in ["Touches", "Att Pen", "Carries", "PrgC"]:
            if col in possession_df.columns:
                possession_cols.append(col)

        # Merge with possession data
        try:
            processed_df = processed_df.merge(
                possession_df[possession_cols],
                on=["Player", "Squad"],
                how="left"
            )

            # Calculate additional metrics
            if "Touches" in processed_df.columns and "Sh" in processed_df.columns:
                processed_df["Sh/Touch"] = processed_df["Sh"] / processed_df["Touches"]

            if "Att Pen" in processed_df.columns and "Sh" in processed_df.columns:
                processed_df["Sh/AttPen"] = processed_df["Sh"] / processed_df["Att Pen"]

        except Exception as e:
            logger.warning(f"Error merging possession data: {str(e)}")

    return processed_df


def process_shot_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process shot quality metrics from shooting data.

    Args:
        df: DataFrame with shooting statistics

    Returns:
        Processed DataFrame with shot quality metrics
    """
    if df.empty:
        return df

    processed_df = df.copy()

    # Calculate shot quality metrics
    # xG per shot (measure of shot quality)
    if "xG" in processed_df.columns and "Sh" in processed_df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            processed_df["xG_per_shot"] = processed_df["xG"] / processed_df["Sh"]
            processed_df["xG_per_shot"] = processed_df["xG_per_shot"].replace([np.inf, -np.inf, np.nan], 0)

    # npxG per non-penalty shot
    if "npxG" in processed_df.columns and "Sh" in processed_df.columns and "PKatt" in processed_df.columns:
        non_pk_shots = processed_df["Sh"] - processed_df["PKatt"]
        with np.errstate(divide='ignore', invalid='ignore'):
            processed_df["npxG_per_shot"] = processed_df["npxG"] / non_pk_shots
            processed_df["npxG_per_shot"] = processed_df["npxG_per_shot"].replace([np.inf, -np.inf, np.nan], 0)

    # Shot accuracy (% on target)
    if "SoT" in processed_df.columns and "Sh" in processed_df.columns:
        processed_df["shot_accuracy"] = processed_df["SoT"] / processed_df["Sh"]

    # Shot conversion rate (goals per shot)
    if "Gls" in processed_df.columns and "Sh" in processed_df.columns:
        processed_df["conversion_rate"] = processed_df["Gls"] / processed_df["Sh"]

    # On-target conversion rate (goals per shot on target)
    if "Gls" in processed_df.columns and "SoT" in processed_df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            processed_df["on_target_conversion"] = processed_df["Gls"] / processed_df["SoT"]
            processed_df["on_target_conversion"] = processed_df["on_target_conversion"].replace([np.inf, -np.inf, np.nan], 0)

    # Calculate shot quality score
    # Combine xG per shot, shot accuracy and shot distance
    if all(col in processed_df.columns for col in ["xG_per_shot", "shot_accuracy"]):
        # Higher xG_per_shot and shot_accuracy are better
        processed_df["shot_quality_score"] = processed_df["xG_per_shot"] * 0.7 + processed_df["shot_accuracy"] * 0.3

        # If distance is available, include it (lower distance is better)
        if "Dist" in processed_df.columns:
            # Normalize distance (invert so lower is better)
            dist_max = processed_df["Dist"].max()
            if dist_max > 0:
                processed_df["norm_dist"] = 1 - (processed_df["Dist"] / dist_max)
                # Include distance in shot quality score
                processed_df["shot_quality_score"] = (
                    processed_df["xG_per_shot"] * 0.6 +
                    processed_df["shot_accuracy"] * 0.25 +
                    processed_df["norm_dist"] * 0.15
                )

    return processed_df.sort_values("xG_per_shot", ascending=False) if "xG_per_shot" in processed_df.columns else processed_df


def process_shooting_classification(df: pd.DataFrame, min_shots: int = 20) -> pd.DataFrame:
    """
    Classify players based on their shooting patterns.

    Args:
        df: DataFrame with shooting statistics
        min_shots: Minimum shots required for classification

    Returns:
        DataFrame with shooting classifications
    """
    if df.empty:
        return df

    # Filter by minimum shots
    processed_df = df[df["Sh"] >= min_shots].copy() if "Sh" in df.columns else df.copy()

    if processed_df.empty:
        return processed_df

    # Calculate metrics needed for classification if not present
    if "shots_p90" not in processed_df.columns and "Sh" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["shots_p90"] = processed_df["Sh"] / processed_df["90s"]

    if "goals_p90" not in processed_df.columns and "Gls" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["goals_p90"] = processed_df["Gls"] / processed_df["90s"]

    if "xG_p90" not in processed_df.columns and "xG" in processed_df.columns and "90s" in processed_df.columns:
        processed_df["xG_p90"] = processed_df["xG"] / processed_df["90s"]

    if "conversion_rate" not in processed_df.columns and "Gls" in processed_df.columns and "Sh" in processed_df.columns:
        processed_df["conversion_rate"] = processed_df["Gls"] / processed_df["Sh"]

    if "shot_accuracy" not in processed_df.columns and "SoT" in processed_df.columns and "Sh" in processed_df.columns:
        processed_df["shot_accuracy"] = processed_df["SoT"] / processed_df["Sh"]

    # Calculate percentile ranks for classification
    metrics_for_classification = []

    for metric in ["shots_p90", "goals_p90", "xG_p90", "conversion_rate", "shot_accuracy"]:
        if metric in processed_df.columns:
            processed_df[f"{metric}_pct"] = processed_df[metric].rank(pct=True)
            metrics_for_classification.append(f"{metric}_pct")

    # Add shot distance percentile if available (low distance is better)
    if "Dist" in processed_df.columns:
        processed_df["dist_pct"] = 1 - processed_df["Dist"].rank(pct=True)
        metrics_for_classification.append("dist_pct")

    # Classify based on patterns
    conditions = []
    categories = []

    # Only proceed with classification if we have enough metrics
    if len(metrics_for_classification) >= 3:
        # Volume shooter: high shots_p90, lower conversion
        if "shots_p90_pct" in processed_df.columns and "conversion_rate_pct" in processed_df.columns:
            conditions.append(
                (processed_df["shots_p90_pct"] > 0.8) &
                (processed_df["conversion_rate_pct"] < 0.5)
            )
            categories.append("Volume Shooter")

        # Clinical finisher: high conversion rate, moderate shot volume
        if "conversion_rate_pct" in processed_df.columns and "shots_p90_pct" in processed_df.columns:
            conditions.append(
                (processed_df["conversion_rate_pct"] > 0.8) &
                (processed_df["shots_p90_pct"] > 0.4) &
                (processed_df["shots_p90_pct"] < 0.8)
            )
            categories.append("Clinical Finisher")

        # Penalty box predator: close shot distance, high shot accuracy
        if "dist_pct" in processed_df.columns and "shot_accuracy_pct" in processed_df.columns:
            conditions.append(
                (processed_df["dist_pct"] > 0.8) &
                (processed_df["shot_accuracy_pct"] > 0.6)
            )
            categories.append("Penalty Box Predator")

        # Long-range shooter: long shot distance, high shot volume
        if "dist_pct" in processed_df.columns and "shots_p90_pct" in processed_df.columns:
            conditions.append(
                (processed_df["dist_pct"] < 0.3) &
                (processed_df["shots_p90_pct"] > 0.6)
            )
            categories.append("Long-Range Shooter")

        # Efficient scorer: good xG per shot, high conversion over xG
        if "xG_p90_pct" in processed_df.columns and "goals_p90_pct" in processed_df.columns:
            conditions.append(
                (processed_df["xG_p90_pct"] > 0.6) &
                (processed_df["goals_p90_pct"] > processed_df["xG_p90_pct"])
            )
            categories.append("Efficient Scorer")

        # Low-volume poacher: low shot volume, high conversion rate
        if "shots_p90_pct" in processed_df.columns and "conversion_rate_pct" in processed_df.columns:
            conditions.append(
                (processed_df["shots_p90_pct"] < 0.3) &
                (processed_df["conversion_rate_pct"] > 0.7)
            )
            categories.append("Low-Volume Poacher")

        # Apply classification
        if conditions and categories:
            import numpy as np
            processed_df["shooting_profile"] = np.select(
                conditions,
                categories,
                default="Balanced Shooter"
            )

    return processed_df
