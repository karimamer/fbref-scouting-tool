from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging

from config.settings import ANALYSIS_WEIGHTS
from src.analysis.metrics import (
    normalize_metric,
    calculate_per_90_metrics,
    calculate_weighted_score,
    get_score_from_config
)
logger = logging.getLogger(__name__)

def analyze_shooting_efficiency(
    shooting_df: pd.DataFrame,
    min_shots: int = 20,
    min_90s: float = 5
) -> pd.DataFrame:
    """
    Analyze shooting efficiency based on conversion rates, shot quality and expected goals.

    Args:
        shooting_df: DataFrame containing shooting statistics
        min_shots: Minimum number of shots to be considered
        min_90s: Minimum number of 90-minute periods played

    Returns:
        DataFrame with efficiency metrics and scores
    """
    if shooting_df.empty:
        return pd.DataFrame()

    # Filter by minimums
    filtered_df = shooting_df[(shooting_df["Sh"] >= min_shots) &
                             (shooting_df["90s"] >= min_90s)].copy()

    if filtered_df.empty:
        logger.warning(f"No players with at least {min_shots} shots and {min_90s} 90s played")
        return pd.DataFrame()

    # Calculate advanced metrics
    filtered_df["conversion_rate"] = filtered_df["Gls"] / filtered_df["Sh"]
    filtered_df["on_target_conversion"] = filtered_df["Gls"] / filtered_df["SoT"]
    filtered_df["shot_quality"] = filtered_df["npxG"] / filtered_df["Sh"]
    filtered_df["finishing_skill"] = filtered_df["Gls"] - filtered_df["xG"]
    filtered_df["non_pk_finishing"] = filtered_df["Gls"] - filtered_df["PK"] - filtered_df["npxG"]

    # Per 90 metrics
    filtered_df["goals_p90"] = filtered_df["Gls"] / filtered_df["90s"]
    filtered_df["shots_p90"] = filtered_df["Sh"] / filtered_df["90s"]
    filtered_df["xG_p90"] = filtered_df["xG"] / filtered_df["90s"]
    filtered_df["npxG_p90"] = filtered_df["npxG"] / filtered_df["90s"]

    # Calculate efficiency score
    metrics = {
        "conversion_rate": 0.25,
        "SoT%": 0.20,
        "shot_quality": 0.20,
        "finishing_skill": 0.15,
        "goals_p90": 0.20
    }

    for metric in metrics.keys():
        if metric in filtered_df.columns:
            filtered_df[f"{metric}_norm"] = normalize_metric(filtered_df[metric])

    # Calculate the overall shooting efficiency score
    filtered_df["shooting_efficiency_score"] = sum(
        filtered_df[f"{metric}_norm"] * weight for metric, weight in metrics.items()
        if f"{metric}_norm" in filtered_df.columns
    )

    return filtered_df.sort_values("shooting_efficiency_score", ascending=False)


def analyze_shooting_profile(
    shooting_df: pd.DataFrame,
    min_shots: int = 20
) -> pd.DataFrame:
    """
    Categorize players by shooting profile based on volume, distance, and efficiency.

    Args:
        shooting_df: DataFrame containing shooting statistics
        min_shots: Minimum number of shots to be considered

    Returns:
        DataFrame with shooting profile classifications
    """
    if shooting_df.empty:
        return pd.DataFrame()

    # Filter by minimum shots
    filtered_df = shooting_df[shooting_df["Sh"] >= min_shots].copy()

    if filtered_df.empty:
        logger.warning(f"No players with at least {min_shots} shots")
        return pd.DataFrame()

    # Calculate metrics for profiling
    filtered_df["shots_p90"] = filtered_df["Sh"] / filtered_df["90s"]
    filtered_df["accuracy"] = filtered_df["SoT"] / filtered_df["Sh"]
    filtered_df["conversion"] = filtered_df["Gls"] / filtered_df["Sh"]

    # Normalize metrics for classification
    for col in ["shots_p90", "accuracy", "conversion", "Dist"]:
        if col in filtered_df.columns:
            filtered_df[f"{col}_norm"] = normalize_metric(filtered_df[col])

    # Classify shooting profiles
    conditions = [
        # High volume, low distance, good accuracy
        (filtered_df["shots_p90_norm"] > 0.7) &
        (filtered_df["Dist_norm"] < 0.3) &
        (filtered_df["accuracy_norm"] > 0.6),

        # Low volume, high accuracy poacher
        (filtered_df["shots_p90_norm"] < 0.4) &
        (filtered_df["accuracy_norm"] > 0.7) &
        (filtered_df["Dist_norm"] < 0.3),

        # Long distance specialist
        (filtered_df["Dist_norm"] > 0.7) &
        (filtered_df["shots_p90_norm"] > 0.5),

        # High efficiency, moderate volume
        (filtered_df["conversion_norm"] > 0.7) &
        (filtered_df["shots_p90_norm"] > 0.4) &
        (filtered_df["shots_p90_norm"] < 0.7),

        # Volume shooter
        (filtered_df["shots_p90_norm"] > 0.8)
    ]

    profile_types = [
        "Penalty Box Scorer",
        "Efficient Poacher",
        "Distance Shooter",
        "Clinical Finisher",
        "Volume Shooter"
    ]

    filtered_df["shooting_profile"] = np.select(
        conditions, profile_types, default="Balanced Shooter"
    )

    return filtered_df


def identify_shot_creation_specialists(
    shooting_df: pd.DataFrame,
    shot_creation_df: pd.DataFrame,
    min_90s: float = 5
) -> pd.DataFrame:
    """
    Identify players who excel at both shooting and creating shooting opportunities.

    Args:
        shooting_df: DataFrame containing shooting statistics
        shot_creation_df: DataFrame containing shot creation statistics
        min_90s: Minimum number of 90-minute periods played

    Returns:
        DataFrame with combined shooting and creation metrics
    """
    if shooting_df.empty or shot_creation_df.empty:
        return pd.DataFrame()

    # Filter by minimum playing time
    shooting = shooting_df[shooting_df["90s"] >= min_90s].copy()
    creation = shot_creation_df[shot_creation_df["90s"] >= min_90s].copy()

    if shooting.empty or creation.empty:
        logger.warning(f"Insufficient data after filtering for min_90s={min_90s}")
        return pd.DataFrame()

    # Get key columns from both dataframes
    shooting_cols = ["Player", "Squad", "Pos", "Age", "90s",
                    "Gls", "Sh", "SoT", "SoT%", "G/Sh", "G/SoT", "xG", "npxG"]
    creation_cols = ["Player", "Squad", "SCA", "GCA", "SCA90", "GCA90"]

    # Ensure all columns exist
    shooting_cols = [col for col in shooting_cols if col in shooting.columns]
    creation_cols = [col for col in creation_cols if col in creation.columns]

    # Merge datasets
    merged_df = shooting[shooting_cols].merge(
        creation[creation_cols],
        on=["Player", "Squad"],
        how="inner"
    )

    if merged_df.empty:
        logger.warning("No matching players between shooting and creation datasets")
        return pd.DataFrame()

    # Calculate combined metrics
    merged_df["goals_p90"] = merged_df["Gls"] / merged_df["90s"]
    merged_df["xG_p90"] = merged_df["xG"] / merged_df["90s"]

    # Calculate contribution score - balance of shooting and creation
    if "SCA90" in merged_df.columns and "GCA90" in merged_df.columns:
        # Normalize metrics
        for col in ["goals_p90", "xG_p90", "SCA90", "GCA90"]:
            merged_df[f"{col}_norm"] = normalize_metric(merged_df[col])

        # Calculate score components
        merged_df["shooting_component"] = (
            merged_df["goals_p90_norm"] * 0.6 +
            merged_df["xG_p90_norm"] * 0.4
        )

        merged_df["creation_component"] = (
            merged_df["SCA90_norm"] * 0.6 +
            merged_df["GCA90_norm"] * 0.4
        )

        # Final score
        merged_df["shot_contribution_score"] = (
            merged_df["shooting_component"] * 0.5 +
            merged_df["creation_component"] * 0.5
        )

        # Classify players based on the balance
        conditions = [
            (merged_df["shooting_component"] > merged_df["creation_component"] * 1.5),
            (merged_df["creation_component"] > merged_df["shooting_component"] * 1.5),
            (abs(merged_df["shooting_component"] - merged_df["creation_component"]) < 0.1)
        ]

        categories = ["Shooter", "Creator", "Balanced Contributor"]

        merged_df["contribution_type"] = np.select(
            conditions, categories, default="Mixed Contributor"
        )

    return merged_df.sort_values("shot_contribution_score", ascending=False)


def calculate_finishing_skill_over_time(
    shooting_df: pd.DataFrame,
    min_90s: float = 10,
    min_shots: int = 30
) -> pd.DataFrame:
    """
    Analyze a player's finishing skill (G-xG) normalized by shots taken.

    Args:
        shooting_df: DataFrame containing shooting statistics
        min_90s: Minimum number of 90-minute periods played
        min_shots: Minimum shots to consider

    Returns:
        DataFrame with finishing skill metrics
    """
    if shooting_df.empty:
        return pd.DataFrame()

    # Filter dataset
    filtered_df = shooting_df[(shooting_df["90s"] >= min_90s) &
                             (shooting_df["Sh"] >= min_shots)].copy()

    if filtered_df.empty:
        logger.warning(f"No players with at least {min_shots} shots and {min_90s} 90s played")
        return pd.DataFrame()

    # Calculate finishing metrics
    filtered_df["goals_above_xG"] = filtered_df["Gls"] - filtered_df["xG"]
    filtered_df["np_goals_above_xG"] = filtered_df["Gls"] - filtered_df["PK"] - filtered_df["npxG"]
    filtered_df["finishing_per_shot"] = filtered_df["goals_above_xG"] / filtered_df["Sh"]
    filtered_df["np_finishing_per_shot"] = filtered_df["np_goals_above_xG"] / (filtered_df["Sh"] - filtered_df["PKatt"])

    # Normalize so average is 100
    avg_finishing = filtered_df["finishing_per_shot"].mean()
    if avg_finishing != 0:
        filtered_df["finishing_index"] = (filtered_df["finishing_per_shot"] / avg_finishing) * 100
    else:
        filtered_df["finishing_index"] = 100

    avg_np_finishing = filtered_df["np_finishing_per_shot"].mean()
    if avg_np_finishing != 0:
        filtered_df["np_finishing_index"] = (filtered_df["np_finishing_per_shot"] / avg_np_finishing) * 100
    else:
        filtered_df["np_finishing_index"] = 100

    # Categorize finishers
    # Above 115: Elite, 105-115: Good, 95-105: Average, 85-95: Below average, Below 85: Poor
    conditions = [
        (filtered_df["np_finishing_index"] >= 115),
        (filtered_df["np_finishing_index"] >= 105) & (filtered_df["np_finishing_index"] < 115),
        (filtered_df["np_finishing_index"] >= 95) & (filtered_df["np_finishing_index"] < 105),
        (filtered_df["np_finishing_index"] >= 85) & (filtered_df["np_finishing_index"] < 95),
        (filtered_df["np_finishing_index"] < 85)
    ]

    categories = ["Elite Finisher", "Good Finisher", "Average Finisher",
                 "Below Average Finisher", "Poor Finisher"]

    filtered_df["finishing_category"] = np.select(conditions, categories, default="Unclassified")

    return filtered_df.sort_values("np_finishing_index", ascending=False)


def analyze_shot_quality(
    shooting_df: pd.DataFrame,
    min_shots: int = 20
) -> pd.DataFrame:
    """
    Analyze shot quality based on xG per shot and shot location metrics.

    Args:
        shooting_df: DataFrame containing shooting statistics
        min_shots: Minimum shots to consider

    Returns:
        DataFrame with shot quality metrics
    """
    if shooting_df.empty:
        return pd.DataFrame()

    # Filter by minimum shots
    filtered_df = shooting_df[shooting_df["Sh"] >= min_shots].copy()

    if filtered_df.empty:
        logger.warning(f"No players with at least {min_shots} shots")
        return pd.DataFrame()

    # Calculate shot quality metrics
    filtered_df["xG_per_shot"] = filtered_df["xG"] / filtered_df["Sh"]
    filtered_df["npxG_per_shot"] = filtered_df["npxG"] / (filtered_df["Sh"] - filtered_df["PKatt"])
    filtered_df["shot_placement"] = filtered_df["SoT"] / filtered_df["Sh"]

    # Shot distance is already in the data

    # Shot selection score (weighted shot quality)
    filtered_df["shot_selection_score"] = (
        normalize_metric(filtered_df["npxG_per_shot"]) * 0.5 +
        normalize_metric(filtered_df["shot_placement"]) * 0.3 -
        normalize_metric(filtered_df["Dist"]) * 0.2  # Lower distance is better
    )

    # Categorize shot selectors
    conditions = [
        (filtered_df["shot_selection_score"] >= 0.8),
        (filtered_df["shot_selection_score"] >= 0.6) & (filtered_df["shot_selection_score"] < 0.8),
        (filtered_df["shot_selection_score"] >= 0.4) & (filtered_df["shot_selection_score"] < 0.6),
        (filtered_df["shot_selection_score"] >= 0.2) & (filtered_df["shot_selection_score"] < 0.4),
        (filtered_df["shot_selection_score"] < 0.2)
    ]

    categories = ["Elite Shot Selector", "Good Shot Selector", "Average Shot Selector",
                 "Below Average Shot Selector", "Poor Shot Selector"]

    filtered_df["shot_selection_category"] = np.select(conditions, categories, default="Unclassified")

    return filtered_df.sort_values("shot_selection_score", ascending=False)
