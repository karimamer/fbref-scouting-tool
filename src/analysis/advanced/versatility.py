from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.analysis.metrics import (
    normalize_metric,
    calculate_per_90_metrics,
    calculate_weighted_score,
    get_score_from_config
)


def calculate_versatility_score(
    passing_df: pd.DataFrame,
    possession_df: pd.DataFrame,
    defensive_df: pd.DataFrame,
    shooting_df: pd.DataFrame = None,
    min_90s: float = 10.0
) -> pd.DataFrame:
    """
    Calculate a versatility score for players based on their performance
    across multiple skill areas.

    Parameters:
    -----------
    passing_df: DataFrame containing passing statistics
    possession_df: DataFrame containing possession statistics
    defensive_df: DataFrame containing defensive statistics
    shooting_df: DataFrame containing shooting statistics (optional)
    min_90s: Minimum number of 90s played to be considered

    Returns:
    --------
    DataFrame with versatility scores and component scores
    """
    # Filter for minimum playing time
    passing_filtered = passing_df[passing_df["90s"] >= min_90s].copy()
    possession_filtered = possession_df[possession_df["90s"] >= min_90s].copy()
    defensive_filtered = defensive_df[defensive_df["90s"] >= min_90s].copy()

    # Calculate per-90 metrics for key stats
    # Passing metrics
    passing_filtered["passes_per_90"] = passing_filtered["total_cmp"] / passing_filtered["90s"]
    passing_filtered["prog_passes_per_90"] = passing_filtered["PrgP"] / passing_filtered["90s"]
    passing_filtered["key_passes_per_90"] = passing_filtered["KP"] / passing_filtered["90s"]
    if "xA" in passing_filtered.columns:
        passing_filtered["xA_per_90"] = passing_filtered["xA"] / passing_filtered["90s"]

    # Possession metrics
    possession_filtered["carries_per_90"] = possession_filtered["Carries"] / possession_filtered["90s"]
    possession_filtered["prog_carries_per_90"] = possession_filtered["PrgC"] / possession_filtered["90s"]
    possession_filtered["carries_into_final_third_per_90"] = possession_filtered["1/3"] / possession_filtered["90s"]

    # Defensive metrics
    defensive_filtered["tackles_per_90"] = defensive_filtered["Tkl"] / defensive_filtered["90s"]
    defensive_filtered["interceptions_per_90"] = defensive_filtered["Int"] / defensive_filtered["90s"]
    if "Blocks" in defensive_filtered.columns:
        defensive_filtered["blocks_per_90"] = defensive_filtered["Blocks"] / defensive_filtered["90s"]

    # Create normalized component scores
    # Passing component
    passing_cols = ["passes_per_90", "prog_passes_per_90", "key_passes_per_90"]
    if "xA_per_90" in passing_filtered.columns:
        passing_cols.append("xA_per_90")

    for col in passing_cols:
        passing_filtered[f"{col}_norm"] = normalize_metric(passing_filtered[col])

    passing_filtered["passing_score"] = passing_filtered[[f"{col}_norm" for col in passing_cols]].mean(axis=1)

    # Possession component
    possession_cols = ["carries_per_90", "prog_carries_per_90", "carries_into_final_third_per_90"]
    for col in possession_cols:
        possession_filtered[f"{col}_norm"] = normalize_metric(possession_filtered[col])

    possession_filtered["possession_score"] = possession_filtered[[f"{col}_norm" for col in possession_cols]].mean(axis=1)

    # Defensive component
    defensive_cols = ["tackles_per_90", "interceptions_per_90"]
    if "blocks_per_90" in defensive_filtered.columns:
        defensive_cols.append("blocks_per_90")

    for col in defensive_cols:
        defensive_filtered[f"{col}_norm"] = normalize_metric(defensive_filtered[col])

    defensive_filtered["defensive_score"] = defensive_filtered[[f"{col}_norm" for col in defensive_cols]].mean(axis=1)

    # If shooting data is provided, add shooting component
    shooting_score = None
    if shooting_df is not None:
        shooting_filtered = shooting_df[shooting_df["90s"] >= min_90s].copy()
        shooting_filtered["shots_per_90"] = shooting_filtered["Sh"] / shooting_filtered["90s"]
        shooting_filtered["goals_per_90"] = shooting_filtered["Gls"] / shooting_filtered["90s"]
        if "xG" in shooting_filtered.columns:
            shooting_filtered["xG_per_90"] = shooting_filtered["xG"] / shooting_filtered["90s"]

        shooting_cols = ["shots_per_90", "goals_per_90"]
        if "xG_per_90" in shooting_filtered.columns:
            shooting_cols.append("xG_per_90")

        for col in shooting_cols:
            shooting_filtered[f"{col}_norm"] = normalize_metric(shooting_filtered[col])

        shooting_filtered["shooting_score"] = shooting_filtered[[f"{col}_norm" for col in shooting_cols]].mean(axis=1)
        shooting_score = shooting_filtered[["Player", "Squad", "Pos", "shooting_score"]]

    # Combine all components
    base_cols = ["Player", "Squad", "Pos", "Age", "90s"]
    passing_component = passing_filtered[base_cols + ["passing_score"]]
    possession_component = possession_filtered[base_cols + ["possession_score"]]
    defensive_component = defensive_filtered[base_cols + ["defensive_score"]]

    # Merge all components
    versatility = passing_component.merge(
        possession_component[["Player", "Squad", "possession_score"]],
        on=["Player", "Squad"],
        how="inner"
    )

    versatility = versatility.merge(
        defensive_component[["Player", "Squad", "defensive_score"]],
        on=["Player", "Squad"],
        how="inner"
    )

    if shooting_score is not None:
        versatility = versatility.merge(
            shooting_score[["Player", "Squad", "shooting_score"]],
            on=["Player", "Squad"],
            how="left"
        )
        versatility["shooting_score"] = versatility["shooting_score"].fillna(0)
        component_weights = {
            "passing_score": 0.25,
            "possession_score": 0.25,
            "defensive_score": 0.25,
            "shooting_score": 0.25
        }
    else:
        component_weights = {
            "passing_score": 0.33,
            "possession_score": 0.33,
            "defensive_score": 0.34
        }

    # Calculate weighted versatility score
    versatility["versatility_score"] = sum(
        versatility[component] * weight
        for component, weight in component_weights.items()
    )

    # Add standard deviation of component scores to measure consistency across areas
    components = list(component_weights.keys())
    versatility["consistency"] = versatility[components].std(axis=1)

    # Final adjustments - higher consistency (std dev) means less versatile
    versatility["adjusted_versatility"] = versatility["versatility_score"] * (1 - normalize_metric(versatility["consistency"]))

    return versatility.sort_values("adjusted_versatility", ascending=False)
