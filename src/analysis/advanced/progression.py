from typing import Dict
import pandas as pd
import numpy as np
from config.settings import ANALYSIS_WEIGHTS, DEFAULT_ANALYSIS_PARAMS
from src.analysis.metrics import (
    normalize_metric,
    calculate_per_90_metrics,
    calculate_weighted_score,
    get_score_from_config
)


def analyze_progressive_actions(
    possession_df: pd.DataFrame,
    passing_df: pd.DataFrame,
    min_90s: float = DEFAULT_ANALYSIS_PARAMS["min_90s"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"]
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive analysis of players' progressive actions.

    Parameters:
    -----------
    possession_df: DataFrame with possession statistics
    passing_df: DataFrame with passing statistics
    min_90s: Minimum 90s played to be included
    top_n: Number of top players to return in each category

    Returns:
    --------
    Dictionary with various progressive action analyses
    """
    # Filter players with minimum playing time
    possession = possession_df[possession_df["90s"] >= min_90s].copy()
    passing = passing_df[passing_df["90s"] >= min_90s].copy()

    # Calculate per 90 metrics for progressive actions
    possession["PrgC_90"] = possession["PrgC"] / possession["90s"]
    possession["PrgDist_90"] = possession["PrgDist"] / possession["90s"]
    possession["final_third_entries_90"] = possession["1/3"] / possession["90s"]
    possession["penalty_area_entries_90"] = possession.get("CPA", 0) / possession["90s"]
    possession["progressive_receives_90"] = possession["PrgR"] / possession["90s"]

    passing["PrgP_90"] = passing["PrgP"] / passing["90s"]

    # Merge the relevant metrics
    base_cols = ["Player", "Squad", "Pos", "Age", "90s"]
    prog_cols_possession = [
        "PrgC_90", "PrgDist_90", "final_third_entries_90",
        "penalty_area_entries_90", "progressive_receives_90"
    ]

    prog_cols_passing = ["PrgP_90"]

    # Create dataframe with all progressive metrics
    progression = possession[base_cols + prog_cols_possession].merge(
        passing[["Player", "Squad"] + prog_cols_passing],
        on=["Player", "Squad"],
        how="inner"
    )

    # Calculate composite scores
    # Carrying progression score - use config weights if available
    if "progressive" in ANALYSIS_WEIGHTS:
        # Try to map our metrics to config
        carrying_weights = {
            "PrgC_90": ANALYSIS_WEIGHTS["progressive"].get("PrgC_norm", 0.40),
            "PrgDist_90": ANALYSIS_WEIGHTS["progressive"].get("PrgDist_norm", 0.30),
            "final_third_entries_90": ANALYSIS_WEIGHTS["progressive"].get("1/3_norm", 0.20),
            "penalty_area_entries_90": 0.10  # Default weight
        }
    else:
        # Default weights if config not available
        carrying_weights = {
            "PrgC_90": 0.40,
            "PrgDist_90": 0.30,
            "final_third_entries_90": 0.20,
            "penalty_area_entries_90": 0.10
        }

    for col in carrying_weights:
        if col in progression.columns:
            progression[f"{col}_norm"] = normalize_metric(progression[col])
        else:
            progression[f"{col}_norm"] = 0

    progression["carrying_progression_score"] = sum(
        progression[f"{col}_norm"] * weight
        for col, weight in carrying_weights.items()
        if f"{col}_norm" in progression.columns
    )

    # Passing progression score
    progression["PrgP_90_norm"] = normalize_metric(progression["PrgP_90"])
    progression["passing_progression_score"] = progression["PrgP_90_norm"]

    # Receiving progression score
    progression["progressive_receives_90_norm"] = normalize_metric(progression["progressive_receives_90"])
    progression["receiving_progression_score"] = progression["progressive_receives_90_norm"]

    # Overall progression score with config weights if available
    # Attempt to create composite weights by combining playmaker and progressive configs
    overall_weights = {}

    if "progressive" in ANALYSIS_WEIGHTS and "playmaker" in ANALYSIS_WEIGHTS:
        # Progressive weights for carrying
        carrying_weight = sum(ANALYSIS_WEIGHTS["progressive"].get(k, 0) for k in ["PrgC_norm", "PrgDist_norm", "1/3_norm"]) / 3
        # Playmaker weights for passing
        passing_weight = ANALYSIS_WEIGHTS["playmaker"].get("PrgP_90_norm", 0.35)
        # Progressive weights for receiving
        receiving_weight = ANALYSIS_WEIGHTS["progressive"].get("PrgR_norm", 0.15)

        total_weight = carrying_weight + passing_weight + receiving_weight
        overall_weights = {
            "carrying_progression_score": carrying_weight / total_weight,
            "passing_progression_score": passing_weight / total_weight,
            "receiving_progression_score": receiving_weight / total_weight
        }
    else:
        overall_weights = {
            "carrying_progression_score": 0.4,
            "passing_progression_score": 0.4,
            "receiving_progression_score": 0.2
        }

    progression["total_progression_score"] = sum(
        progression[component] * weight
        for component, weight in overall_weights.items()
    )

    # Identify specialists and all-rounders
    progression["progression_versatility"] = 1 - progression[
        ["carrying_progression_score", "passing_progression_score", "receiving_progression_score"]
    ].std(axis=1)

    # Classification based on strengths
    conditions = [
        (progression["carrying_progression_score"] > progression["passing_progression_score"]) &
        (progression["carrying_progression_score"] > progression["receiving_progression_score"]),

        (progression["passing_progression_score"] > progression["carrying_progression_score"]) &
        (progression["passing_progression_score"] > progression["receiving_progression_score"]),

        (progression["receiving_progression_score"] > progression["carrying_progression_score"]) &
        (progression["receiving_progression_score"] > progression["passing_progression_score"])
    ]

    choices = ["Carrier", "Passer", "Receiver"]
    progression["progression_type"] = np.select(conditions, choices, default="Balanced")

    # Prepare return dict with different sorted views
    results = {
        "overall_progressors": progression.sort_values("total_progression_score", ascending=False).head(top_n),
        "top_carriers": progression.sort_values("carrying_progression_score", ascending=False).head(top_n),
        "top_passers": progression.sort_values("passing_progression_score", ascending=False).head(top_n),
        "top_receivers": progression.sort_values("receiving_progression_score", ascending=False).head(top_n),
        "versatile_progressors": progression.sort_values(["total_progression_score", "progression_versatility"],
                                                       ascending=[False, False]).head(top_n)
    }

    return results
