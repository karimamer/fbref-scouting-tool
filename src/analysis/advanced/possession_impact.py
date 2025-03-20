import pandas as pd
from config.settings import ANALYSIS_WEIGHTS, DEFAULT_ANALYSIS_PARAMS
from src.analysis.metrics import (
    normalize_metric,
    calculate_per_90_metrics,
    calculate_weighted_score,
    get_score_from_config
)

def get_expected_possession_impact(possession_df: pd.DataFrame, min_90s: float = DEFAULT_ANALYSIS_PARAMS["min_90s"]) -> pd.DataFrame:
    """
    Calculate Expected Possession Impact (xPI) - a metric estimating a player's overall
    contribution to team possession.

    Parameters:
    -----------
    possession_df: DataFrame with possession statistics
    min_90s: Minimum 90s played to be included

    Returns:
    --------
    DataFrame with xPI metrics
    """
    # Filter by playing time
    poss = possession_df[possession_df["90s"] >= min_90s].copy()

    # Calculate per 90 metrics
    poss["touches_90"] = poss["Touches"] / poss["90s"]
    poss["carries_90"] = poss["Carries"] / poss["90s"]
    poss["succ_dribbles_90"] = poss["Succ"] / poss["90s"]
    poss["prog_carries_90"] = poss["PrgC"] / poss["90s"]
    poss["final_third_entries_90"] = poss["1/3"] / poss["90s"]
    poss["penalty_area_entries_90"] = poss.get("CPA", 0) / poss["90s"]
    poss["prog_receives_90"] = poss["PrgR"] / poss["90s"]

    # Calculate possession retention ratio
    poss["possession_actions"] = poss["Carries"] + poss["Rec"]
    poss["possession_losses"] = poss["Mis"] + poss["Dis"]
    poss["retention_ratio"] = 1 - (poss["possession_losses"] / poss["possession_actions"])

    # Normalize metrics
    metrics = [
        "touches_90", "carries_90", "succ_dribbles_90", "prog_carries_90",
        "final_third_entries_90", "prog_receives_90", "retention_ratio"
    ]

    for metric in metrics:
        if metric in poss.columns:
            poss[f"{metric}_norm"] = normalize_metric(poss[metric])

    # Include penalty area entries if available
    if "penalty_area_entries_90" in poss.columns:
        metrics.append("penalty_area_entries_90")
        poss["penalty_area_entries_90_norm"] = normalize_metric(poss["penalty_area_entries_90"])

    # Weight components using config weights if available
    # Try to map our metrics to the progressive metrics in the config
    if "progressive" in ANALYSIS_WEIGHTS:
        weights = {
            "touches_90_norm": 0.05,  # Base weight
            "carries_90_norm": 0.10,  # Base weight
            "succ_dribbles_90_norm": 0.15,  # Base weight
            "prog_carries_90_norm": ANALYSIS_WEIGHTS["progressive"].get("PrgC_norm", 0.20),
            "final_third_entries_90_norm": ANALYSIS_WEIGHTS["progressive"].get("1/3_norm", 0.15),
            "prog_receives_90_norm": ANALYSIS_WEIGHTS["progressive"].get("PrgR_norm", 0.15),
            "retention_ratio_norm": 0.20  # Base weight
        }

        if "penalty_area_entries_90_norm" in poss.columns:
            weights["penalty_area_entries_90_norm"] = 0.15  # Additional weight

            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
    else:
        # Default weights if config not available
        weights = {
            "touches_90_norm": 0.05,
            "carries_90_norm": 0.10,
            "succ_dribbles_90_norm": 0.15,
            "prog_carries_90_norm": 0.20,
            "final_third_entries_90_norm": 0.15,
            "prog_receives_90_norm": 0.15,
            "retention_ratio_norm": 0.20
        }

        if "penalty_area_entries_90_norm" in poss.columns:
            # Adjust weights to make room for penalty area entries
            weights = {k: v * 0.85 for k, v in weights.items()}
            weights["penalty_area_entries_90_norm"] = 0.15

    # Calculate xPI
    poss["xPI"] = sum(
        poss[metric] * weight
        for metric, weight in weights.items()
        if metric in poss.columns
    )

    # Position adjustments - normalize xPI within position groups
    position_groups = {
        "Defenders": poss["Pos"].str.contains("DF"),
        "Midfielders": poss["Pos"].str.contains("MF"),
        "Forwards": poss["Pos"].str.contains("FW")
    }

    poss["position_group"] = "Other"
    for group, mask in position_groups.items():
        poss.loc[mask, "position_group"] = group

    # For each position group, calculate relative xPI
    for group in position_groups:
        group_mask = poss["position_group"] == group
        if group_mask.sum() > 0:
            poss.loc[group_mask, "position_relative_xPI"] = normalize_metric(poss.loc[group_mask, "xPI"])

    return poss.sort_values("xPI", ascending=False)
