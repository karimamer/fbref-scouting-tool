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
from config.settings import ANALYSIS_WEIGHTS, DEFAULT_ANALYSIS_PARAMS

def calculate_versatility_score(
    passing_df: pd.DataFrame,
    possession_df: pd.DataFrame,
    defensive_df: pd.DataFrame,
    shooting_df: pd.DataFrame = None,
    min_90s: float = DEFAULT_ANALYSIS_PARAMS["min_90s"]
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
    # Passing component - use weights from config if available
    passing_cols = ["passes_per_90", "prog_passes_per_90", "key_passes_per_90"]
    if "xA_per_90" in passing_filtered.columns:
        passing_cols.append("xA_per_90")

    for col in passing_cols:
        passing_filtered[f"{col}_norm"] = normalize_metric(passing_filtered[col])

    # Get weights from config for playmaker or use default
    if "playmaker" in ANALYSIS_WEIGHTS:
        # Map config weights to our column names
        playmaker_weights = {
            "passes_per_90_norm": ANALYSIS_WEIGHTS["playmaker"].get("total_Cmp%_norm", 0.25),
            "prog_passes_per_90_norm": ANALYSIS_WEIGHTS["playmaker"].get("PrgP_90_norm", 0.35),
            "key_passes_per_90_norm": ANALYSIS_WEIGHTS["playmaker"].get("KP_90_norm", 0.30),
            "xA_per_90_norm": ANALYSIS_WEIGHTS["playmaker"].get("Ast_90_norm", 0.10)
        }

        # Adjust weights for actual available columns
        available_weights = {k: v for k, v in playmaker_weights.items() if k.split('_norm')[0] in passing_cols}
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}

        passing_filtered["passing_score"] = sum(
            passing_filtered[f"{col}_norm"] * normalized_weights.get(f"{col}_norm", 1/len(passing_cols))
            for col in passing_cols
        )
    else:
        # Default approach if no configs available
        passing_filtered["passing_score"] = passing_filtered[[f"{col}_norm" for col in passing_cols]].mean(axis=1)

    # Possession component - use weights from config if available
    possession_cols = ["carries_per_90", "prog_carries_per_90", "carries_into_final_third_per_90"]
    for col in possession_cols:
        possession_filtered[f"{col}_norm"] = normalize_metric(possession_filtered[col])

    # Get weights from config for progressive or use default
    if "progressive" in ANALYSIS_WEIGHTS:
        # Map config weights to our column names
        progressive_weights = {
            "carries_per_90_norm": 0.15,  # Not in config, adding default
            "prog_carries_per_90_norm": ANALYSIS_WEIGHTS["progressive"].get("PrgC_norm", 0.30),
            "carries_into_final_third_per_90_norm": ANALYSIS_WEIGHTS["progressive"].get("1/3_norm", 0.20)
        }

        # Adjust weights for actual available columns
        available_weights = {k: v for k, v in progressive_weights.items() if k.split('_norm')[0] in possession_cols}
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}

        possession_filtered["possession_score"] = sum(
            possession_filtered[f"{col}_norm"] * normalized_weights.get(f"{col}_norm", 1/len(possession_cols))
            for col in possession_cols
        )
    else:
        # Default approach if no configs available
        possession_filtered["possession_score"] = possession_filtered[[f"{col}_norm" for col in possession_cols]].mean(axis=1)

    # Defensive component - use weights from config if available
    defensive_cols = ["tackles_per_90", "interceptions_per_90"]
    if "blocks_per_90" in defensive_filtered.columns:
        defensive_cols.append("blocks_per_90")

    for col in defensive_cols:
        defensive_filtered[f"{col}_norm"] = normalize_metric(defensive_filtered[col])

    # Get weights from config for pressing or use default
    if "pressing" in ANALYSIS_WEIGHTS:
        # Map config weights to our column names
        pressing_weights = {
            "tackles_per_90_norm": ANALYSIS_WEIGHTS["pressing"].get("Tkl_90_norm", 0.35),
            "interceptions_per_90_norm": ANALYSIS_WEIGHTS["pressing"].get("Int_90_norm", 0.30)
        }

        # Add blocks if available
        if "blocks_per_90" in defensive_filtered.columns:
            pressing_weights["blocks_per_90_norm"] = 0.15  # Default value

        # Adjust weights for actual available columns
        available_weights = {k: v for k, v in pressing_weights.items() if k.split('_norm')[0] in defensive_cols}
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}

        defensive_filtered["defensive_score"] = sum(
            defensive_filtered[f"{col}_norm"] * normalized_weights.get(f"{col}_norm", 1/len(defensive_cols))
            for col in defensive_cols
        )
    else:
        # Default approach if no configs available
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

        # Get weights from config for forwards or use default
        if "forward" in ANALYSIS_WEIGHTS:
            # Map config weights to our column names
            forward_weights = {
                "shots_per_90_norm": 0.15,  # Not in config, adding default
                "goals_per_90_norm": ANALYSIS_WEIGHTS["forward"].get("Gls_90_norm", 0.20),
                "xG_per_90_norm": 0.15  # Not directly in config
            }

            # Adjust weights for actual available columns
            available_weights = {k: v for k, v in forward_weights.items() if k.split('_norm')[0] in shooting_cols}
            total_weight = sum(available_weights.values())
            normalized_weights = {k: v/total_weight for k, v in available_weights.items()}

            shooting_filtered["shooting_score"] = sum(
                shooting_filtered[f"{col}_norm"] * normalized_weights.get(f"{col}_norm", 1/len(shooting_cols))
                for col in shooting_cols
            )
        else:
            # Default approach if no configs available
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

        # Use complete_midfielder weights if available
        if "complete_midfielder" in ANALYSIS_WEIGHTS:
            component_weights = {
                "passing_score": ANALYSIS_WEIGHTS["complete_midfielder"].get("playmaker_score_norm", 0.30),
                "possession_score": ANALYSIS_WEIGHTS["complete_midfielder"].get("progression_score_norm", 0.40),
                "defensive_score": ANALYSIS_WEIGHTS["complete_midfielder"].get("pressing_score_norm", 0.30),
                "shooting_score": 0.15  # Additional component not in config
            }

            # Normalize weights to sum to 1
            total_weight = sum(component_weights.values())
            component_weights = {k: v/total_weight for k, v in component_weights.items()}
        else:
            component_weights = {
                "passing_score": 0.25,
                "possession_score": 0.25,
                "defensive_score": 0.25,
                "shooting_score": 0.25
            }
    else:
        # Use complete_midfielder weights if available
        if "complete_midfielder" in ANALYSIS_WEIGHTS:
            component_weights = {
                "passing_score": ANALYSIS_WEIGHTS["complete_midfielder"].get("playmaker_score_norm", 0.33),
                "possession_score": ANALYSIS_WEIGHTS["complete_midfielder"].get("progression_score_norm", 0.33),
                "defensive_score": ANALYSIS_WEIGHTS["complete_midfielder"].get("pressing_score_norm", 0.34)
            }

            # Normalize weights to sum to 1
            total_weight = sum(component_weights.values())
            component_weights = {k: v/total_weight for k, v in component_weights.items()}
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
        if component in versatility.columns
    )

    # Add standard deviation of component scores to measure consistency across areas
    components = [c for c in component_weights.keys() if c in versatility.columns]
    versatility["consistency"] = versatility[components].std(axis=1)

    # Final adjustments - higher consistency (std dev) means less versatile
    versatility["adjusted_versatility"] = versatility["versatility_score"] * (1 - normalize_metric(versatility["consistency"]))

    return versatility.sort_values("adjusted_versatility", ascending=False)
