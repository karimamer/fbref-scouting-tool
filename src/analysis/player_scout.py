"""
Player scouting analysis functions.
"""
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

def identify_playmakers(passing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify creative midfielders based on progressive passing and creation metrics.

    Args:
        passing_df: DataFrame containing passing statistics

    Returns:
        DataFrame with playmaker scores
    """
    if passing_df.empty:
        return pd.DataFrame()

    playmaker_metrics = passing_df.copy()

    # Calculate per-90 metrics
    per90_metrics = ["PrgP", "KP", "Ast"]
    playmaker_metrics = calculate_per_90_metrics(playmaker_metrics, per90_metrics)

    # Define metrics to use for playmaker assessment
    metrics = {
        "PrgP_90": ANALYSIS_WEIGHTS["playmaker"]["PrgP_90_norm"],
        "KP_90": ANALYSIS_WEIGHTS["playmaker"]["KP_90_norm"],
        "total_Cmp%": ANALYSIS_WEIGHTS["playmaker"]["total_Cmp%_norm"],
        "Ast_90": ANALYSIS_WEIGHTS["playmaker"]["Ast_90_norm"]
    }

    # Calculate playmaker score
    result = calculate_weighted_score(
        playmaker_metrics,
        metrics,
        "playmaker_score"
    )

    return result.sort_values("playmaker_score", ascending=False)


def find_clinical_forwards(
    shooting_df: pd.DataFrame,
    min_shots: int = 20
) -> pd.DataFrame:
    """
    Identify efficient forwards based on shooting and conversion metrics.

    Args:
        shooting_df: DataFrame containing shooting statistics
        min_shots: Minimum number of shots taken

    Returns:
        DataFrame with efficiency scores
    """
    if shooting_df.empty:
        return pd.DataFrame()

    # Filter by minimum shots
    shooting_analysis = shooting_df[shooting_df["Sh"] >= min_shots].copy()

    if shooting_analysis.empty:
        logger.warning(f"No players with at least {min_shots} shots")
        return pd.DataFrame()

    # Calculate derived metrics
    shooting_analysis["Sh_90"] = shooting_analysis["Sh"] / shooting_analysis["90s"]
    shooting_analysis["Gls_90"] = shooting_analysis["Gls"] / shooting_analysis["90s"]
    shooting_analysis["conversion_rate"] = shooting_analysis["Gls"] / shooting_analysis["Sh"]
    shooting_analysis["xG_difference"] = shooting_analysis["Gls"] - shooting_analysis["xG"]

    # Define metrics for forward efficiency
    metrics = {
        "conversion_rate": ANALYSIS_WEIGHTS["forward"]["conversion_rate_norm"],
        "SoT%": ANALYSIS_WEIGHTS["forward"]["SoT%_norm"],
        "xG_difference": ANALYSIS_WEIGHTS["forward"]["xG_difference_norm"],
        "Gls_90": ANALYSIS_WEIGHTS["forward"]["Gls_90_norm"]
    }

    # Calculate efficiency score
    result = calculate_weighted_score(
        shooting_analysis,
        metrics,
        "efficiency_score"
    )

    return result.sort_values("efficiency_score", ascending=False)


def analyze_progressive_midfielders(possession_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify midfielders who excel at moving the ball forward.

    Args:
        possession_df: DataFrame containing possession statistics

    Returns:
        DataFrame with progression scores
    """
    if possession_df.empty:
        return pd.DataFrame()

    midfield_metrics = possession_df.copy()

    # Calculate per-90 metrics for progression
    per90_metrics = ["PrgDist", "PrgC", "1/3", "PrgR"]
    midfield_metrics = calculate_per_90_metrics(midfield_metrics, per90_metrics)

    # Get weights from configuration
    metrics = {}
    for metric in per90_metrics:
        metrics[f"{metric}_90"] = ANALYSIS_WEIGHTS["progressive"].get(f"{metric}_norm", 0.25)

    # Calculate progression score
    result = calculate_weighted_score(
        midfield_metrics,
        metrics,
        "progression_score"
    )

    return result.sort_values("progression_score", ascending=False)


def identify_pressing_midfielders(defensive_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find midfielders who excel in pressing and defensive actions.

    Args:
        defensive_df: DataFrame containing defensive statistics

    Returns:
        DataFrame with pressing scores
    """
    if defensive_df.empty:
        return pd.DataFrame()

    # Filter for midfielders
    defensive_mids = defensive_df[
        defensive_df["Pos"].str.contains("MF", na=False)
    ].copy()

    if defensive_mids.empty:
        logger.warning("No midfielders found in defensive statistics")
        return pd.DataFrame()

    # Calculate per-90 metrics
    per90_metrics = ["Tkl", "Int", "Att 3rd"]
    defensive_mids = calculate_per_90_metrics(defensive_mids, per90_metrics)

    # Rename Att 3rd per 90 for consistency
    if "Att 3rd_90" in defensive_mids.columns:
        defensive_mids["Att3rd_90"] = defensive_mids["Att 3rd_90"]

    # Define metrics for pressing assessment
    metrics = {
        "Tkl_90": ANALYSIS_WEIGHTS["pressing"]["Tkl_90_norm"],
        "Int_90": ANALYSIS_WEIGHTS["pressing"]["Int_90_norm"],
        "Tkl%": ANALYSIS_WEIGHTS["pressing"]["Tkl%_norm"],
        "Att3rd_90": ANALYSIS_WEIGHTS["pressing"]["Att3rd_90_norm"]
    }

    # Calculate pressing score
    result = calculate_weighted_score(
        defensive_mids,
        metrics,
        "pressing_score"
    )

    return result.sort_values("pressing_score", ascending=False)


def find_complete_midfielders(
    passing_df: pd.DataFrame,
    possession_df: pd.DataFrame,
    defensive_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify well-rounded midfielders who contribute in multiple areas.

    Args:
        passing_df: DataFrame containing passing statistics
        possession_df: DataFrame containing possession statistics
        defensive_df: DataFrame containing defensive statistics

    Returns:
        DataFrame with complete midfielder scores
    """
    # Calculate individual component scores
    progressive = analyze_progressive_midfielders(possession_df)
    defensive = identify_pressing_midfielders(defensive_df)
    playmaking = identify_playmakers(passing_df)

    if progressive.empty or defensive.empty or playmaking.empty:
        logger.warning("One or more component analyses empty, cannot calculate complete midfielder score")
        return pd.DataFrame()

    # Combine scores
    base_cols = ["Player", "Squad", "Age", "Pos"]
    complete_score = progressive[base_cols + ["progression_score"]].copy()

    # Merge scores from different analyses
    complete_score = complete_score.merge(
        defensive[["Player", "pressing_score"]],
        on="Player",
        how="inner"
    )
    complete_score = complete_score.merge(
        playmaking[["Player", "playmaker_score"]],
        on="Player",
        how="inner"
    )

    # Normalize component scores
    for col in ["progression_score", "pressing_score", "playmaker_score"]:
        complete_score[f"{col}_norm"] = normalize_metric(complete_score[col])

    # Calculate complete midfielder score using weights from config
    weights = ANALYSIS_WEIGHTS["complete_midfielder"]
    complete_score["complete_midfielder_score"] = (
        complete_score["progression_score_norm"] * weights["progression_score_norm"] +
        complete_score["pressing_score_norm"] * weights["pressing_score_norm"] +
        complete_score["playmaker_score_norm"] * weights["playmaker_score_norm"]
    )

    return complete_score.sort_values("complete_midfielder_score", ascending=False)


def analyze_passing_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze passing quality and chance creation for players.

    Args:
        df: DataFrame containing passing statistics

    Returns:
        DataFrame with passing quality scores
    """
    if df.empty:
        return df

    # Avoid modifying the original dataframe
    df_filtered = df.copy()

    # Calculate per-90 metrics
    df_filtered['passes_per_90'] = df_filtered['total_cmp'] / df_filtered['90s']
    df_filtered['prog_passes_per_90'] = df_filtered['PrgP'] / df_filtered['90s']
    df_filtered['key_passes_per_90'] = df_filtered['KP'] / df_filtered['90s']
    df_filtered['xA_per_90'] = df_filtered['xA'] / df_filtered['90s']

    # Calculate component scores
    df_filtered['passing_accuracy_score'] = (
        df_filtered['total_Cmp%'] / 100 *
        np.where(df_filtered['total_cmp'] > df_filtered['total_cmp'].median(), 1.2, 1.0)
    )

    df_filtered['progression_score'] = (
        df_filtered['prog_passes_per_90'] / df_filtered['prog_passes_per_90'].max() +
        df_filtered['PrgDist'] / df_filtered['PrgDist'].max()
    ) / 2

    df_filtered['chance_creation_score'] = (
        df_filtered['key_passes_per_90'] / df_filtered['key_passes_per_90'].max() +
        df_filtered['xA_per_90'] / df_filtered['xA_per_90'].max() +
        df_filtered['PPA'] / df_filtered['PPA'].max()
    ) / 3

    # Calculate overall passing quality score
    df_filtered['passing_quality_score'] = (
        df_filtered['passing_accuracy_score'] * 0.3 +
        df_filtered['progression_score'] * 0.3 +
        df_filtered['chance_creation_score'] * 0.4
    )

    # Round for readability
    cols_to_round = [
        'passes_per_90', 'prog_passes_per_90', 'key_passes_per_90', 'xA_per_90',
        'passing_accuracy_score', 'progression_score', 'chance_creation_score',
        'passing_quality_score'
    ]
    df_filtered[cols_to_round] = df_filtered[cols_to_round].round(3)

    # Return sorted result with relevant columns
    result = df_filtered.sort_values('passing_quality_score', ascending=False)

    return result[[
        'Player', 'Squad', 'Comp', '90s',
        'passes_per_90', 'total_Cmp%', 'prog_passes_per_90',
        'key_passes_per_90', 'xA_per_90',
        'passing_accuracy_score', 'progression_score', 'chance_creation_score',
        'passing_quality_score'
    ]]
