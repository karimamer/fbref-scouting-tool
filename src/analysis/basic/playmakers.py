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
