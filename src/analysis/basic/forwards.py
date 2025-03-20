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
