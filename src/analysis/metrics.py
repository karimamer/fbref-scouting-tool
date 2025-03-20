from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging

from config.settings import ANALYSIS_WEIGHTS

logger = logging.getLogger(__name__)

def normalize_metric(series: pd.Series, method: str = 'robust') -> pd.Series:
    """
    Normalize a metric to 0-1 scale using specified method.

    Args:
        series: Series to normalize
        method: Normalization method ('robust', 'minmax', 'zscore')

    Returns:
        Normalized series
    """
    if series.empty:
        return series

    if method == 'robust':
        # Use percentiles to be robust to outliers
        min_val = series.quantile(0.05)
        max_val = series.quantile(0.95)
        return (series - min_val) / (max_val - min_val)

    elif method == 'minmax':
        # Standard min-max normalization
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val)

    elif method == 'zscore':
        # Z-score normalization
        return (series - series.mean()) / series.std()

    else:
        logger.warning(f"Unknown normalization method: {method}, using robust scaling")
        min_val = series.quantile(0.05)
        max_val = series.quantile(0.95)
        return (series - min_val) / (max_val - min_val)


def calculate_per_90_metrics(
    df: pd.DataFrame,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Calculate per-90 minute metrics.

    Args:
        df: DataFrame with player statistics
        metrics: List of metrics to normalize to per-90

    Returns:
        DataFrame with added per-90 columns
    """
    if df.empty or "90s" not in df.columns:
        return df

    result = df.copy()

    for metric in metrics:
        if metric in result.columns:
            result[f"{metric}_90"] = result[metric] / result["90s"]

    return result


def calculate_weighted_score(
    df: pd.DataFrame,
    metrics: Dict[str, float],
    score_name: str,
    normalize_method: str = 'robust'
) -> pd.DataFrame:
    """
    Calculate a weighted score for players based on normalized metrics.

    Args:
        df: DataFrame with player statistics
        metrics: Dictionary mapping metric names to weights
        score_name: Name for the output score column
        normalize_method: Method for normalizing metrics

    Returns:
        DataFrame with added score column
    """
    if df.empty:
        return df

    result = df.copy()

    # First normalize each metric
    for metric_name in metrics.keys():
        if metric_name in result.columns:
            result[f"{metric_name}_norm"] = normalize_metric(
                result[metric_name], method=normalize_method
            )
        else:
            logger.warning(f"Metric '{metric_name}' not found in DataFrame for {score_name} calculation")

    # Calculate the weighted score
    result[score_name] = 0.0
    total_applied_weight = 0.0

    for metric_name, weight in metrics.items():
        norm_col = f"{metric_name}_norm"
        if norm_col in result.columns:
            result[score_name] += result[norm_col] * weight
            total_applied_weight += weight

    # Normalize by total applied weight if not all metrics were available
    if total_applied_weight > 0 and total_applied_weight != 1.0:
        result[score_name] = result[score_name] / total_applied_weight

    return result


def get_score_from_config(
    df: pd.DataFrame,
    score_type: str,
    custom_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Calculate a player score using weights from configuration.

    Args:
        df: DataFrame with player statistics
        score_type: Type of score to calculate (from ANALYSIS_WEIGHTS)
        custom_weights: Optional custom weights to override configuration

    Returns:
        DataFrame with added score
    """
    if score_type not in ANALYSIS_WEIGHTS and not custom_weights:
        logger.error(f"Score type '{score_type}' not found in configuration and no custom weights provided")
        return df

    # Use custom weights if provided, otherwise use config
    weights = custom_weights if custom_weights else ANALYSIS_WEIGHTS[score_type]
    score_name = f"{score_type}_score"

    return calculate_weighted_score(df, weights, score_name)


def combine_scores(
    dfs: Dict[str, pd.DataFrame],
    score_types: List[str],
    weights: Optional[Dict[str, float]] = None,
    final_score_name: str = "combined_score"
) -> pd.DataFrame:
    """
    Combine multiple scores from different DataFrames into a single score.

    Args:
        dfs: Dictionary of DataFrames with calculated scores
        score_types: List of score types to combine
        weights: Optional weights for each score type
        final_score_name: Name for the combined score column

    Returns:
        DataFrame with combined score
    """
    if not dfs or not score_types:
        return pd.DataFrame()

    # Start with a base DataFrame containing player identifiers
    base_cols = ["Player", "Squad", "Age", "Pos"]
    result = None

    for df_name, df in dfs.items():
        if df.empty:
            continue

        if result is None:
            # Initialize with the first non-empty DataFrame
            result = df[base_cols].copy()

        score_col = f"{df_name}_score"
        if score_col in df.columns and df_name in score_types:
            # Merge the score into the result DataFrame
            result = result.merge(
                df[["Player", score_col]],
                on="Player",
                how="left"
            )

    if result is None or result.empty:
        return pd.DataFrame()

    # Calculate the combined score using weights
    result[final_score_name] = 0.0
    total_weight = 0.0

    for score_type in score_types:
        score_col = f"{score_type}_score"
        if score_col in result.columns:
            weight = weights.get(score_type, 1.0) if weights else 1.0
            result[final_score_name] += result[score_col] * weight
            total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        result[final_score_name] = result[final_score_name] / total_weight

    return result.sort_values(final_score_name, ascending=False)
