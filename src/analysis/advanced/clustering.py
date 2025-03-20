from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config.settings import ANALYSIS_WEIGHTS, DEFAULT_ANALYSIS_PARAMS
from src.analysis.metrics import (
    normalize_metric,
    calculate_per_90_metrics,
    calculate_weighted_score,
    get_score_from_config
)

def cluster_player_profiles(
    df: pd.DataFrame,
    metrics: List[str],
    n_clusters: int = 5,
    position_group: Optional[str] = None,
    min_90s: float = DEFAULT_ANALYSIS_PARAMS["min_90s"]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cluster players based on their statistical profiles.

    Parameters:
    -----------
    df: DataFrame with player statistics
    metrics: List of metrics to use for clustering
    n_clusters: Number of clusters to form
    position_group: Filter to specific position group (e.g., "MF", "DF")
    min_90s: Minimum 90s played to be included

    Returns:
    --------
    Tuple containing:
    - DataFrame with cluster assignments
    - Dictionary with cluster centers and other information
    """
    # Filter data
    filtered_df = df[df["90s"] >= min_90s].copy()

    if position_group:
        filtered_df = filtered_df[filtered_df["Pos"].str.contains(position_group)]

    # Ensure all metrics exist
    for metric in metrics:
        if metric not in filtered_df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

    # Select features for clustering
    X = filtered_df[metrics].copy()

    # Handle NaN values - replace with column means
    X = X.fillna(X.mean())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal number of clusters if not provided
    if n_clusters is None:
        silhouette_scores = []
        K = range(2, min(11, len(filtered_df) // 10))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        n_clusters = K[np.argmax(silhouette_scores)]

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    filtered_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Get cluster centers and convert back to original scale
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=metrics
    )

    # Calculate cluster sizes
    cluster_sizes = filtered_df["cluster"].value_counts().to_dict()

    # Identify representative players for each cluster (closest to center)
    representative_players = {}
    for cluster_id in range(n_clusters):
        cluster_df = filtered_df[filtered_df["cluster"] == cluster_id]
        if len(cluster_df) == 0:
            continue

        cluster_scaled = scaler.transform(cluster_df[metrics].fillna(0))
        center = kmeans.cluster_centers_[cluster_id]

        # Calculate Euclidean distance to center for each player
        distances = np.sqrt(((cluster_scaled - center) ** 2).sum(axis=1))
        closest_idx = distances.argmin()

        representative_players[cluster_id] = {
            "player": cluster_df.iloc[closest_idx]["Player"],
            "team": cluster_df.iloc[closest_idx]["Squad"],
            "position": cluster_df.iloc[closest_idx]["Pos"]
        }

    # Return results
    cluster_info = {
        "centers": centers,
        "sizes": cluster_sizes,
        "representatives": representative_players,
        "feature_names": metrics
    }

    return filtered_df, cluster_info

def find_undervalued_players(
    analysis_df: pd.DataFrame,
    value_col: str = "Market Value (M€)",
    performance_col: str = "adjusted_versatility",
    age_penalty: bool = True,
    max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"]
) -> pd.DataFrame:
    """
    Identify potentially undervalued players by comparing performance metrics
    with market value.

    Parameters:
    -----------
    analysis_df: DataFrame with player analysis including performance metrics
    value_col: Column name containing player market values
    performance_col: Column name containing the performance metric to evaluate
    age_penalty: Whether to apply a penalty for older players
    max_age: Maximum age for player consideration

    Returns:
    --------
    DataFrame with value analysis
    """
    if value_col not in analysis_df.columns:
        raise ValueError(f"Market value column '{value_col}' not found in DataFrame")

    if performance_col not in analysis_df.columns:
        raise ValueError(f"Performance column '{performance_col}' not found in DataFrame")

    value_analysis = analysis_df.copy()

    # Apply any thresholds from configuration if available
    if "Age" in value_analysis.columns:
        value_analysis = value_analysis[value_analysis["Age"] <= max_age]

    # Normalize performance and value
    value_analysis["norm_performance"] = normalize_metric(value_analysis[performance_col])
    value_analysis["norm_value"] = normalize_metric(value_analysis[value_col])

    # Apply age penalty if desired
    if age_penalty and "Age" in value_analysis.columns:
        # Convert Age column to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(value_analysis["Age"]):
            value_analysis["Age_numeric"] = value_analysis["Age"].str.split("-").str[0].astype(float)
        else:
            value_analysis["Age_numeric"] = value_analysis["Age"]

        # Older players get penalized in value calculation (diminishing return curve)
        min_age = value_analysis["Age_numeric"].min()
        age_factor = 1 - normalize_metric(value_analysis["Age_numeric"] - min_age) * 0.3
        value_analysis["age_adjusted_performance"] = value_analysis["norm_performance"] * age_factor
        perf_col = "age_adjusted_performance"
    else:
        perf_col = "norm_performance"

    # Calculate value rating (performance relative to value)
    value_analysis["value_rating"] = value_analysis[perf_col] / (value_analysis["norm_value"] + 0.1)

    # Highlight extreme outliers (very high performance for low value)
    threshold = value_analysis["value_rating"].quantile(0.9)
    value_analysis["potential_bargain"] = value_analysis["value_rating"] > threshold

    return value_analysis.sort_values("value_rating", ascending=False)
