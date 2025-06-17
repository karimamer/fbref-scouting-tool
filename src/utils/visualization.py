import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from .viz_base import BaseVisualizer, create_radar_chart, create_scatter_plot, create_bar_chart, create_heatmap

def create_radar_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Comparison",
    max_players: int = 5,
    output_file: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Create a radar chart comparing multiple players across different metrics.
    
    Deprecated: Use viz_base.create_radar_chart instead.
    """
    return create_radar_chart(
        df=df,
        metrics=metrics,
        player_column=player_column,
        title=title,
        max_players=max_players,
        output_file=output_file,
        normalize=normalize
    )

def create_scatter_comparison(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    player_column: str = "Player",
    title: str = "Player Comparison",
    labeled_players: int = 10,
    output_file: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot comparing players on two metrics.
    
    Deprecated: Use viz_base.create_scatter_plot instead.
    """
    return create_scatter_plot(
        df=df,
        x_metric=x_metric,
        y_metric=y_metric,
        color_by=color_by,
        size_by=size_by,
        player_column=player_column,
        title=title,
        labeled_players=labeled_players,
        output_file=output_file
    )

def create_bar_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Comparison",
    max_players: int = 10,
    sort_by: Optional[str] = None,
    output_file: Optional[str] = None,
    horizontal: bool = True,
    stacked: bool = False
) -> plt.Figure:
    """
    Create a bar chart comparing players across metrics.
    
    Deprecated: Use viz_base.create_bar_chart instead.
    """
    return create_bar_chart(
        df=df,
        metrics=metrics,
        player_column=player_column,
        title=title,
        max_players=max_players,
        sort_by=sort_by,
        output_file=output_file,
        horizontal=horizontal,
        stacked=stacked
    )

def create_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Metrics Heatmap",
    max_players: int = 15,
    output_file: Optional[str] = None,
    normalize: bool = True,
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Create a heatmap of multiple metrics for multiple players.
    
    Deprecated: Use viz_base.create_heatmap instead.
    """
    return create_heatmap(
        df=df,
        metrics=metrics,
        player_column=player_column,
        title=title,
        max_players=max_players,
        output_file=output_file,
        normalize=normalize,
        cmap=cmap
    )

def create_dashboard(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "visualizations",
    prefix: str = ""
) -> List[str]:
    """
    Create a full dashboard of visualizations from analysis results.
    """
    viz = BaseVisualizer()
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # 1. Versatility radar chart
    if "versatile_players" in results and not results["versatile_players"].empty:
        versatile_df = results["versatile_players"].head(5)
        metrics = ["passing_score", "possession_score", "defensive_score"]
        if "shooting_score" in versatile_df.columns:
            metrics.append("shooting_score")

        output_file = os.path.join(output_dir, f"{prefix}versatility_radar.png")
        create_radar_chart(
            versatile_df,
            metrics=metrics,
            title="Top Players by Versatility",
            output_file=output_file,
            visualizer=viz
        )
        created_files.append(output_file)

    # 2. Progressive actions comparison
    prog_metrics = ["overall_progressors", "top_carriers", "top_passers"]
    for metric in prog_metrics:
        if metric in results and not results[metric].empty:
            df = results[metric].head(10)
            sort_col = "total_progression_score" if "total_progression_score" in df.columns else None

            output_file = os.path.join(output_dir, f"{prefix}{metric}_comparison.png")
            
            # Select appropriate metrics based on availability
            if all(col in df.columns for col in ["carrying_progression_score", "passing_progression_score", "receiving_progression_score"]):
                bar_metrics = ["carrying_progression_score", "passing_progression_score", "receiving_progression_score"]
            else:
                bar_metrics = [col for col in ["PrgC", "PrgP", "PrgR"] if col in df.columns]
            
            if bar_metrics:
                create_bar_chart(
                    df,
                    metrics=bar_metrics,
                    title=f"Top {metric.replace('_', ' ').title()}",
                    sort_by=sort_col,
                    output_file=output_file,
                    stacked=True,
                    visualizer=viz
                )
                created_files.append(output_file)

    # 3. Possession Impact scatter
    if "possession_impact" in results and not results["possession_impact"].empty:
        xpi_df = results["possession_impact"].copy()

        # Create position column for coloring
        color_by = None
        if "Pos" in xpi_df.columns:
            xpi_df["Position"] = xpi_df["Pos"].apply(
                lambda x: "Defender" if "DF" in x
                else "Midfielder" if "MF" in x
                else "Forward" if "FW" in x
                else "Other"
            )
            color_by = "Position"

        size_by = "touches_90" if "touches_90" in xpi_df.columns else None

        if "90s" in xpi_df.columns and "xPI" in xpi_df.columns:
            output_file = os.path.join(output_dir, f"{prefix}possession_impact.png")
            create_scatter_plot(
                xpi_df,
                x_metric="90s",
                y_metric="xPI",
                color_by=color_by,
                size_by=size_by,
                title="Expected Possession Impact (xPI)",
                output_file=output_file,
                visualizer=viz
            )
            created_files.append(output_file)

    # 4. Midfielder clusters
    if "midfielder_clusters" in results and "cluster" in results["midfielder_clusters"].columns:
        mf_df = results["midfielder_clusters"].copy()

        if len(mf_df) >= 10 and "PrgP" in mf_df.columns and "PrgC" in mf_df.columns:
            output_file = os.path.join(output_dir, f"{prefix}midfielder_clusters.png")
            create_scatter_plot(
                mf_df,
                x_metric="PrgP",
                y_metric="PrgC",
                color_by="cluster",
                title="Midfielder Clusters: Progressive Passes vs Carries",
                output_file=output_file,
                visualizer=viz
            )
            created_files.append(output_file)

    # 5. Heatmap of top versatile players
    if "versatile_players" in results and not results["versatile_players"].empty:
        versatile_df = results["versatile_players"].head(15)

        # Select metrics for heatmap
        heatmap_metrics = [
            col for col in ["passing_score", "possession_score", "defensive_score",
                           "shooting_score", "versatility_score", "adjusted_versatility"]
            if col in versatile_df.columns
        ]

        if len(heatmap_metrics) >= 3:
            output_file = os.path.join(output_dir, f"{prefix}versatility_heatmap.png")
            create_heatmap(
                versatile_df,
                metrics=heatmap_metrics,
                title="Top Players Versatility Breakdown",
                output_file=output_file,
                visualizer=viz
            )
            created_files.append(output_file)

    return created_files
