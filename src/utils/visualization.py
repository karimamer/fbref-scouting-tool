import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union

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

    Parameters:
    -----------
    df: DataFrame containing player data
    metrics: List of metrics to include in the comparison
    player_column: Column name that contains player names
    title: Title for the chart
    max_players: Maximum number of players to include
    output_file: If provided, save the figure to this path
    normalize: Whether to normalize metrics to 0-1 scale

    Returns:
    --------
    matplotlib Figure object
    """
    # Select top players and required columns
    players_df = df.head(max_players).copy()

    # Ensure all metrics exist
    missing_metrics = [m for m in metrics if m not in players_df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metrics in dataframe: {missing_metrics}")

    # Normalize metrics if requested
    if normalize:
        for metric in metrics:
            min_val = players_df[metric].min()
            max_val = players_df[metric].max()
            if max_val > min_val:
                players_df[f"{metric}_norm"] = (players_df[metric] - min_val) / (max_val - min_val)
            else:
                players_df[f"{metric}_norm"] = 0.5  # Default if all values are the same
        plotting_metrics = [f"{m}_norm" for m in metrics]
    else:
        plotting_metrics = metrics

    # Number of metrics
    num_metrics = len(plotting_metrics)

    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(players_df)))

    # Plot each player
    for i, (_, player) in enumerate(players_df.iterrows()):
        values = [player[m] for m in plotting_metrics]
        values += values[:1]  # Close the circle

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=player[player_column])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Set labels
    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    plt.xticks(angles[:-1], metric_labels, size=12)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Add title
    plt.title(title, size=15)

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

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

    Parameters:
    -----------
    df: DataFrame containing player data
    x_metric: Metric to plot on x-axis
    y_metric: Metric to plot on y-axis
    color_by: Column to use for point coloring
    size_by: Column to use for point sizing
    player_column: Column name that contains player names
    title: Title for the chart
    labeled_players: Number of players to label in the plot
    output_file: If provided, save the figure to this path

    Returns:
    --------
    matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare plot parameters
    plot_kwargs = {
        'x': x_metric,
        'y': y_metric,
        'data': df,
        'alpha': 0.7
    }

    # Add color grouping if specified
    if color_by and color_by in df.columns:
        plot_kwargs['hue'] = color_by

    # Add size variation if specified
    if size_by and size_by in df.columns:
        plot_kwargs['size'] = size_by
        plot_kwargs['sizes'] = (20, 200)

    # Create scatter plot
    sns.scatterplot(**plot_kwargs)

    # Add player labels for top players
    importance_metric = y_metric  # Default to y-axis metric for importance

    for _, player in df.sort_values(importance_metric, ascending=False).head(labeled_players).iterrows():
        plt.text(
            player[x_metric] + (df[x_metric].max() - df[x_metric].min()) * 0.01,
            player[y_metric] + (df[y_metric].max() - df[y_metric].min()) * 0.01,
            player[player_column],
            fontsize=9
        )

    # Format chart
    plt.title(title, fontsize=16)
    plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

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

    Parameters:
    -----------
    df: DataFrame containing player data
    metrics: List of metrics to include in the comparison
    player_column: Column name that contains player names
    title: Title for the chart
    max_players: Maximum number of players to include
    sort_by: Metric to sort players by
    output_file: If provided, save the figure to this path
    horizontal: Whether to create horizontal bar chart
    stacked: Whether to create stacked bar chart

    Returns:
    --------
    matplotlib Figure object
    """
    # Select data
    players_df = df.head(max_players).copy()

    # Sort if requested
    if sort_by:
        if sort_by in players_df.columns:
            players_df = players_df.sort_values(sort_by, ascending=False)
        else:
            print(f"Warning: Sort column '{sort_by}' not found in dataframe")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8) if horizontal else (10, 10))

    # Determine plot orientation
    plot_func = players_df.plot.barh if horizontal else players_df.plot.bar

    # Create bar chart
    plot_func(
        x=player_column,
        y=metrics,
        ax=ax,
        stacked=stacked,
        width=0.8,
        alpha=0.8
    )

    # Format chart
    plt.title(title, fontsize=16)
    if horizontal:
        plt.xlabel(', '.join([m.replace('_', ' ').title() for m in metrics]), fontsize=12)
        plt.ylabel("Players", fontsize=12)
    else:
        plt.xlabel("Players", fontsize=12)
        plt.ylabel(', '.join([m.replace('_', ' ').title() for m in metrics]), fontsize=12)

    plt.legend(title='Metrics')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x' if horizontal else 'y')
    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

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

    Parameters:
    -----------
    df: DataFrame containing player data
    metrics: List of metrics to include in the heatmap
    player_column: Column name that contains player names
    title: Title for the chart
    max_players: Maximum number of players to include
    output_file: If provided, save the figure to this path
    normalize: Whether to normalize metrics for better comparison
    cmap: Colormap to use

    Returns:
    --------
    matplotlib Figure object
    """
    # Select data
    players_df = df.head(max_players).copy()

    # Set player as index
    players_df = players_df.set_index(player_column)

    # Select only the metrics we want to display
    metrics_df = players_df[metrics].copy()

    # Normalize if requested
    if normalize:
        for col in metrics_df.columns:
            metrics_df[col] = (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        metrics_df,
        annot=True,
        cmap=cmap,
        fmt=".2f" if normalize else ".1f",
        linewidths=0.5,
        ax=ax
    )

    # Format chart
    plt.title(title, fontsize=16)
    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

def create_dashboard(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "visualizations",
    prefix: str = ""
) -> List[str]:
    """
    Create a full dashboard of visualizations from analysis results.

    Parameters:
    -----------
    results: Dictionary of analysis results
    output_dir: Directory to save visualizations
    prefix: Prefix for output filenames

    Returns:
    --------
    List of created visualization filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    created_files = []

    # 1. Versatility radar chart
    if "versatile_players" in results and not results["versatile_players"].empty:
        versatile_df = results["versatile_players"].head(5)
        metrics = ["passing_score", "possession_score", "defensive_score"]
        if "shooting_score" in versatile_df.columns:
            metrics.append("shooting_score")

        output_file = os.path.join(output_dir, f"{prefix}versatility_radar.png")
        create_radar_comparison(
            versatile_df,
            metrics=metrics,
            title="Top Players by Versatility",
            output_file=output_file
        )
        created_files.append(output_file)

    # 2. Progressive actions comparison
    prog_metrics = ["overall_progressors", "top_carriers", "top_passers"]
    for metric in prog_metrics:
        if metric in results and not results[metric].empty:
            df = results[metric].head(10)
            if "total_progression_score" in df.columns:
                sort_col = "total_progression_score"
            else:
                sort_col = None

            output_file = os.path.join(output_dir, f"{prefix}{metric}_comparison.png")
            create_bar_comparison(
                df,
                metrics=["carrying_progression_score", "passing_progression_score", "receiving_progression_score"]
                if all(col in df.columns for col in ["carrying_progression_score", "passing_progression_score", "receiving_progression_score"])
                else ["PrgC", "PrgP", "PrgR"],
                title=f"Top {metric.replace('_', ' ').title()}",
                sort_by=sort_col,
                output_file=output_file,
                stacked=True
            )
            created_files.append(output_file)

    # 3. Possession Impact scatter
    if "possession_impact" in results and not results["possession_impact"].empty:
        xpi_df = results["possession_impact"].copy()

        # Create position column for coloring
        if "Pos" in xpi_df.columns:
            xpi_df["Position"] = xpi_df["Pos"].apply(
                lambda x: "Defender" if "DF" in x
                else "Midfielder" if "MF" in x
                else "Forward" if "FW" in x
                else "Other"
            )
            color_by = "Position"
        else:
            color_by = None

        # Choose size variable if available
        size_by = "touches_90" if "touches_90" in xpi_df.columns else None

        output_file = os.path.join(output_dir, f"{prefix}possession_impact.png")
        create_scatter_comparison(
            xpi_df,
            x_metric="90s",
            y_metric="xPI",
            color_by=color_by,
            size_by=size_by,
            title="Expected Possession Impact (xPI)",
            output_file=output_file
        )
        created_files.append(output_file)

    # 4. Midfielder clusters
    if "midfielder_clusters" in results and "cluster" in results["midfielder_clusters"].columns:
        mf_df = results["midfielder_clusters"].copy()

        if len(mf_df) >= 10 and "PrgP" in mf_df.columns and "PrgC" in mf_df.columns:
            output_file = os.path.join(output_dir, f"{prefix}midfielder_clusters.png")
            create_scatter_comparison(
                mf_df,
                x_metric="PrgP",
                y_metric="PrgC",
                color_by="cluster",
                title="Midfielder Clusters: Progressive Passes vs Carries",
                output_file=output_file
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
                output_file=output_file
            )
            created_files.append(output_file)

    return created_files
