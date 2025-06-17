import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseVisualizer:
    """Base class for all visualization functionality."""
    
    DEFAULT_STYLE = {
        'figure.figsize': (12, 8),
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300
    }
    
    DEFAULT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def __init__(self, style_config: Optional[Dict] = None):
        """Initialize visualizer with style configuration."""
        self.style_config = {**self.DEFAULT_STYLE, **(style_config or {})}
        self._apply_style()
    
    def _apply_style(self):
        """Apply matplotlib style configuration."""
        plt.rcParams.update(self.style_config)
    
    def save_figure(self, fig: plt.Figure, output_file: str, **kwargs) -> None:
        """Save figure with consistent parameters."""
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_kwargs.update(kwargs)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.savefig(output_file, **save_kwargs)
        logger.info(f"Saved visualization to {output_file}")
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Validate dataframe has required columns."""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return df.copy()
    
    def normalize_metrics(self, df: pd.DataFrame, metrics: List[str], invert_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize metrics to 0-1 scale."""
        result_df = df.copy()
        invert_metrics = invert_metrics or []
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            if max_val == min_val:
                result_df[f"{metric}_norm"] = 0.5
            else:
                normalized = (df[metric] - min_val) / (max_val - min_val)
                if metric in invert_metrics:
                    normalized = 1 - normalized
                result_df[f"{metric}_norm"] = normalized
        
        return result_df
    
    def format_metric_name(self, metric: str) -> str:
        """Format metric names for display."""
        return metric.replace('_', ' ').replace('/', ' per ').title()
    
    def add_grid(self, ax: plt.Axes, alpha: float = 0.3, **kwargs) -> None:
        """Add consistent grid styling."""
        grid_params = {'alpha': alpha, 'linestyle': '--'}
        grid_params.update(kwargs)
        ax.grid(True, **grid_params)


def create_radar_chart(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Comparison",
    max_players: int = 5,
    output_file: Optional[str] = None,
    normalize: bool = True,
    invert_metrics: Optional[List[str]] = None,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a radar chart comparing multiple players across different metrics.
    
    Args:
        df: DataFrame containing player data
        metrics: List of metrics to include in the comparison
        player_column: Column name that contains player names
        title: Title for the chart
        max_players: Maximum number of players to include
        output_file: If provided, save the figure to this path
        normalize: Whether to normalize metrics to 0-1 scale
        invert_metrics: List of metrics where lower values are better
        visualizer: BaseVisualizer instance for styling
    
    Returns:
        matplotlib Figure object
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate and prepare data
    required_cols = [player_column] + metrics
    df = viz.validate_dataframe(df, required_cols)
    players_df = df.head(max_players).copy()
    
    # Normalize metrics if requested
    if normalize:
        players_df = viz.normalize_metrics(players_df, metrics, invert_metrics)
        plotting_metrics = [f"{m}_norm" for m in metrics]
    else:
        plotting_metrics = metrics
    
    # Create radar chart
    num_metrics = len(plotting_metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each player
    for i, (_, player) in enumerate(players_df.iterrows()):
        values = [player[m] for m in plotting_metrics]
        values += values[:1]  # Close the circle
        
        color = viz.DEFAULT_COLORS[i % len(viz.DEFAULT_COLORS)]
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=player[player_column])
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Format chart
    metric_labels = [viz.format_metric_name(m) for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=12)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.set_title(title, size=15, pad=20)
    
    # Save if requested
    if output_file:
        viz.save_figure(fig, output_file)
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    player_column: str = "Player",
    title: str = "Player Comparison",
    labeled_players: int = 10,
    output_file: Optional[str] = None,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a scatter plot comparing players on two metrics.
    
    Args:
        df: DataFrame containing player data
        x_metric: Metric to plot on x-axis
        y_metric: Metric to plot on y-axis
        color_by: Column to use for point coloring
        size_by: Column to use for point sizing
        player_column: Column name that contains player names
        title: Title for the chart
        labeled_players: Number of players to label in the plot
        output_file: If provided, save the figure to this path
        visualizer: BaseVisualizer instance for styling
    
    Returns:
        matplotlib Figure object
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate data
    required_cols = [player_column, x_metric, y_metric]
    if color_by:
        required_cols.append(color_by)
    if size_by:
        required_cols.append(size_by)
    
    df = viz.validate_dataframe(df, required_cols)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare plot parameters
    plot_kwargs = {
        'x': x_metric,
        'y': y_metric,
        'data': df,
        'alpha': 0.7,
        'ax': ax
    }
    
    if color_by:
        plot_kwargs['hue'] = color_by
    if size_by:
        plot_kwargs['size'] = size_by
        plot_kwargs['sizes'] = (20, 200)
    
    # Create scatter plot
    sns.scatterplot(**plot_kwargs)
    
    # Add player labels for top players
    importance_metric = y_metric  # Default to y-axis metric for importance
    top_players = df.sort_values(importance_metric, ascending=False).head(labeled_players)
    
    for _, player in top_players.iterrows():
        ax.text(
            player[x_metric] + (df[x_metric].max() - df[x_metric].min()) * 0.01,
            player[y_metric] + (df[y_metric].max() - df[y_metric].min()) * 0.01,
            player[player_column],
            fontsize=9
        )
    
    # Format chart
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(viz.format_metric_name(x_metric), fontsize=12)
    ax.set_ylabel(viz.format_metric_name(y_metric), fontsize=12)
    viz.add_grid(ax)
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        viz.save_figure(fig, output_file)
    
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Comparison",
    max_players: int = 10,
    sort_by: Optional[str] = None,
    output_file: Optional[str] = None,
    horizontal: bool = True,
    stacked: bool = False,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a bar chart comparing players across metrics.
    
    Args:
        df: DataFrame containing player data
        metrics: List of metrics to include in the comparison
        player_column: Column name that contains player names
        title: Title for the chart
        max_players: Maximum number of players to include
        sort_by: Metric to sort players by
        output_file: If provided, save the figure to this path
        horizontal: Whether to create horizontal bar chart
        stacked: Whether to create stacked bar chart
        visualizer: BaseVisualizer instance for styling
    
    Returns:
        matplotlib Figure object
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate and prepare data
    required_cols = [player_column] + metrics
    df = viz.validate_dataframe(df, required_cols)
    players_df = df.head(max_players).copy()
    
    # Sort if requested
    if sort_by and sort_by in players_df.columns:
        players_df = players_df.sort_values(sort_by, ascending=False)
    elif sort_by:
        logger.warning(f"Sort column '{sort_by}' not found in dataframe")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8) if horizontal else (10, 10))
    
    # Create bar chart
    plot_func = players_df.plot.barh if horizontal else players_df.plot.bar
    plot_func(
        x=player_column,
        y=metrics,
        ax=ax,
        stacked=stacked,
        width=0.8,
        alpha=0.8
    )
    
    # Format chart
    ax.set_title(title, fontsize=16)
    metric_labels = ', '.join([viz.format_metric_name(m) for m in metrics])
    
    if horizontal:
        ax.set_xlabel(metric_labels, fontsize=12)
        ax.set_ylabel("Players", fontsize=12)
        viz.add_grid(ax, axis='x')
    else:
        ax.set_xlabel("Players", fontsize=12)
        ax.set_ylabel(metric_labels, fontsize=12)
        viz.add_grid(ax, axis='y')
    
    ax.legend(title='Metrics', labels=[viz.format_metric_name(m) for m in metrics])
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        viz.save_figure(fig, output_file)
    
    return fig


def create_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Metrics Heatmap",
    max_players: int = 15,
    output_file: Optional[str] = None,
    normalize: bool = True,
    cmap: str = "viridis",
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a heatmap of multiple metrics for multiple players.
    
    Args:
        df: DataFrame containing player data
        metrics: List of metrics to include in the heatmap
        player_column: Column name that contains player names
        title: Title for the chart
        max_players: Maximum number of players to include
        output_file: If provided, save the figure to this path
        normalize: Whether to normalize metrics for better comparison
        cmap: Colormap to use
        visualizer: BaseVisualizer instance for styling
    
    Returns:
        matplotlib Figure object
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate and prepare data
    required_cols = [player_column] + metrics
    df = viz.validate_dataframe(df, required_cols)
    players_df = df.head(max_players).copy()
    
    # Set player as index and select metrics
    players_df = players_df.set_index(player_column)
    metrics_df = players_df[metrics].copy()
    
    # Normalize if requested
    if normalize:
        for col in metrics_df.columns:
            col_min, col_max = metrics_df[col].min(), metrics_df[col].max()
            if col_max != col_min:
                metrics_df[col] = (metrics_df[col] - col_min) / (col_max - col_min)
            else:
                metrics_df[col] = 0.5
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        metrics_df,
        annot=True,
        cmap=cmap,
        fmt=".2f" if normalize else ".1f",
        linewidths=0.5,
        ax=ax,
        xticklabels=[viz.format_metric_name(m) for m in metrics]
    )
    
    # Format chart
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        viz.save_figure(fig, output_file)
    
    return fig