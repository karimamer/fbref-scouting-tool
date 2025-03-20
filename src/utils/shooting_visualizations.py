import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union

def create_finishing_scatter(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 20,
    title: str = "Finishing Skill vs. Expected Goals",
    labeled_players: int = 10
) -> plt.Figure:
    """
    Create a scatter plot showing goals vs. xG with finishing skill highlighted.

    Args:
        df: DataFrame with shooting statistics
        output_file: Path to save the output file
        min_shots: Minimum shots filter
        title: Plot title
        labeled_players: Number of players to label in the plot

    Returns:
        matplotlib Figure object
    """
    # Filter data
    plot_df = df[df["Sh"] >= min_shots].copy()

    # Create goals vs xG metrics if they don't exist
    if "G-xG" not in plot_df.columns and "Gls" in plot_df.columns and "xG" in plot_df.columns:
        plot_df["G-xG"] = plot_df["Gls"] - plot_df["xG"]

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    scatter = sns.scatterplot(
        data=plot_df,
        x="xG",
        y="Gls",
        size="Sh",
        sizes=(20, 200),
        hue="G-xG",
        palette="RdBu_r",
        alpha=0.7
    )

    # Add reference line (y=x, where goals = xG)
    max_val = max(plot_df["xG"].max(), plot_df["Gls"].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

    # Add annotations for top finishers
    top_performers = plot_df.sort_values("G-xG", ascending=False).head(labeled_players)

    for _, row in top_performers.iterrows():
        plt.annotate(
            row["Player"],
            xy=(row["xG"], row["Gls"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    # Add annotations for bottom finishers
    bottom_performers = plot_df.sort_values("G-xG").head(labeled_players)

    for _, row in bottom_performers.iterrows():
        plt.annotate(
            row["Player"],
            xy=(row["xG"], row["Gls"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    # Set plot attributes
    plt.title(title, fontsize=14)
    plt.xlabel("Expected Goals (xG)", fontsize=12)
    plt.ylabel("Goals", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add a colorbar legend
    norm = plt.Normalize(plot_df["G-xG"].min(), plot_df["G-xG"].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Goals - Expected Goals (G-xG)", fontsize=10)

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return plt.gcf()

def create_shot_quality_distribution(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 20,
    top_n: int = 15
) -> plt.Figure:
    """
    Create a bar chart showing npxG per shot for top players.

    Args:
        df: DataFrame with shooting statistics
        output_file: Path to save the output file
        min_shots: Minimum shots filter
        top_n: Number of top players to show

    Returns:
        matplotlib Figure object
    """
    # Filter data
    plot_df = df[df["Sh"] >= min_shots].copy()

    # Calculate npxG per shot if it doesn't exist
    if "npxG_per_shot" not in plot_df.columns:
        plot_df["npxG_per_shot"] = plot_df["npxG"] / plot_df["Sh"]

    # Sort and get top players
    plot_df = plot_df.sort_values("npxG_per_shot", ascending=False).head(top_n)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Create bar chart
    ax = sns.barplot(
        y="Player",
        x="npxG_per_shot",
        data=plot_df,
        palette="viridis"
    )

    # Add shot count as text
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(
            row.npxG_per_shot + 0.005,
            i,
            f"Shots: {int(row.Sh)}",
            va='center'
        )

    # Set plot attributes
    plt.title("Players with Highest Expected Goals per Shot", fontsize=14)
    plt.xlabel("Non-Penalty xG per Shot", fontsize=12)
    plt.ylabel("Player", fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return plt.gcf()

def create_shooting_profile_radar(
    df: pd.DataFrame,
    players: List[str],
    output_file: Optional[str] = None,
    min_90s: float = 5
) -> plt.Figure:
    """
    Create a radar chart comparing shooting profiles of selected players.

    Args:
        df: DataFrame with shooting statistics
        players: List of player names to compare
        output_file: Path to save the output file
        min_90s: Minimum 90s played filter

    Returns:
        matplotlib Figure object
    """
    # Filter data for selected players and minimum minutes
    plot_df = df[(df["Player"].isin(players)) & (df["90s"] >= min_90s)].copy()

    if len(plot_df) == 0:
        raise ValueError("No matching players found with the specified criteria")

    # Metrics to compare
    metrics = ["Sh/90", "SoT%", "G/Sh", "Dist", "npxG/Sh", "G-xG"]

    # Ensure all metrics exist or calculate them
    if "Sh/90" not in plot_df.columns and "Sh" in plot_df.columns:
        plot_df["Sh/90"] = plot_df["Sh"] / plot_df["90s"]

    if "G/Sh" not in plot_df.columns and "Gls" in plot_df.columns:
        plot_df["G/Sh"] = plot_df["Gls"] / plot_df["Sh"]

    if "npxG/Sh" not in plot_df.columns and "npxG" in plot_df.columns:
        plot_df["npxG/Sh"] = plot_df["npxG"] / plot_df["Sh"]

    # Check which metrics are available
    available_metrics = [m for m in metrics if m in plot_df.columns]

    if len(available_metrics) < 3:
        raise ValueError("Not enough metrics available for radar chart")

    # Normalize each metric for comparison
    for metric in available_metrics:
        max_val = plot_df[metric].max()
        min_val = plot_df[metric].min()
        # Handle special case for Dist where lower is better
        if metric == "Dist":
            if max_val != min_val:
                plot_df[f"{metric}_norm"] = 1 - ((plot_df[metric] - min_val) / (max_val - min_val))
            else:
                plot_df[f"{metric}_norm"] = 0.5
        else:
            if max_val != min_val:
                plot_df[f"{metric}_norm"] = (plot_df[metric] - min_val) / (max_val - min_val)
            else:
                plot_df[f"{metric}_norm"] = 0.5

    # Set up the radar chart
    num_metrics = len(available_metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot each player
    for i, player_name in enumerate(players):
        player_data = plot_df[plot_df["Player"] == player_name]

        if len(player_data) == 0:
            continue

        values = [player_data[f"{m}_norm"].values[0] for m in available_metrics]
        values += values[:1]  # Close the circle

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=player_name)
        ax.fill(angles, values, alpha=0.1)

    # Set chart properties
    metric_labels = [m.replace("_", " ") for m in available_metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Shooting Profile Comparison", size=15)

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig

def create_shot_distance_histogram(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 20,
    overlay_top_scorers: bool = True
) -> plt.Figure:
    """
    Create a histogram of shot distances with top scorers highlighted.

    Args:
        df: DataFrame with shooting statistics
        output_file: Path to save the output file
        min_shots: Minimum shots filter
        overlay_top_scorers: Whether to overlay distribution for top scorers

    Returns:
        matplotlib Figure object
    """
    # Filter data
    plot_df = df[df["Sh"] >= min_shots].copy()

    if "Dist" not in plot_df.columns:
        raise ValueError("Shot distance data not available")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Main distribution
    sns.histplot(
        data=plot_df,
        x="Dist",
        kde=True,
        color="skyblue",
        alpha=0.6,
        label="All Players"
    )

    # Overlay top scorers if requested
    if overlay_top_scorers:
        top_scorers = plot_df.sort_values("Gls", ascending=False).head(10)
        sns.histplot(
            data=top_scorers,
            x="Dist",
            kde=True,
            color="red",
            alpha=0.4,
            label="Top 10 Scorers"
        )

    # Set plot attributes
    plt.title("Distribution of Shot Distances", fontsize=14)
    plt.xlabel("Shot Distance (yards)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return plt.gcf()

def create_shooting_metrics_dashboard(
    shooting_df: pd.DataFrame,
    output_dir: str = "visualizations/shooting",
    min_shots: int = 20,
    min_90s: float = 5
) -> List[str]:
    """
    Create a comprehensive dashboard of shooting visualizations.

    Args:
        shooting_df: DataFrame with shooting statistics
        output_dir: Directory to save visualizations
        min_shots: Minimum shots filter
        min_90s: Minimum 90s played filter

    Returns:
        List of created visualization file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    created_files = []

    # 1. Finishing ability scatter plot
    try:
        output_file = os.path.join(output_dir, "finishing_skill.png")
        create_finishing_scatter(
            shooting_df,
            output_file=output_file,
            min_shots=min_shots,
            title="Player Finishing Skill: Goals vs. Expected Goals"
        )
        created_files.append(output_file)
    except Exception as e:
        print(f"Error creating finishing scatter: {str(e)}")

    # 2. Shot quality distribution
    try:
        output_file = os.path.join(output_dir, "shot_quality.png")
        create_shot_quality_distribution(
            shooting_df,
            output_file=output_file,
            min_shots=min_shots
        )
        created_files.append(output_file)
    except Exception as e:
        print(f"Error creating shot quality distribution: {str(e)}")

    # 3. Shot distance histogram
    try:
        output_file = os.path.join(output_dir, "shot_distance.png")
        create_shot_distance_histogram(
            shooting_df,
            output_file=output_file,
            min_shots=min_shots
        )
        created_files.append(output_file)
    except Exception as e:
        print(f"Error creating shot distance histogram: {str(e)}")

    # 4. Radar comparison of top scorers
    try:
        # Get top 5 goal scorers
        top_scorers = shooting_df.sort_values("Gls", ascending=False).head(5)["Player"].tolist()

        if len(top_scorers) >= 3:  # Need at least 3 players for a meaningful radar
            output_file = os.path.join(output_dir, "top_scorers_radar.png")
            create_shooting_profile_radar(
                shooting_df,
                players=top_scorers[:5],  # Limit to 5 players
                output_file=output_file,
                min_90s=min_90s
            )
            created_files.append(output_file)
    except Exception as e:
        print(f"Error creating shooting profile radar: {str(e)}")

    return created_files
