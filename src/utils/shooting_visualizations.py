import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from .viz_base import BaseVisualizer, create_radar_chart
import logging

logger = logging.getLogger(__name__)

def create_finishing_scatter(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 20,
    title: str = "Finishing Skill vs. Expected Goals",
    labeled_players: int = 10,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a scatter plot showing goals vs. xG with finishing skill highlighted.
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate and filter data
    required_cols = ["Player", "Sh", "Gls", "xG"]
    df = viz.validate_dataframe(df, required_cols)
    plot_df = df[df["Sh"] >= min_shots].copy()
    
    if len(plot_df) == 0:
        raise ValueError(f"No players found with at least {min_shots} shots")

    # Create G-xG metric if it doesn't exist
    if "G-xG" not in plot_df.columns:
        plot_df["G-xG"] = plot_df["Gls"] - plot_df["xG"]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot
    scatter = sns.scatterplot(
        data=plot_df,
        x="xG",
        y="Gls",
        size="Sh",
        sizes=(20, 200),
        hue="G-xG",
        palette="RdBu_r",
        alpha=0.7,
        ax=ax
    )

    # Add reference line (y=x, where goals = xG)
    max_val = max(plot_df["xG"].max(), plot_df["Gls"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="Expected line (G=xG)")

    # Add annotations for top and bottom performers
    top_performers = plot_df.sort_values("G-xG", ascending=False).head(labeled_players//2)
    bottom_performers = plot_df.sort_values("G-xG").head(labeled_players//2)
    
    performers = pd.concat([top_performers, bottom_performers])
    
    for _, row in performers.iterrows():
        ax.annotate(
            row["Player"],
            xy=(row["xG"], row["Gls"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8
        )

    # Format chart
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Expected Goals (xG)", fontsize=12)
    ax.set_ylabel("Goals", fontsize=12)
    viz.add_grid(ax)

    # Add colorbar
    norm = plt.Normalize(plot_df["G-xG"].min(), plot_df["G-xG"].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Goals - Expected Goals (G-xG)", fontsize=10)
    
    plt.tight_layout()

    # Save if output file is provided
    if output_file:
        viz.save_figure(fig, output_file)

    return fig

def create_shot_quality_distribution(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 20,
    top_n: int = 15,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a bar chart showing npxG per shot for top players.
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate and filter data
    required_cols = ["Player", "Sh", "npxG"]
    df = viz.validate_dataframe(df, required_cols)
    plot_df = df[df["Sh"] >= min_shots].copy()
    
    if len(plot_df) == 0:
        raise ValueError(f"No players found with at least {min_shots} shots")

    # Calculate npxG per shot if it doesn't exist
    if "npxG_per_shot" not in plot_df.columns:
        plot_df["npxG_per_shot"] = plot_df["npxG"] / plot_df["Sh"]
        plot_df["npxG_per_shot"] = plot_df["npxG_per_shot"].replace([np.inf, -np.inf], 0)

    # Sort and get top players
    plot_df = plot_df.sort_values("npxG_per_shot", ascending=False).head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bar chart
    sns.barplot(
        y="Player",
        x="npxG_per_shot",
        data=plot_df,
        hue="Player",
        palette="viridis",
        legend=False,
        ax=ax
    )

    # Add shot count annotations
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(
            row.npxG_per_shot + (plot_df["npxG_per_shot"].max() * 0.01),
            i,
            f"Shots: {int(row.Sh)}",
            va='center',
            fontsize=9
        )

    # Format chart
    ax.set_title("Players with Highest Expected Goals per Shot", fontsize=16)
    ax.set_xlabel("Non-Penalty xG per Shot", fontsize=12)
    ax.set_ylabel("Player", fontsize=12)
    viz.add_grid(ax, axis='x')
    
    plt.tight_layout()

    # Save if output file is provided
    if output_file:
        viz.save_figure(fig, output_file)

    return fig

def create_shooting_profile_radar(
    df: pd.DataFrame,
    players: List[str],
    output_file: Optional[str] = None,
    min_90s: float = 5,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a radar chart comparing shooting profiles of selected players.
    """
    viz = visualizer or BaseVisualizer()
    
    # Basic validation
    required_cols = ["Player", "90s"]
    df = viz.validate_dataframe(df, required_cols)
    
    # Filter data for selected players and minimum minutes
    plot_df = df[(df["Player"].isin(players)) & (df["90s"] >= min_90s)].copy()

    if len(plot_df) == 0:
        raise ValueError("No matching players found with the specified criteria")

    # Define possible metrics and calculate missing ones
    metrics = ["Sh/90", "SoT%", "G/Sh", "Dist", "npxG/Sh", "G-xG"]
    
    # Calculate derived metrics if possible
    if "Sh/90" not in plot_df.columns and "Sh" in plot_df.columns:
        plot_df["Sh/90"] = plot_df["Sh"] / plot_df["90s"]
        plot_df["Sh/90"] = plot_df["Sh/90"].replace([np.inf, -np.inf], 0)

    if "G/Sh" not in plot_df.columns and "Gls" in plot_df.columns and "Sh" in plot_df.columns:
        plot_df["G/Sh"] = plot_df["Gls"] / plot_df["Sh"]
        plot_df["G/Sh"] = plot_df["G/Sh"].replace([np.inf, -np.inf], 0)

    if "npxG/Sh" not in plot_df.columns and "npxG" in plot_df.columns and "Sh" in plot_df.columns:
        plot_df["npxG/Sh"] = plot_df["npxG"] / plot_df["Sh"]
        plot_df["npxG/Sh"] = plot_df["npxG/Sh"].replace([np.inf, -np.inf], 0)
        
    if "G-xG" not in plot_df.columns and "Gls" in plot_df.columns and "xG" in plot_df.columns:
        plot_df["G-xG"] = plot_df["Gls"] - plot_df["xG"]

    # Check which metrics are available
    available_metrics = [m for m in metrics if m in plot_df.columns]

    if len(available_metrics) < 3:
        raise ValueError(f"Not enough metrics available for radar chart. Available: {available_metrics}")

    # Use the centralized radar chart function with distance as inverted metric
    return create_radar_chart(
        df=plot_df,
        metrics=available_metrics,
        player_column="Player",
        title="Shooting Profile Comparison",
        max_players=len(players),
        output_file=output_file,
        normalize=True,
        invert_metrics=["Dist"],  # Lower distance is better
        visualizer=viz
    )

def create_shot_distance_histogram(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 20,
    overlay_top_scorers: bool = True,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create a histogram of shot distances with top scorers highlighted.
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate and filter data
    required_cols = ["Player", "Sh", "Dist"]
    if overlay_top_scorers:
        required_cols.append("Gls")
        
    df = viz.validate_dataframe(df, required_cols)
    plot_df = df[df["Sh"] >= min_shots].copy()
    
    if len(plot_df) == 0:
        raise ValueError(f"No players found with at least {min_shots} shots")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Main distribution
    sns.histplot(
        data=plot_df,
        x="Dist",
        kde=True,
        color="skyblue",
        alpha=0.6,
        label="All Players",
        ax=ax
    )

    # Overlay top scorers if requested
    if overlay_top_scorers and "Gls" in plot_df.columns:
        top_scorers = plot_df.sort_values("Gls", ascending=False).head(10)
        if len(top_scorers) > 0:
            sns.histplot(
                data=top_scorers,
                x="Dist",
                kde=True,
                color="red",
                alpha=0.4,
                label="Top 10 Scorers",
                ax=ax
            )

    # Format chart
    ax.set_title("Distribution of Shot Distances", fontsize=16)
    ax.set_xlabel("Shot Distance (yards)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    viz.add_grid(ax)
    ax.legend()
    
    plt.tight_layout()

    # Save if output file is provided
    if output_file:
        viz.save_figure(fig, output_file)

    return fig

def create_positional_shooting_comparison(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 15,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create box plots comparing shooting metrics across positions.
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate data
    required_cols = ["Player", "Pos", "Sh", "Gls", "xG"]
    df = viz.validate_dataframe(df, required_cols)
    plot_df = df[df["Sh"] >= min_shots].copy()
    
    if len(plot_df) == 0:
        raise ValueError(f"No players found with at least {min_shots} shots")

    # Simplify positions
    plot_df["Position"] = plot_df["Pos"].apply(
        lambda x: "Defender" if "DF" in x
        else "Midfielder" if "MF" in x  
        else "Forward" if "FW" in x
        else "Other"
    )
    
    # Calculate metrics
    plot_df["G/Sh"] = plot_df["Gls"] / plot_df["Sh"]
    plot_df["G-xG"] = plot_df["Gls"] - plot_df["xG"]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Goals per shot by position
    sns.boxplot(data=plot_df, x="Position", y="G/Sh", ax=axes[0,0])
    axes[0,0].set_title("Goals per Shot by Position")
    axes[0,0].set_ylabel("Goals/Shot")
    
    # 2. Goals above expected by position
    sns.boxplot(data=plot_df, x="Position", y="G-xG", ax=axes[0,1])
    axes[0,1].set_title("Goals Above Expected by Position")
    axes[0,1].set_ylabel("Goals - xG")
    axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. Shot volume by position
    sns.boxplot(data=plot_df, x="Position", y="Sh", ax=axes[1,0])
    axes[1,0].set_title("Shot Volume by Position")
    axes[1,0].set_ylabel("Total Shots")
    
    # 4. xG by position
    sns.boxplot(data=plot_df, x="Position", y="xG", ax=axes[1,1])
    axes[1,1].set_title("Expected Goals by Position")
    axes[1,1].set_ylabel("xG")
    
    for ax in axes.flat:
        viz.add_grid(ax)
        
    plt.suptitle("Shooting Performance by Position", fontsize=16)
    plt.tight_layout()
    
    if output_file:
        viz.save_figure(fig, output_file)
        
    return fig

def create_age_performance_correlation(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_shots: int = 20,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create scatter plots showing age vs shooting performance.
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate data
    required_cols = ["Player", "Age", "Sh", "Gls", "xG"]
    df = viz.validate_dataframe(df, required_cols)
    plot_df = df[df["Sh"] >= min_shots].copy()
    
    if len(plot_df) == 0:
        raise ValueError(f"No players found with at least {min_shots} shots")

    # Calculate metrics
    plot_df["G/Sh"] = plot_df["Gls"] / plot_df["Sh"]
    plot_df["G-xG"] = plot_df["Gls"] - plot_df["xG"]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Age vs Goals per Shot
    sns.scatterplot(data=plot_df, x="Age", y="G/Sh", size="Sh", sizes=(20, 200), alpha=0.7, ax=axes[0])
    z = np.polyfit(plot_df["Age"], plot_df["G/Sh"], 1)
    p = np.poly1d(z)
    axes[0].plot(plot_df["Age"], p(plot_df["Age"]), "r--", alpha=0.8)
    axes[0].set_title("Age vs Goals per Shot")
    axes[0].set_ylabel("Goals/Shot")
    
    # 2. Age vs Finishing (G-xG)
    sns.scatterplot(data=plot_df, x="Age", y="G-xG", size="Sh", sizes=(20, 200), alpha=0.7, ax=axes[1])
    z = np.polyfit(plot_df["Age"], plot_df["G-xG"], 1)
    p = np.poly1d(z)
    axes[1].plot(plot_df["Age"], p(plot_df["Age"]), "r--", alpha=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title("Age vs Finishing Ability")
    axes[1].set_ylabel("Goals - xG")
    
    for ax in axes:
        viz.add_grid(ax)
        
    plt.suptitle("Age and Shooting Performance Correlation", fontsize=16)
    plt.tight_layout()
    
    if output_file:
        viz.save_figure(fig, output_file)
        
    return fig

def create_team_shooting_analysis(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    min_team_shots: int = 50,
    top_teams: int = 15,
    visualizer: Optional[BaseVisualizer] = None
) -> plt.Figure:
    """
    Create team-level shooting analysis visualization.
    """
    viz = visualizer or BaseVisualizer()
    
    # Validate data
    required_cols = ["Player", "Squad", "Sh", "Gls", "xG"]
    df = viz.validate_dataframe(df, required_cols)
    
    # Aggregate by team
    team_stats = df.groupby("Squad").agg({
        "Sh": "sum",
        "Gls": "sum", 
        "xG": "sum",
        "Player": "count"
    }).rename(columns={"Player": "Players"})
    
    # Filter teams with enough shots
    team_stats = team_stats[team_stats["Sh"] >= min_team_shots]
    
    if len(team_stats) == 0:
        raise ValueError(f"No teams found with at least {min_team_shots} shots")
    
    # Calculate team metrics
    team_stats["G/Sh"] = team_stats["Gls"] / team_stats["Sh"]
    team_stats["G-xG"] = team_stats["Gls"] - team_stats["xG"]
    team_stats["Team"] = team_stats.index
    
    # Sort by goals and take top teams
    team_stats = team_stats.sort_values("Gls", ascending=False).head(top_teams)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Team shooting efficiency
    sns.barplot(data=team_stats.reset_index(), y="Squad", x="G/Sh", ax=axes[0,0])
    axes[0,0].set_title("Team Shooting Efficiency (Goals/Shot)")
    axes[0,0].set_xlabel("Goals per Shot")
    
    # 2. Team finishing vs expected
    colors = ['red' if x < 0 else 'green' for x in team_stats["G-xG"]]
    axes[0,1].barh(team_stats.index, team_stats["G-xG"], color=colors, alpha=0.7)
    axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0,1].set_title("Team Finishing vs Expected")
    axes[0,1].set_xlabel("Goals - xG")
    
    # 3. Shot volume vs conversion
    sns.scatterplot(data=team_stats.reset_index(), x="Sh", y="G/Sh", size="Players", 
                   sizes=(50, 300), alpha=0.7, ax=axes[1,0])
    axes[1,0].set_title("Shot Volume vs Conversion Rate")
    axes[1,0].set_xlabel("Total Shots")
    axes[1,0].set_ylabel("Goals/Shot")
    
    # 4. Goals vs xG scatter
    sns.scatterplot(data=team_stats.reset_index(), x="xG", y="Gls", size="Sh",
                   sizes=(50, 300), alpha=0.7, ax=axes[1,1])
    # Add reference line
    max_val = max(team_stats["xG"].max(), team_stats["Gls"].max()) * 1.1
    axes[1,1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    axes[1,1].set_title("Team Goals vs Expected Goals")
    axes[1,1].set_xlabel("Expected Goals")
    axes[1,1].set_ylabel("Goals")
    
    for ax in axes.flat:
        viz.add_grid(ax)
        
    plt.suptitle(f"Team Shooting Analysis - Top {top_teams} Teams by Goals", fontsize=16)
    plt.tight_layout()
    
    if output_file:
        viz.save_figure(fig, output_file)
        
    return fig

def create_shooting_metrics_dashboard(
    shooting_df: pd.DataFrame,
    output_dir: str = "visualizations/shooting",
    min_shots: int = 20,
    min_90s: float = 5,
    visualizer: Optional[BaseVisualizer] = None
) -> List[str]:
    """
    Create a comprehensive dashboard of shooting visualizations.
    """
    viz = visualizer or BaseVisualizer()
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # 1. Finishing ability scatter plot
    try:
        output_file = os.path.join(output_dir, "finishing_skill.png")
        create_finishing_scatter(
            shooting_df,
            output_file=output_file,
            min_shots=min_shots,
            title="Player Finishing Skill: Goals vs. Expected Goals",
            visualizer=viz
        )
        created_files.append(output_file)
        logger.info(f"Created finishing scatter plot: {output_file}")
    except Exception as e:
        logger.error(f"Error creating finishing scatter: {str(e)}")

    # 2. Shot quality distribution
    try:
        output_file = os.path.join(output_dir, "shot_quality.png")
        create_shot_quality_distribution(
            shooting_df,
            output_file=output_file,
            min_shots=min_shots,
            visualizer=viz
        )
        created_files.append(output_file)
        logger.info(f"Created shot quality distribution: {output_file}")
    except Exception as e:
        logger.error(f"Error creating shot quality distribution: {str(e)}")

    # 3. Shot distance histogram
    try:
        output_file = os.path.join(output_dir, "shot_distance.png")
        create_shot_distance_histogram(
            shooting_df,
            output_file=output_file,
            min_shots=min_shots,
            visualizer=viz
        )
        created_files.append(output_file)
        logger.info(f"Created shot distance histogram: {output_file}")
    except Exception as e:
        logger.error(f"Error creating shot distance histogram: {str(e)}")

    # 4. Radar comparison of top scorers
    try:
        if "Gls" in shooting_df.columns:
            top_scorers = shooting_df.sort_values("Gls", ascending=False).head(5)["Player"].tolist()

            if len(top_scorers) >= 3:  # Need at least 3 players for meaningful radar
                output_file = os.path.join(output_dir, "top_scorers_radar.png")
                create_shooting_profile_radar(
                    shooting_df,
                    players=top_scorers[:5],
                    output_file=output_file,
                    min_90s=min_90s,
                    visualizer=viz
                )
                created_files.append(output_file)
                logger.info(f"Created shooting profile radar: {output_file}")
            else:
                logger.warning(f"Only {len(top_scorers)} top scorers found, need at least 3 for radar chart")
        else:
            logger.warning("Goals column not found, skipping radar chart")
    except Exception as e:
        logger.error(f"Error creating shooting profile radar: {str(e)}")

    # 5. Positional shooting comparison
    try:
        output_file = os.path.join(output_dir, "positional_comparison.png")
        create_positional_shooting_comparison(
            shooting_df,
            output_file=output_file,
            min_shots=min_shots//2,  # Use lower threshold for position analysis
            visualizer=viz
        )
        created_files.append(output_file)
        logger.info(f"Created positional shooting comparison: {output_file}")
    except Exception as e:
        logger.error(f"Error creating positional comparison: {str(e)}")

    # 6. Age vs performance correlation
    try:
        if "Age" in shooting_df.columns:
            output_file = os.path.join(output_dir, "age_performance.png")
            create_age_performance_correlation(
                shooting_df,
                output_file=output_file,
                min_shots=min_shots,
                visualizer=viz
            )
            created_files.append(output_file)
            logger.info(f"Created age performance correlation: {output_file}")
        else:
            logger.warning("Age column not found, skipping age correlation analysis")
    except Exception as e:
        logger.error(f"Error creating age performance correlation: {str(e)}")

    # 7. Team shooting analysis
    try:
        if "Squad" in shooting_df.columns:
            output_file = os.path.join(output_dir, "team_analysis.png")
            create_team_shooting_analysis(
                shooting_df,
                output_file=output_file,
                min_team_shots=50,
                visualizer=viz
            )
            created_files.append(output_file)
            logger.info(f"Created team shooting analysis: {output_file}")
        else:
            logger.warning("Squad column not found, skipping team analysis")
    except Exception as e:
        logger.error(f"Error creating team shooting analysis: {str(e)}")

    return created_files
