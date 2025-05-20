import os
import warnings # Added for deprecation warnings
import pandas as pd
import numpy as np
from src.utils.visualization import plot_radar_chart
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
    DEPRECATED: Create a radar chart comparing shooting profiles of selected players.
    Use plot_radar_chart from src.utils.visualization for a more general and updated solution.

    Args:
        df: DataFrame with shooting statistics
        players: List of player names to compare
        output_file: Path to save the output file
        min_90s: Minimum 90s played filter

    Returns:
        matplotlib Figure object
    """
    warnings.warn(
        "create_shooting_profile_radar is deprecated and will be removed in a future version. "
        "Use plot_radar_chart from src.utils.visualization instead.",
        FutureWarning,
        stacklevel=2 # Points the warning to the caller of this function
    )
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
        top_scorers_names = shooting_df.sort_values("Gls", ascending=False).head(5)["Player"].tolist()

        if len(top_scorers_names) < 3: # Original check: Need at least 3 players for a meaningful radar
            print(f"Not enough top scorers ({len(top_scorers_names)}) to generate a meaningful radar chart. Skipping.")
        else:
            # Filter the DataFrame for the selected players and minimum 90s played
            plot_df = shooting_df[
                (shooting_df["Player"].isin(top_scorers_names)) &
                (shooting_df["90s"] >= min_90s)
            ].copy() # Use .copy() to avoid SettingWithCopyWarning

            if plot_df.empty:
                print(f"No players found matching criteria for shooting radar chart (top scorers list: {top_scorers_names}, min_90s={min_90s}). Skipping.")
            # Check if enough unique players remain after filtering. plot_radar_chart needs at least one.
            elif len(plot_df["Player"].unique()) < 1: 
                print(f"Not enough unique players ({len(plot_df['Player'].unique())}) after min_90s filter for shooting radar. Skipping.")
            else:
                # Define the metrics for the radar chart.
                metric_columns = ["Sh/90", "SoT%", "G/Sh", "Dist", "npxG/Sh", "G-xG"]
                
                # Calculate missing metrics if they are not already in the DataFrame.
                # This includes handling potential division by zero.
                if "90s" in plot_df.columns and "Sh" in plot_df.columns:
                    if "Sh/90" not in plot_df.columns:
                        plot_df["Sh/90"] = np.nan # Initialize column
                        # Calculate Sh/90 only for rows where 90s > small epsilon to avoid division by zero/very small numbers
                        valid_90s_mask = plot_df["90s"] > 1e-6 
                        plot_df.loc[valid_90s_mask, "Sh/90"] = plot_df.loc[valid_90s_mask, "Sh"] / plot_df.loc[valid_90s_mask, "90s"]
                
                if "Sh" in plot_df.columns:
                    if "G/Sh" not in plot_df.columns and "Gls" in plot_df.columns:
                        plot_df["G/Sh"] = np.nan # Initialize column
                        valid_sh_mask = plot_df["Sh"] > 0 # Shots must be positive integer
                        plot_df.loc[valid_sh_mask, "G/Sh"] = plot_df.loc[valid_sh_mask, "Gls"] / plot_df.loc[valid_sh_mask, "Sh"]
                    
                    if "npxG/Sh" not in plot_df.columns and "npxG" in plot_df.columns:
                        plot_df["npxG/Sh"] = np.nan # Initialize column
                        valid_sh_mask = plot_df["Sh"] > 0
                        plot_df.loc[valid_sh_mask, "npxG/Sh"] = plot_df.loc[valid_sh_mask, "npxG"] / plot_df.loc[valid_sh_mask, "Sh"]

                if "G-xG" not in plot_df.columns and "Gls" in plot_df.columns and "xG" in plot_df.columns:
                    plot_df["G-xG"] = plot_df["Gls"] - plot_df["xG"]

                # Filter out metrics that are not in plot_df or are all NaN after calculation.
                # These are the metrics that will actually be plotted.
                available_metrics = [
                    m for m in metric_columns 
                    if m in plot_df.columns and plot_df[m].notna().any()
                ]

                if len(available_metrics) < 3:
                    print(f"Not enough available and valid metrics ({len(available_metrics)}) for shooting radar chart. Need at least 3. Available: {available_metrics}. Skipping.")
                else:
                    # Drop rows that have NaN for any of the available_metrics to be plotted.
                    # This ensures that each player passed to plot_radar_chart has data for all axes.
                    final_plot_df = plot_df.dropna(subset=available_metrics).copy() 
                    
                    # Get the list of player names from the filtered and NaN-dropped DataFrame.
                    entity_names_list = final_plot_df["Player"].unique().tolist() # Use unique() to be safe.
                    
                    if final_plot_df.empty or not entity_names_list:
                         print(f"No players remaining with complete data for available metrics ({available_metrics}) after NaN drop. Skipping radar chart.")
                    elif len(entity_names_list) < 1: # Double check, though covered by plot_df.empty and unique list check
                         print(f"Not enough players ({len(entity_names_list)}) for radar chart after processing NaNs. Skipping.")
                    else:
                        # Prepare data for plot_radar_chart:
                        # 1. Create a list of single-row DataFrames, one for each player.
                        # 2. Create a list of player names corresponding to these DataFrames.
                        data_frames_list = []
                        for player_name in entity_names_list: # Iterate based on unique players in final_plot_df
                            player_specific_data = final_plot_df[final_plot_df["Player"] == player_name]
                            # Ensure we take only the first row if somehow duplicates exist (should not with unique names)
                            # And ensure columns are in the correct 'available_metrics' order.
                            player_data_df = pd.DataFrame(player_specific_data[available_metrics].iloc[[0]].values, columns=available_metrics)
                            data_frames_list.append(player_data_df)
                        
                        if not data_frames_list: # Should be caught by earlier checks on entity_names_list
                             print(f"Dataframe list for radar chart is empty despite having entity names. Skipping.")
                        else:
                            output_file = os.path.join(output_dir, "top_scorers_radar.png")
                            # Define normalization specifications, e.g., 'Dist' where lower is better.
                            normalization_specs = {'Dist': 'inverted'} 

                            # Call the generic radar chart plotting function.
                            plot_radar_chart(
                                data_frames=data_frames_list,
                                metric_columns=available_metrics, # Use the validated list of metrics
                                entity_names=entity_names_list,
                                title="Shooting Profile Comparison", # Hardcoded title from old function
                                normalize=True, # plot_radar_chart defaults to True, this is for clarity
                                normalization_specs=normalization_specs,
                                output_file=output_file
                            )
                            created_files.append(output_file)
    except Exception as e:
        print(f"Error creating shooting profile radar: {str(e)}")

    return created_files
