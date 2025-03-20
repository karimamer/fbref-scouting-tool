from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Import advanced analysis functions from new module locations
from src.analysis.advanced.versatility import calculate_versatility_score
from src.analysis.advanced.progression import analyze_progressive_actions
from src.analysis.advanced.possession_impact import get_expected_possession_impact
from src.analysis.advanced.clustering import cluster_player_profiles, find_undervalued_players

# Import base analysis functions from existing locations
from src.analysis.basic.midfielders import (
    analyze_progressive_midfielders,
    identify_pressing_midfielders
)
from src.analysis.basic.playmakers import identify_playmakers
from src.analysis.basic.forwards import find_clinical_forwards

# Import data handling functions
from src.data.loaders import readfromhtml
from src.data.processors import process_player_stats, process_passing_df, process_shooting_df

# Import database operations
from src.db.operations import insert_dataframe

# Import utility functions
from src.utils.normalization import normalize_metric


def run_advanced_analysis(
    passing_df: pd.DataFrame,
    possession_df: pd.DataFrame,
    defensive_df: pd.DataFrame,
    shooting_df: pd.DataFrame,
    min_90s: float = 10.0,
    max_age: int = 30,
    top_n: int = 20,
    save_to_db: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run advanced player analysis on the provided dataframes.

    Parameters:
    -----------
    passing_df: DataFrame with passing statistics
    possession_df: DataFrame with possession statistics
    defensive_df: DataFrame with defensive statistics
    shooting_df: DataFrame with shooting statistics
    min_90s: Minimum 90s played to be included in analysis
    max_age: Maximum player age to include
    top_n: Number of top players to return in each category
    save_to_db: Whether to save results to database

    Returns:
    --------
    Dictionary with analysis results
    """
    results = {}

    # Filter by age if needed
    if max_age is not None:
        passing_df = passing_df[passing_df["Age"] <= max_age].copy()
        possession_df = possession_df[possession_df["Age"] <= max_age].copy()
        defensive_df = defensive_df[defensive_df["Age"] <= max_age].copy()
        shooting_df = shooting_df[shooting_df["Age"] <= max_age].copy()

    # 1. Calculate player versatility scores
    print("Calculating player versatility scores...")
    versatility = calculate_versatility_score(
        passing_df=passing_df,
        possession_df=possession_df,
        defensive_df=defensive_df,
        shooting_df=shooting_df,
        min_90s=min_90s
    )
    results["versatile_players"] = versatility.head(top_n)

    # 2. Analyze progressive actions
    print("Analyzing progressive actions...")
    progression_results = analyze_progressive_actions(
        possession_df=possession_df,
        passing_df=passing_df,
        min_90s=min_90s,
        top_n=top_n
    )
    results.update(progression_results)

    # 3. Expected Possession Impact
    print("Calculating Expected Possession Impact (xPI)...")
    xpi_results = get_expected_possession_impact(
        possession_df=possession_df,
        min_90s=min_90s
    )
    results["possession_impact"] = xpi_results.head(top_n)

    # 4. Cluster players by position group
    print("Clustering player profiles...")
    # Midfielders
    midfield_metrics = [
        "PrgP", "PrgC", "Carries", "KP", "Tkl", "Int", "Att 3rd",
        "final_third_entries_90", "prog_carries_90", "touches_90"
    ]

    # Ensure metrics are calculated
    possession_filtered = possession_df[possession_df["90s"] >= min_90s].copy()
    possession_filtered["final_third_entries_90"] = possession_filtered["1/3"] / possession_filtered["90s"]
    possession_filtered["prog_carries_90"] = possession_filtered["PrgC"] / possession_filtered["90s"]
    possession_filtered["touches_90"] = possession_filtered["Touches"] / possession_filtered["90s"]

    # Run clustering for midfielders
    try:
        df_with_clusters, cluster_info = cluster_player_profiles(
            df=possession_filtered,
            metrics=[m for m in midfield_metrics if m in possession_filtered.columns],
            n_clusters=5,
            position_group="MF",
            min_90s=min_90s
        )
        results["midfielder_clusters"] = df_with_clusters[df_with_clusters["Pos"].str.contains("MF")].head(top_n)
    except Exception as e:
        print(f"Error in midfielder clustering: {str(e)}")

    # Save results to database if requested
    if save_to_db:
        print("Saving results to database...")
        for name, df in results.items():
            try:
                insert_dataframe(df, f"advanced_{name}")
                print(f"Saved {name} to database")
            except Exception as e:
                print(f"Error saving {name} to database: {str(e)}")

    # Add extra data to the results for visualization
    for name, df in results.items():
        if 'Player' in df.columns and len(df) > 0:
            df['analysis_type'] = name.replace('_', ' ').title()

    return results

def generate_advanced_report(results: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a formatted report from the advanced analysis results.

    Parameters:
    -----------
    results: Dict[str, pd.DataFrame]
        Analysis results from run_advanced_analysis function

    Returns:
    --------
    str: Formatted report
    """
    report = ["# Advanced Player Analysis Report\n"]

    # Report sections and descriptions
    sections = {
        "versatile_players": "## Most Versatile Players\nPlayers who excel across multiple skill areas (passing, possession, defense).",
        "overall_progressors": "## Top Overall Progressors\nPlayers who excel at moving the ball forward through carries, passes, and receiving.",
        "top_carriers": "## Top Ball Carriers\nPlayers who excel at progressing the ball through dribbling and carrying.",
        "top_passers": "## Top Progressive Passers\nPlayers who excel at progressing the ball through passing.",
        "top_receivers": "## Top Progressive Receivers\nPlayers who excel at finding space to receive progressive passes.",
        "versatile_progressors": "## Most Versatile Progressors\nPlayers who can progress the ball effectively in multiple ways.",
        "possession_impact": "## Highest Expected Possession Impact (xPI)\nPlayers with the greatest overall impact on their team's possession play.",
        "midfielder_clusters": "## Midfielder Profile Clusters\nGroups of midfielders with similar statistical profiles."
    }

    # Add each section to the report
    for section_name, section_header in sections.items():
        if section_name in results and not results[section_name].empty:
            report.append(section_header)
            report.append("\n")

            # Select appropriate columns based on the section
            if section_name == "versatile_players":
                display_cols = ["Player", "Squad", "Pos", "Age", "90s",
                                "passing_score", "possession_score", "defensive_score",
                                "adjusted_versatility"]
            elif section_name == "midfielder_clusters":
                display_cols = ["Player", "Squad", "Pos", "Age", "90s", "cluster"]
            elif "progressors" in section_name or "carriers" in section_name or "passers" in section_name or "receivers" in section_name:
                display_cols = ["Player", "Squad", "Pos", "Age", "90s", "progression_type"]
                if "total_progression_score" in results[section_name].columns:
                    display_cols.append("total_progression_score")
            elif section_name == "possession_impact":
                display_cols = ["Player", "Squad", "Pos", "Age", "90s", "xPI", "position_relative_xPI"]
            else:
                # Default columns
                display_cols = ["Player", "Squad", "Pos", "Age", "90s"]

            # Only include columns that actually exist in the dataframe
            cols_to_display = [col for col in display_cols if col in results[section_name].columns]
            df_section = results[section_name][cols_to_display].head(10)

            # Format the table
            report.append(df_section.to_markdown(index=False, floatfmt=".2f"))
            report.append("\n\n")

    # Final formatting
    return "\n".join(report)
