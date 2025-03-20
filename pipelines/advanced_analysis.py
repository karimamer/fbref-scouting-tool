import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from config.settings import (
    DEFAULT_ANALYSIS_PARAMS,
    ADVANCED_ANALYSIS_PARAMS,
    VISUALIZATION_DIR,
    REPORTS_DIR
)
from src.data.loaders import DataLoader
from src.data.processors import (
    process_passing_stats,
    process_shooting_stats,
    process_defensive_stats
)
from src.analysis.advanced.versatility import calculate_versatility_score
from src.analysis.advanced.progression import analyze_progressive_actions
from src.analysis.advanced.possession_impact import get_expected_possession_impact
from src.analysis.advanced.clustering import cluster_player_profiles, find_undervalued_players
from src.db.operations import DatabaseManager
from src.utils.logging_setup import setup_logging, log_execution_time, log_data_stats
from src.utils.visualization import create_dashboard

# Set up logging
logger = logging.getLogger(__name__)

def run_advanced_analysis(
    min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
    positions: List[str] = None,
    min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
    max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
    force_reload: bool = False,
    save_to_db: bool = True,
    create_visualizations: bool = True,
    visualization_dir: str = VISUALIZATION_DIR,
    report_file: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run advanced player analysis with enhanced metrics and visualizations.

    Args:
        min_shots: Minimum number of shots for forward analysis
        top_n: Number of top players to return in each category
        positions: List of positions to filter for
        min_90s: Minimum number of 90-minute periods played
        max_age: Maximum player age to include
        force_reload: If True, reload data from source
        save_to_db: If True, save results to database
        create_visualizations: If True, generate visualization charts
        visualization_dir: Directory to save visualizations
        report_file: Path to save the analysis report

    Returns:
        Dictionary containing different analysis results
    """
    logger.info("Starting advanced player analysis")
    start_time = datetime.now()

    # Use default positions if none provided
    if positions is None:
        positions = DEFAULT_ANALYSIS_PARAMS["positions"]

    # Store parameters for logging and metadata
    params = {
        "min_shots": min_shots,
        "top_n": top_n,
        "positions": positions,
        "min_90s": min_90s,
        "max_age": max_age,
        "analysis_type": "advanced",
        "analysis_date": datetime.now().isoformat()
    }

    # Initialize data loader
    data_loader = DataLoader(cache_enabled=True)

    # Load and process data
    logger.info("Loading and processing data for advanced analysis")
    passing_stats = process_passing_stats(
        data_loader.get_data("passing", force_reload=force_reload)
    )
    shooting_stats = process_shooting_stats(
        data_loader.get_data("shooting", force_reload=force_reload)
    )
    possession_stats = data_loader.get_data("possession", force_reload=force_reload)
    defensive_stats = process_defensive_stats(
        data_loader.get_data("defense", force_reload=force_reload)
    )

    # Log data loading stats
    log_data_stats(logger, passing_stats, "passing_stats")
    log_data_stats(logger, shooting_stats, "shooting_stats")
    log_data_stats(logger, possession_stats, "possession_stats")
    log_data_stats(logger, defensive_stats, "defensive_stats")

    # Filter by age - properly handle Age column format
    if max_age is not None:
        # Convert Age to numeric - handle format like "30-039" or "29-226"
        logger.info(f"Filtering players by age (max_age={max_age})")

        # Handle passing stats age
        if "Age" in passing_stats.columns:
            # Check if Age is already numeric
            if not pd.api.types.is_numeric_dtype(passing_stats["Age"]):
                # Extract main age number before the dash
                passing_stats["Age_numeric"] = passing_stats["Age"].str.split("-").str[0].astype(int)
                passing_stats = passing_stats[passing_stats["Age_numeric"] <= max_age].copy()
            else:
                passing_stats = passing_stats[passing_stats["Age"] <= max_age].copy()

        # Handle possession stats age
        if "Age" in possession_stats.columns:
            if not pd.api.types.is_numeric_dtype(possession_stats["Age"]):
                possession_stats["Age_numeric"] = possession_stats["Age"].str.split("-").str[0].astype(int)
                possession_stats = possession_stats[possession_stats["Age_numeric"] <= max_age].copy()
            else:
                possession_stats = possession_stats[possession_stats["Age"] <= max_age].copy()

        # Handle defensive stats age
        if "Age" in defensive_stats.columns:
            if not pd.api.types.is_numeric_dtype(defensive_stats["Age"]):
                defensive_stats["Age_numeric"] = defensive_stats["Age"].str.split("-").str[0].astype(int)
                defensive_stats = defensive_stats[defensive_stats["Age_numeric"] <= max_age].copy()
            else:
                defensive_stats = defensive_stats[defensive_stats["Age"] <= max_age].copy()

        # Handle shooting stats age
        if "Age" in shooting_stats.columns:
            if not pd.api.types.is_numeric_dtype(shooting_stats["Age"]):
                shooting_stats["Age_numeric"] = shooting_stats["Age"].str.split("-").str[0].astype(int)
                shooting_stats = shooting_stats[shooting_stats["Age_numeric"] <= max_age].copy()
            else:
                shooting_stats = shooting_stats[shooting_stats["Age"] <= max_age].copy()

    results = {}

    # 1. Calculate player versatility scores
    logger.info("Calculating player versatility scores")
    versatility = calculate_versatility_score(
        passing_df=passing_stats,
        possession_df=possession_stats,
        defensive_df=defensive_stats,
        shooting_df=shooting_stats,
        min_90s=min_90s
    )
    results["versatile_players"] = versatility.head(top_n)

    # 2. Analyze progressive actions
    logger.info("Analyzing progressive actions")
    progression_results = analyze_progressive_actions(
        possession_df=possession_stats,
        passing_df=passing_stats,
        min_90s=min_90s,
        top_n=top_n
    )
    results.update(progression_results)

    # 3. Expected Possession Impact
    logger.info("Calculating Expected Possession Impact (xPI)")
    xpi_results = get_expected_possession_impact(
        possession_df=possession_stats,
        min_90s=min_90s
    )
    results["possession_impact"] = xpi_results.head(top_n)

    # 4. Cluster players by position group
    logger.info("Clustering player profiles")
    cluster_count = ADVANCED_ANALYSIS_PARAMS.get("cluster_count", 5)

    # Define metrics for clustering different position groups
    clustering_metrics = {
        "MF": [
            "PrgP", "PrgC", "Carries", "KP", "Tkl", "Int", "Att 3rd",
            "final_third_entries_90", "prog_carries_90", "touches_90"
        ],
        "FW": [
            "Gls", "npxG", "Sh", "SoT%", "G-xG", "PrgC", "Att Pen", "CPA"
        ],
        "DF": [
            "Tkl", "TklW", "Int", "Blocks", "Clr", "PrgDist", "PrgP", "PrgC"
        ]
    }

    # Ensure metrics are calculated for clustering
    possession_filtered = possession_stats[possession_stats["90s"] >= min_90s].copy()
    possession_filtered["final_third_entries_90"] = possession_filtered["1/3"] / possession_filtered["90s"]
    possession_filtered["prog_carries_90"] = possession_filtered["PrgC"] / possession_filtered["90s"]
    possession_filtered["touches_90"] = possession_filtered["Touches"] / possession_filtered["90s"]

    # Run clustering for each position group
    for position, metrics in clustering_metrics.items():
        try:
            position_df = possession_filtered[possession_filtered["Pos"].str.contains(position)]
            if len(position_df) >= cluster_count * 2:  # Ensure enough players for meaningful clusters
                df_with_clusters, cluster_info = cluster_player_profiles(
                    df=possession_filtered,
                    metrics=[m for m in metrics if m in possession_filtered.columns],
                    n_clusters=cluster_count,
                    position_group=position,
                    min_90s=min_90s
                )
                results[f"{position.lower()}_clusters"] = df_with_clusters[
                    df_with_clusters["Pos"].str.contains(position)
                ].head(top_n)

                # Store cluster info for reporting
                results[f"{position.lower()}_cluster_info"] = pd.DataFrame({
                    "cluster_id": list(cluster_info["representatives"].keys()),
                    "representative_player": [info["player"] for info in cluster_info["representatives"].values()],
                    "representative_team": [info["team"] for info in cluster_info["representatives"].values()],
                    "cluster_size": [cluster_info["sizes"].get(cluster_id, 0)
                                    for cluster_id in cluster_info["representatives"].keys()]
                })
        except Exception as e:
            logger.error(f"Error in {position} clustering: {str(e)}")

    # Store analysis parameters
    results["parameters"] = pd.DataFrame([params])

    # Create visualizations if requested
    if create_visualizations:
        logger.info("Creating visualizations")
        try:
            os.makedirs(visualization_dir, exist_ok=True)
            viz_files = create_dashboard(results, output_dir=visualization_dir, prefix="advanced_")
            logger.info(f"Created {len(viz_files)} visualization files in {visualization_dir}")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    # Generate report if requested
    if report_file:
        logger.info(f"Generating advanced analysis report to {report_file}")
        report = generate_advanced_report(results)
        try:
            # Ensure reports directory exists
            os.makedirs(os.path.dirname(os.path.abspath(report_file)), exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report)
        except Exception as e:
            logger.error(f"Error writing report to {report_file}: {str(e)}")

    # Save results to database if requested
    if save_to_db:
        logger.info("Saving advanced analysis results to database")
        save_results_to_db(results, params)

    # Log execution time
    log_execution_time(logger, start_time, "Advanced player analysis")

    return results

def save_results_to_db(
    results: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any]
) -> None:
    """
    Save analysis results to the database.

    Args:
        results: Dictionary of analysis results
        metadata: Metadata to include with each table
    """
    try:
        with DatabaseManager() as db:
            for name, df in results.items():
                if not df.empty:
                    table_name = f"advanced_{name}"
                    db.insert_dataframe(df, table_name, metadata=metadata)
                    logger.info(f"Saved {table_name} to database with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error saving results to database: {str(e)}")

def generate_advanced_report(results: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a formatted report from the advanced analysis results.

    Args:
        results: Analysis results dictionary

    Returns:
        Formatted markdown report
    """
    report = ["# Advanced Soccer Player Analysis Report\n"]
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Add parameters if available
    if "parameters" in results and not results["parameters"].empty:
        params = results["parameters"].iloc[0]
        report.append("## Analysis Parameters\n")
        for param, value in params.items():
            if param != "analysis_date":
                report.append(f"- **{param}**: {value}")
        report.append("\n")

    # Define sections with descriptions
    sections = {
        "versatile_players": "## Most Versatile Players\nPlayers who excel across multiple skill areas (passing, possession, defense).",
        "overall_progressors": "## Top Overall Progressors\nPlayers who excel at moving the ball forward through carries, passes, and receiving.",
        "top_carriers": "## Top Ball Carriers\nPlayers who excel at progressing the ball through dribbling and carrying.",
        "top_passers": "## Top Progressive Passers\nPlayers who excel at progressing the ball through passing.",
        "top_receivers": "## Top Progressive Receivers\nPlayers who excel at finding space to receive progressive passes.",
        "versatile_progressors": "## Most Versatile Progressors\nPlayers who can progress the ball effectively in multiple ways.",
        "possession_impact": "## Highest Expected Possession Impact (xPI)\nPlayers with the greatest overall impact on their team's possession play.",
        "mf_clusters": "## Midfielder Profile Clusters\nGroups of midfielders with similar statistical profiles.",
        "fw_clusters": "## Forward Profile Clusters\nGroups of forwards with similar statistical profiles.",
        "df_clusters": "## Defender Profile Clusters\nGroups of defenders with similar statistical profiles."
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
            elif "clusters" in section_name:
                display_cols = ["Player", "Squad", "Pos", "Age", "90s", "cluster"]

                # Add cluster info if available
                info_key = f"{section_name.split('_')[0]}_cluster_info"
                if info_key in results and not results[info_key].empty:
                    report.append("### Cluster Representatives\n")
                    report.append(results[info_key].to_markdown(index=False))
                    report.append("\n### Cluster Members\n")
            elif section_name in ["overall_progressors", "top_carriers", "top_passers", "top_receivers", "versatile_progressors"]:
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

    # Add visualization references if they exist
    viz_dir = VISUALIZATION_DIR
    if os.path.exists(viz_dir) and any(file.startswith("advanced_") for file in os.listdir(viz_dir)):
        report.append("## Visualizations\n")
        report.append("The following visualizations were generated as part of this analysis:\n")

        viz_files = [f for f in os.listdir(viz_dir) if f.startswith("advanced_")]
        for viz_file in viz_files:
            report.append(f"- [{viz_file}]({os.path.join(viz_dir, viz_file)})")

        report.append("\n")

    return "\n".join(report)

if __name__ == "__main__":
    # Initialize logging
    setup_logging()

    # Simple argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Run advanced soccer player analysis")
    parser.add_argument("--min-shots", type=int, default=DEFAULT_ANALYSIS_PARAMS["min_shots"])
    parser.add_argument("--top-n", type=int, default=DEFAULT_ANALYSIS_PARAMS["top_n"])
    parser.add_argument("--min-90s", type=int, default=DEFAULT_ANALYSIS_PARAMS["min_90s"])
    parser.add_argument("--max-age", type=int, default=DEFAULT_ANALYSIS_PARAMS["max_age"])
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--no-db", action="store_true", help="Don't save to database")
    parser.add_argument("--no-viz", action="store_true", help="Don't create visualizations")
    parser.add_argument("--report", type=str, help="Path to save report file")
    args = parser.parse_args()

    # Run analysis
    results = run_advanced_analysis(
        min_shots=args.min_shots,
        top_n=args.top_n,
        min_90s=args.min_90s,
        max_age=args.max_age,
        force_reload=args.force_reload,
        save_to_db=not args.no_db,
        create_visualizations=not args.no_viz,
        report_file=args.report
    )

    # Print basic information
    if "versatile_players" in results and not results["versatile_players"].empty:
        print("\nTop 5 Most Versatile Players:")
        top_players = results["versatile_players"][["Player", "Squad", "Pos", "adjusted_versatility"]].head()
        print(top_players.to_string(index=False))
