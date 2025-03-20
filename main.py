"""
Main entry point for the soccer analysis application.
"""
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from config.settings import DEFAULT_ANALYSIS_PARAMS
from src.data.loaders import DataLoader
from src.data.processors import (
    process_passing_stats,
    process_shooting_stats,
    process_defensive_stats
)
from src.analysis.basic.playmakers import identify_playmakers
from src.analysis.basic.forwards import find_clinical_forwards
from src.analysis.basic.midfielders import (
    analyze_progressive_midfielders,
    identify_pressing_midfielders,
    find_complete_midfielders,
    analyze_passing_quality
)
# Import advanced analysis modules
from src.analysis.advanced.versatility import calculate_versatility_score
from src.analysis.advanced.progression import analyze_progressive_actions
from src.analysis.advanced.possession_impact import get_expected_possession_impact
from src.analysis.advanced.clustering import cluster_player_profiles, find_undervalued_players
from src.db.operations import DatabaseManager
from src.utils.logging_setup import setup_logging, log_execution_time, log_data_stats
from src.utils.visualization import create_dashboard


# Set up logging
logger = setup_logging()


def analyze_players(
    min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
    positions: List[str] = None,
    min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
    max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
    force_reload: bool = False,
    save_to_db: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive player analysis combining all statistics and analysis methods.

    Args:
        min_shots: Minimum number of shots for forward analysis
        top_n: Number of top players to return in each category
        positions: List of positions to filter for
        min_90s: Minimum number of 90-minute periods played
        max_age: Maximum player age to include
        force_reload: If True, reload data from source
        save_to_db: If True, save results to database

    Returns:
        Dictionary containing different analysis results
    """
    logger.info("Starting player analysis")
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
        "analysis_date": datetime.now().isoformat()
    }

    # Initialize data loader
    data_loader = DataLoader(cache_enabled=True)

    # Load and process data
    logger.info("Loading and processing data")
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
    shot_creation_stats = data_loader.get_data("shot_creation", force_reload=force_reload)

    # Log data loading stats
    log_data_stats(logger, passing_stats, "passing_stats")
    log_data_stats(logger, shooting_stats, "shooting_stats")
    log_data_stats(logger, possession_stats, "possession_stats")
    log_data_stats(logger, defensive_stats, "defensive_stats")
    log_data_stats(logger, shot_creation_stats, "shot_creation_stats")

    # Run analyses
    logger.info("Running player analyses")
    results = {}

    # Store raw data
    results["top_passers"] = passing_stats.head(top_n)
    results["top_shooters"] = shooting_stats.head(top_n)
    results["top_creators"] = shot_creation_stats.head(top_n)

    # Run specialized analyses
    results["playmakers"] = identify_playmakers(passing_stats).head(top_n)
    results["clinical_forwards"] = find_clinical_forwards(
        shooting_stats, min_shots=min_shots
    ).head(top_n)
    results["progressive_midfielders"] = analyze_progressive_midfielders(
        possession_stats
    ).head(top_n)
    results["pressing_midfielders"] = identify_pressing_midfielders(
        defensive_stats
    ).head(top_n)
    results["passing_quality"] = analyze_passing_quality(passing_stats).head(top_n)
    results["complete_midfielders"] = find_complete_midfielders(
        passing_stats, possession_stats, defensive_stats
    ).head(top_n)

    # Store analysis parameters
    results["parameters"] = pd.DataFrame([params])

    # Save results to database if requested
    if save_to_db:
        logger.info("Saving results to database")
        save_results_to_db(results, params)

    # Log execution time
    log_execution_time(logger, start_time, "Player analysis")

    return results


def run_advanced_analysis(
    min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
    positions: List[str] = None,
    min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
    max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
    force_reload: bool = False,
    save_to_db: bool = True,
    create_visualizations: bool = True
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

    # Filter by age
    if max_age is not None:
        passing_stats = passing_stats[passing_stats["Age"] <= max_age].copy()
        possession_stats = possession_stats[possession_stats["Age"] <= max_age].copy()
        defensive_stats = defensive_stats[defensive_stats["Age"] <= max_age].copy()
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
    midfield_metrics = [
        "PrgP", "PrgC", "Carries", "KP", "Tkl", "Int", "Att 3rd",
        "final_third_entries_90", "prog_carries_90", "touches_90"
    ]

    # Ensure metrics are calculated
    possession_filtered = possession_stats[possession_stats["90s"] >= min_90s].copy()
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
        logger.error(f"Error in midfielder clustering: {str(e)}")

    # Store analysis parameters
    results["parameters"] = pd.DataFrame([params])

    # Create visualizations if requested
    if create_visualizations:
        logger.info("Creating visualizations")
        try:
            viz_files = create_dashboard(results, prefix="advanced_")
            logger.info(f"Created {len(viz_files)} visualization files")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    # Save results to database if requested
    if save_to_db:
        logger.info("Saving advanced analysis results to database")
        save_results_to_db(results, params, table_prefix="advanced_")

    # Log execution time
    log_execution_time(logger, start_time, "Advanced player analysis")

    return results


def save_results_to_db(
    results: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    table_prefix: str = ""
) -> None:
    """
    Save analysis results to the database.

    Args:
        results: Dictionary of analysis results
        metadata: Metadata to include with each table
        table_prefix: Optional prefix for table names
    """
    try:
        with DatabaseManager() as db:
            for name, df in results.items():
                if not df.empty:
                    table_name = f"{table_prefix}{name}"
                    db.insert_dataframe(df, table_name, metadata=metadata)
                    logger.info(f"Saved {table_name} to database with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error saving results to database: {str(e)}")


def generate_analysis_report(results: Dict[str, pd.DataFrame], report_type="basic") -> str:
    """
    Generate a formatted report from the analysis results.

    Args:
        results: Analysis results from analyze_players function
        report_type: Type of report to generate (basic or advanced)

    Returns:
        Formatted markdown report
    """
    report = [f"# Soccer Player {report_type.title()} Analysis Report\n"]
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Add parameters if available
    if "parameters" in results and not results["parameters"].empty:
        params = results["parameters"].iloc[0]
        report.append("## Analysis Parameters\n")
        for param, value in params.items():
            if param != "analysis_date":
                report.append(f"- **{param}**: {value}")
        report.append("\n")

    if report_type == "basic":
        # Add each analysis section for basic report
        for category, df in results.items():
            if category != "parameters" and not df.empty:
                # Format category name for display
                display_name = category.replace('_', ' ').title()
                report.append(f"## {display_name}\n")

                # Select key columns for display
                display_cols = ["Player", "Squad", "Age", "Pos"]

                # Add score column if it exists
                score_cols = [col for col in df.columns if col.endswith('_score')]
                if score_cols:
                    display_cols.extend(score_cols)

                # Ensure all requested columns exist in the dataframe
                display_cols = [col for col in display_cols if col in df.columns]

                # Convert to markdown table
                report.append(df[display_cols].head(10).to_markdown(index=False))
                report.append("\n")
    else:
        # Add each section to the report for advanced report
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

    return "\n".join(report)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Soccer Player Analysis Tool")

    parser.add_argument(
        "--analysis-type",
        choices=["basic", "advanced", "both"],
        default="basic",
        help="Type of analysis to run"
    )
    parser.add_argument(
        "--min-shots",
        type=int,
        default=DEFAULT_ANALYSIS_PARAMS["min_shots"],
        help="Minimum number of shots for forward analysis"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_ANALYSIS_PARAMS["top_n"],
        help="Number of top players to return in each category"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=DEFAULT_ANALYSIS_PARAMS["positions"],
        help="List of positions to filter for"
    )
    parser.add_argument(
        "--min-90s",
        type=int,
        default=DEFAULT_ANALYSIS_PARAMS["min_90s"],
        help="Minimum number of 90-minute periods played"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=DEFAULT_ANALYSIS_PARAMS["max_age"],
        help="Maximum player age to include"
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force data reload from source"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to database"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Do not create visualizations (for advanced analysis)"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        help="Path to save report markdown file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Store common parameters
    analysis_params = {
        "min_shots": args.min_shots,
        "top_n": args.top_n,
        "positions": args.positions,
        "min_90s": args.min_90s,
        "max_age": args.max_age,
        "force_reload": args.force_reload,
        "save_to_db": not args.no_save
    }

    # Run basic analysis if requested
    basic_results = None
    if args.analysis_type in ["basic", "both"]:
        logger.info("Running basic analysis")
        basic_results = analyze_players(**analysis_params)

        # Generate and print basic report
        basic_report = generate_analysis_report(basic_results, report_type="basic")
        print(basic_report)

        # Save basic report to file if specified
        if args.report_file and args.analysis_type == "basic":
            try:
                with open(args.report_file, 'w') as f:
                    f.write(basic_report)
                logger.info(f"Basic report saved to {args.report_file}")
            except Exception as e:
                logger.error(f"Error saving basic report to file: {str(e)}")

    # Run advanced analysis if requested
    advanced_results = None
    if args.analysis_type in ["advanced", "both"]:
        logger.info("Running advanced analysis")
        advanced_results = run_advanced_analysis(
            **analysis_params,
            create_visualizations=not args.no_visualizations
        )

        # Generate and print advanced report
        advanced_report = generate_analysis_report(advanced_results, report_type="advanced")
        print(advanced_report)

        # Save advanced report to file if specified
        if args.report_file and args.analysis_type == "advanced":
            try:
                with open(args.report_file, 'w') as f:
                    f.write(advanced_report)
                logger.info(f"Advanced report saved to {args.report_file}")
            except Exception as e:
                logger.error(f"Error saving advanced report to file: {str(e)}")

    # Save combined report if both analysis types were run
    if args.report_file and args.analysis_type == "both":
        try:
            combined_report = "# Combined Soccer Analysis Report\n\n"
            combined_report += "## Basic Analysis\n\n"
            combined_report += generate_analysis_report(basic_results, report_type="basic")
            combined_report += "\n\n## Advanced Analysis\n\n"
            combined_report += generate_analysis_report(advanced_results, report_type="advanced")

            with open(args.report_file, 'w') as f:
                f.write(combined_report)
            logger.info(f"Combined report saved to {args.report_file}")
        except Exception as e:
            logger.error(f"Error saving combined report to file: {str(e)}")
