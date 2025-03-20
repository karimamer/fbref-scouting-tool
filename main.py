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
from src.analysis.player_scout import (
    analyze_progressive_midfielders,
    find_clinical_forwards,
    find_complete_midfielders,
    identify_playmakers,
    identify_pressing_midfielders,
    analyze_passing_quality
)
from src.db.operations import DatabaseManager
from src.utils.logging_setup import setup_logging, log_execution_time, log_data_stats


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
                    db.insert_dataframe(df, name, metadata=metadata)
                    logger.info(f"Saved {name} to database with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error saving results to database: {str(e)}")


def generate_analysis_report(results: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a formatted report from the analysis results.

    Args:
        results: Analysis results from analyze_players function

    Returns:
        Formatted markdown report
    """
    report = ["# Soccer Player Analysis Report\n"]
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Add parameters if available
    if "parameters" in results and not results["parameters"].empty:
        params = results["parameters"].iloc[0]
        report.append("## Analysis Parameters\n")
        for param, value in params.items():
            if param != "analysis_date":
                report.append(f"- **{param}**: {value}")
        report.append("\n")

    # Add each analysis section
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

    return "\n".join(report)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Soccer Player Analysis Tool")

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
        "--report-file",
        type=str,
        help="Path to save report markdown file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Perform comprehensive analysis
    results = analyze_players(
        min_shots=args.min_shots,
        top_n=args.top_n,
        positions=args.positions,
        min_90s=args.min_90s,
        max_age=args.max_age,
        force_reload=args.force_reload,
        save_to_db=not args.no_save
    )

    # Generate and print report
    report = generate_analysis_report(results)
    print(report)

    # Save report to file if specified
    if args.report_file:
        try:
            with open(args.report_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.report_file}")
        except Exception as e:
            logger.error(f"Error saving report to file: {str(e)}")
