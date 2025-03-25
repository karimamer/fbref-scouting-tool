import argparse
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

from config.settings import DEFAULT_ANALYSIS_PARAMS
from src.data.loaders import DataLoader
from src.data.processors import (
    process_passing_stats,
    process_shooting_stats,
    process_defensive_stats,
    process_combined_shooting_data,  # Add the new processor
    process_shot_quality             # Add the new processor
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

from src.analysis.advanced.shooting_analyzer import (
    analyze_shooting_efficiency,
    analyze_shooting_profile,
    identify_shot_creation_specialists,
    calculate_finishing_skill_over_time,
    analyze_shot_quality
)

from src.db.operations import DatabaseManager
from src.utils.logging_setup import setup_logging, log_execution_time, log_data_stats
from src.utils.visualization import create_dashboard

# Import the new shooting visualizations
from src.utils.shooting_visualizations import create_shooting_metrics_dashboard
from src.utils.pipeline_helpers import filter_by_age
# Set up logging
logger = setup_logging()


def analyze_players(
    min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
    positions: List[str] = DEFAULT_ANALYSIS_PARAMS["positions"],
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

    # Add new shooting analyses
    results["shooting_efficiency"] = analyze_shooting_efficiency(
        shooting_stats, min_shots=min_shots, min_90s=min_90s
    ).head(top_n)

    results["shooting_profiles"] = analyze_shooting_profile(
        shooting_stats, min_shots=min_shots
    ).head(top_n)

    results["shot_quality"] = analyze_shot_quality(
        shooting_stats, min_shots=min_shots
    ).head(top_n)

    results["finishing_skill"] = calculate_finishing_skill_over_time(
        shooting_stats, min_90s=min_90s, min_shots=min_shots
    ).head(top_n)

    # Run combined shot creation and shooting analysis if available
    if not shot_creation_stats.empty:
        results["shot_creation_specialists"] = identify_shot_creation_specialists(
            shooting_stats, shot_creation_stats, min_90s=min_90s
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
    positions: List[str] = DEFAULT_ANALYSIS_PARAMS["positions"],
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
        passing_stats = filter_by_age(passing_stats, max_age)
        possession_stats = filter_by_age(possession_stats, max_age)
        defensive_stats = filter_by_age(defensive_stats, max_age)
        shooting_stats = filter_by_age(shooting_stats, max_age)

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


def run_shooting_analysis(
    min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
    positions: List[str] = DEFAULT_ANALYSIS_PARAMS["positions"],
    min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
    max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
    force_reload: bool = False,
    save_to_db: bool = True,
    create_visualizations: bool = True,
    output_dir: str = "visualizations/shooting"
) -> Dict[str, pd.DataFrame]:
    """
    Run focused shooting analysis with enhanced metrics and visualizations.

    Args:
        min_shots: Minimum number of shots for analysis
        top_n: Number of top players to return in each category
        positions: List of positions to filter for
        min_90s: Minimum number of 90-minute periods played
        max_age: Maximum player age to include
        force_reload: If True, reload data from source
        save_to_db: If True, save results to database
        create_visualizations: If True, generate visualization charts
        output_dir: Directory to save visualization files

    Returns:
        Dictionary containing different shooting analysis results
    """
    logger.info("Starting specialized shooting analysis")
    start_time = datetime.now()

    # Store parameters for logging and metadata
    params = {
        "min_shots": min_shots,
        "top_n": top_n,
        "positions": positions,
        "min_90s": min_90s,
        "max_age": max_age,
        "analysis_type": "shooting",
        "analysis_date": datetime.now().isoformat()
    }

    # Initialize data loader
    data_loader = DataLoader(cache_enabled=True)

    # Load and process data
    logger.info("Loading and processing data for shooting analysis")
    shooting_stats = process_shooting_stats(
        data_loader.get_data("shooting", force_reload=force_reload),
        min_shots=min_shots
    )

    # Load supporting data
    try:
        shot_creation_stats = data_loader.get_data("shot_creation", force_reload=force_reload)
        possession_stats = data_loader.get_data("possession", force_reload=force_reload)
    except Exception as e:
        logger.warning(f"Could not load supporting data: {str(e)}")
        shot_creation_stats = pd.DataFrame()
        possession_stats = pd.DataFrame()

    # Log data stats
    log_data_stats(logger, shooting_stats, "shooting_stats")
    log_data_stats(logger, shot_creation_stats, "shot_creation_stats")
    log_data_stats(logger, possession_stats, "possession_stats")

    # Filter by age
    if max_age is not None and "Age" in shooting_stats.columns:
        # Convert Age to numeric if needed
        if not pd.api.types.is_numeric_dtype(shooting_stats["Age"]):
            # Extract main age number before the dash
            shooting_stats["Age_numeric"] = shooting_stats["Age"].str.split("-").str[0].astype(int)
            shooting_stats = shooting_stats[shooting_stats["Age_numeric"] <= max_age].copy()
        else:
            shooting_stats = shooting_stats[shooting_stats["Age"] <= max_age].copy()

    # Create combined shooting data
    combined_shooting_data = process_combined_shooting_data(
        shooting_stats,
        shot_creation_stats,
        possession_stats
    )

    # Calculate specialized shot quality metrics
    shot_quality_data = process_shot_quality(shooting_stats)

    # Initialize results dictionary
    results = {}

    # Run all shooting analyses

    # 1. Standard efficiency analysis
    results["clinical_forwards"] = find_clinical_forwards(
        shooting_stats, min_shots=min_shots
    ).head(top_n)

    # 2. Enhanced shooting efficiency analysis
    results["shooting_efficiency"] = analyze_shooting_efficiency(
        shooting_stats, min_shots=min_shots, min_90s=min_90s
    ).head(top_n)

    # 3. Shooting profile categorization
    results["shooting_profiles"] = analyze_shooting_profile(
        shooting_stats, min_shots=min_shots
    ).head(top_n)

    # 4. Shot quality analysis
    results["shot_quality"] = analyze_shot_quality(
        shot_quality_data, min_shots=min_shots
    ).head(top_n)

    # 5. Finishing skill analysis
    results["finishing_skill"] = calculate_finishing_skill_over_time(
        shooting_stats, min_90s=min_90s, min_shots=min_shots
    ).head(top_n)

    # 6. Combined shot creation analysis if available
    if not shot_creation_stats.empty:
        results["shot_creation_specialists"] = identify_shot_creation_specialists(
            shooting_stats, shot_creation_stats, min_90s=min_90s
        ).head(top_n)

    # Add the processed data for reference
    results["combined_shooting_data"] = combined_shooting_data.head(top_n * 3)
    results["shot_quality_data"] = shot_quality_data.head(top_n * 3)

    # Add parameters to results
    results["parameters"] = pd.DataFrame([params])

    # Create visualizations if requested
    if create_visualizations:
        logger.info("Creating shooting visualizations")
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            viz_files = create_shooting_metrics_dashboard(
                shooting_stats,
                output_dir=output_dir,
                min_shots=min_shots,
                min_90s=min_90s
            )
            logger.info(f"Created {len(viz_files)} visualization files")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    # Save results to database if requested
    if save_to_db:
        logger.info("Saving shooting analysis results to database")
        save_results_to_db(results, params, table_prefix="shooting_")

    # Log execution time
    log_execution_time(logger, start_time, "Shooting analysis")

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
        report_type: Type of report to generate (basic, advanced, or shooting)

    Returns:
        Formatted markdown report
    """
    import os

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

    elif report_type == "shooting":
        # Define sections for shooting report
        sections = {
            "clinical_forwards": "## Clinical Forwards\nForwards who excel at finishing their chances.",
            "shooting_efficiency": "## Shooting Efficiency\nPlayers with the best overall shooting efficiency.",
            "shooting_profiles": "## Shooting Profiles\nClassification of players based on their shooting patterns.",
            "finishing_skill": "## Finishing Skill\nPlayers who consistently outperform their expected goals.",
            "shot_quality": "## Shot Quality\nPlayers who take the highest quality shots.",
            "shot_creation_specialists": "## Shot Creation Specialists\nPlayers who excel at both shooting and creating shots."
        }

        # Add each section to the report
        for section_name, section_header in sections.items():
            if section_name in results and not results[section_name].empty:
                report.append(section_header)
                report.append("\n")

                # Select appropriate columns based on the section
                if section_name == "clinical_forwards":
                    display_cols = ["Player", "Squad", "Pos", "Age", "90s", "Gls", "Sh", "conversion_rate", "efficiency_score"]
                elif section_name == "shooting_efficiency":
                    display_cols = ["Player", "Squad", "Pos", "Age", "90s", "Gls", "Sh", "SoT%", "G/Sh", "shooting_efficiency_score"]
                elif section_name == "shooting_profiles":
                    display_cols = ["Player", "Squad", "Pos", "Age", "90s", "Sh", "SoT%", "Dist", "shooting_profile"]
                elif section_name == "finishing_skill":
                    display_cols = ["Player", "Squad", "Pos", "Age", "90s", "Gls", "xG", "np_goals_above_xG", "finishing_category"]
                elif section_name == "shot_quality":
                    display_cols = ["Player", "Squad", "Pos", "Age", "90s", "Sh", "npxG_per_shot", "shot_selection_category"]
                elif section_name == "shot_creation_specialists":
                    display_cols = ["Player", "Squad", "Pos", "Age", "90s", "Gls", "SCA90", "GCA90", "contribution_type", "shot_contribution_score"]
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
        viz_dir = "visualizations/shooting"
        if os.path.exists(viz_dir) and any(file.endswith(('.png', '.jpg')) for file in os.listdir(viz_dir)):
            report.append("## Visualizations\n")
            report.append("The following visualizations were generated as part of this analysis:\n")

            viz_files = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.jpg'))]
            for viz_file in viz_files:
                report.append(f"- [{viz_file}]({os.path.join(viz_dir, viz_file)})")

            report.append("\n")

    else:  # Advanced report
        # Add each section to the report
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
        choices=["basic", "advanced", "shooting", "all"],
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
    if args.analysis_type in ["basic", "all"]:
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
    if args.analysis_type in ["advanced", "all"]:
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

    # Run shooting analysis if requested
    shooting_results = None
    if args.analysis_type in ["shooting", "all"]:
        logger.info("Running shooting analysis")
        shooting_results = run_shooting_analysis(
            **analysis_params,
            create_visualizations=not args.no_visualizations,
            output_dir="visualizations/shooting"
        )

        # Generate and print shooting report
        shooting_report = generate_analysis_report(shooting_results, report_type="shooting")
        print(shooting_report)

        # Save shooting report to file if specified
        if args.report_file and args.analysis_type == "shooting":
            try:
                with open(args.report_file, 'w') as f:
                    f.write(shooting_report)
                logger.info(f"Shooting report saved to {args.report_file}")
            except Exception as e:
                logger.error(f"Error saving shooting report to file: {str(e)}")

    # Save combined report if all analysis types were run
    if args.report_file and args.analysis_type == "all":
        try:
            combined_report = "# Combined Soccer Analysis Report\n\n"

            if basic_results:
                combined_report += "## Basic Analysis\n\n"
                combined_report += generate_analysis_report(basic_results, report_type="basic")
                combined_report += "\n\n"

            if advanced_results:
                combined_report += "## Advanced Analysis\n\n"
                combined_report += generate_analysis_report(advanced_results, report_type="advanced")
                combined_report += "\n\n"

            if shooting_results:
                combined_report += "## Shooting Analysis\n\n"
                combined_report += generate_analysis_report(shooting_results, report_type="shooting")
                combined_report += "\n\n"

            with open(args.report_file, 'w') as f:
                f.write(combined_report)
            logger.info(f"Combined report saved to {args.report_file}")
        except Exception as e:
            logger.error(f"Error saving combined report to file: {str(e)}")
