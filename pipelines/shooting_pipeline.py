import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from config.settings import DEFAULT_ANALYSIS_PARAMS
from src.data.loaders import DataLoader
from src.data.processors import process_shooting_stats
from src.analysis.basic.forwards import find_clinical_forwards
from src.analysis.basic.midfielders import find_complete_midfielders

# Import the new shooting analysis functions
from src.analysis.shooting_analyzer import (
    analyze_shooting_efficiency,
    analyze_shooting_profile,
    identify_shot_creation_specialists,
    calculate_finishing_skill_over_time,
    analyze_shot_quality
)
from src.utils.visualization import create_dashboard
from src.utils.shooting_visualizations import create_shooting_metrics_dashboard
from src.db.operations import DatabaseManager
from src.utils.logging_setup import log_execution_time, log_data_stats

logger = logging.getLogger(__name__)

class ShootingAnalysisPipeline:
    """Pipeline for comprehensive shooting statistics analysis."""

    def __init__(
        self,
        min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
        top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
        positions: Optional[List[str]] = None,
        min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
        max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
        cache_enabled: bool = True,
        save_to_db: bool = True,
        output_dir: Optional[str] = None,
        visualization_dir: Optional[str] = None
    ):
        """
        Initialize the shooting analysis pipeline.

        Args:
            min_shots: Minimum number of shots for analysis
            top_n: Number of top players to return in each category
            positions: List of positions to filter for
            min_90s: Minimum number of 90-minute periods played
            max_age: Maximum player age to include
            cache_enabled: Whether to cache loaded data
            save_to_db: Whether to save results to database
            output_dir: Directory to save report outputs
            visualization_dir: Directory to save visualizations
        """
        self.min_shots = min_shots
        self.top_n = top_n
        self.positions = positions or DEFAULT_ANALYSIS_PARAMS["positions"]
        self.min_90s = min_90s
        self.max_age = max_age
        self.cache_enabled = cache_enabled
        self.save_to_db = save_to_db
        self.output_dir = output_dir
        self.visualization_dir = visualization_dir or "visualizations/shooting"

        # Create output directories if they don't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if self.visualization_dir and not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)

        # Initialize components
        self.data_loader = DataLoader(cache_enabled=self.cache_enabled)
        self.db_manager = DatabaseManager() if self.save_to_db else None

        # Store results
        self.results: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Any] = {}

    def load_data(self, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all required data for shooting analysis.

        Args:
            force_reload: Whether to force reload data from source

        Returns:
            Dictionary of loaded dataframes
        """
        logger.info("Loading data for shooting analysis")
        start_time = datetime.now()

        # Load raw data
        data = {}
        data["shooting"] = self.data_loader.get_data("shooting", force_reload=force_reload)
        data["shot_creation"] = self.data_loader.get_data("shot_creation", force_reload=force_reload)

        # Optional data that might enhance the analysis
        try:
            data["passing"] = self.data_loader.get_data("passing", force_reload=force_reload)
        except Exception as e:
            logger.warning(f"Could not load passing data: {str(e)}")
            data["passing"] = pd.DataFrame()

        try:
            data["possession"] = self.data_loader.get_data("possession", force_reload=force_reload)
        except Exception as e:
            logger.warning(f"Could not load possession data: {str(e)}")
            data["possession"] = pd.DataFrame()

        # Process data
        data["shooting_processed"] = process_shooting_stats(data["shooting"])

        # Log data stats
        for name, df in data.items():
            log_data_stats(logger, df, name)

        log_execution_time(logger, start_time, "Data loading")
        return data

    def run_analyses(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Run all shooting analyses on the loaded data.

        Args:
            data: Dictionary of dataframes to analyze

        Returns:
            Dictionary of analysis results
        """
        logger.info("Running shooting analyses")
        start_time = datetime.now()

        results = {}

        # 1. Basic clinical_forwards analysis (existing functionality)
        results["clinical_forwards"] = find_clinical_forwards(
            data["shooting_processed"],
            min_shots=self.min_shots
        ).head(self.top_n)

        # 2. Enhanced shooting efficiency analysis
        results["shooting_efficiency"] = analyze_shooting_efficiency(
            data["shooting_processed"],
            min_shots=self.min_shots,
            min_90s=self.min_90s
        ).head(self.top_n)

        # 3. Shooting profile analysis
        results["shooting_profiles"] = analyze_shooting_profile(
            data["shooting_processed"],
            min_shots=self.min_shots
        )

        # 4. Finishing skill analysis
        results["finishing_skill"] = calculate_finishing_skill_over_time(
            data["shooting_processed"],
            min_90s=self.min_90s,
            min_shots=self.min_shots
        ).head(self.top_n)

        # 5. Shot quality analysis
        results["shot_quality"] = analyze_shot_quality(
            data["shooting_processed"],
            min_shots=self.min_shots
        ).head(self.top_n)

        # 6. Shot creation specialists (if shot creation data is available)
        if "shot_creation" in data and not data["shot_creation"].empty:
            results["shot_creation_specialists"] = identify_shot_creation_specialists(
                data["shooting_processed"],
                data["shot_creation"],
                min_90s=self.min_90s
            ).head(self.top_n)

        log_execution_time(logger, start_time, "Shooting analysis execution")
        return results

    def save_results(self) -> None:
        """Save analysis results to database and/or files."""
        if not self.results:
            logger.warning("No results to save")
            return

        start_time = datetime.now()

        # Save to database if enabled
        if self.save_to_db and self.db_manager:
            logger.info("Saving shooting results to database")
            try:
                with self.db_manager:
                    for name, df in self.results.items():
                        if not df.empty:
                            table_name = f"shooting_{name}"
                            self.db_manager.insert_dataframe(
                                df, table_name, metadata=self.metadata
                            )
                            logger.info(f"Saved {name} to database with {len(df)} rows")
            except Exception as e:
                logger.error(f"Error saving to database: {str(e)}")

        # Save to CSV files if output directory is specified
        if self.output_dir:
            logger.info(f"Saving results to {self.output_dir}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            try:
                for name, df in self.results.items():
                    if not df.empty:
                        file_path = os.path.join(self.output_dir, f"shooting_{name}_{timestamp}.csv")
                        df.to_csv(file_path, index=False)
                        logger.info(f"Saved {name} to {file_path}")

                # Save metadata
                metadata_path = os.path.join(self.output_dir, f"shooting_metadata_{timestamp}.csv")
                pd.DataFrame([self.metadata]).to_csv(metadata_path, index=False)
            except Exception as e:
                logger.error(f"Error saving to files: {str(e)}")

        log_execution_time(logger, start_time, "Saving results")

    def create_visualizations(self) -> List[str]:
        """
        Create visualizations for shooting analysis results.

        Returns:
            List of created visualization file paths
        """
        logger.info("Creating shooting visualizations")
        start_time = datetime.now()

        created_files = []

        try:
            if "shooting_processed" in self.data:
                # Create comprehensive shooting dashboard
                viz_files = create_shooting_metrics_dashboard(
                    self.data["shooting_processed"],
                    output_dir=self.visualization_dir,
                    min_shots=self.min_shots,
                    min_90s=self.min_90s
                )
                created_files.extend(viz_files)
                logger.info(f"Created {len(viz_files)} shooting visualizations")

            # Additional custom visualizations for specific analyses
            if "shooting_profiles" in self.results and not self.results["shooting_profiles"].empty:
                # Example custom visualization for shooting profiles
                pass
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

        log_execution_time(logger, start_time, "Creating visualizations")
        return created_files

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive shooting analysis report.

        Args:
            output_file: Optional file path to save the report

        Returns:
            Report content as string
        """
        logger.info("Generating shooting analysis report")
        start_time = datetime.now()

        report = ["# Shooting Analysis Report\n"]
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Add parameters
        report.append("## Analysis Parameters\n")
        for param, value in self.metadata.items():
            if param != "analysis_date":
                report.append(f"- **{param}**: {value}")
        report.append("\n")

        # Add each analysis section
        sections = {
            "clinical_forwards": "## Clinical Forwards\nForwards who excel at finishing their chances.",
            "shooting_efficiency": "## Shooting Efficiency\nPlayers with the best overall shooting efficiency.",
            "shooting_profiles": "## Shooting Profiles\nClassification of players based on their shooting patterns.",
            "finishing_skill": "## Finishing Skill\nPlayers who consistently outperform their expected goals.",
            "shot_quality": "## Shot Quality\nPlayers who take the highest quality shots.",
            "shot_creation_specialists": "## Shot Creation Specialists\nPlayers who excel at both shooting and creating shots."
        }

        for section_name, section_header in sections.items():
            if section_name in self.results and not self.results[section_name].empty:
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
                cols_to_display = [col for col in display_cols if col in self.results[section_name].columns]
                df_section = self.results[section_name][cols_to_display].head(10)

                # Format the table
                report.append(df_section.to_markdown(index=False, floatfmt=".2f"))
                report.append("\n\n")

        # Add visualization references if they exist
        if self.visualization_dir and os.path.exists(self.visualization_dir):
            report.append("## Visualizations\n")
            report.append("The following visualizations were generated as part of this analysis:\n")

            viz_files = [f for f in os.listdir(self.visualization_dir) if f.endswith(('.png', '.jpg'))]
            for viz_file in viz_files:
                report.append(f"- [{viz_file}]({os.path.join(self.visualization_dir, viz_file)})")

            report.append("\n")

        # Combine report sections
        report_text = "\n".join(report)

        # Save to file if path provided
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")

        log_execution_time(logger, start_time, "Report generation")
        return report_text

    def run(self, force_reload: bool = False, output_file: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Run the complete shooting analysis pipeline.

        Args:
            force_reload: Whether to force reload data from source
            output_file: Path to save the analysis report

        Returns:
            Dictionary of analysis results
        """
        logger.info("Starting shooting analysis pipeline")
        pipeline_start = datetime.now()

        # Create metadata
        self.metadata = {
            "min_shots": self.min_shots,
            "top_n": self.top_n,
            "positions": str(self.positions),
            "min_90s": self.min_90s,
            "max_age": self.max_age,
            "analysis_type": "shooting",
            "analysis_date": datetime.now().isoformat()
        }

        # Load data
        self.data = self.load_data(force_reload=force_reload)

        # Run analyses
        self.results = self.run_analyses(self.data)

        # Add parameters to results
        self.results["parameters"] = pd.DataFrame([self.metadata])

        # Create visualizations
        if self.visualization_dir:
            self.create_visualizations()

        # Save results
        self.save_results()

        # Generate report if requested
        if output_file:
            self.generate_report(output_file)

        # Log total execution time
        log_execution_time(logger, pipeline_start, "Complete shooting analysis pipeline")

        return self.results


def run_shooting_analysis(
    min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
    positions: Optional[List[str]] = None,
    min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
    max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
    force_reload: bool = False,
    save_to_db: bool = True,
    output_dir: Optional[str] = None,
    visualization_dir: Optional[str] = None,
    report_file: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete shooting analysis pipeline with the given parameters.

    This function provides a simple interface to run the full pipeline
    without having to create a ShootingAnalysisPipeline instance directly.

    Args:
        min_shots: Minimum number of shots for analysis
        top_n: Number of top players to return in each category
        positions: List of positions to filter for
        min_90s: Minimum number of 90-minute periods played
        max_age: Maximum player age to include
        force_reload: Whether to force reload data from source
        save_to_db: Whether to save results to database
        output_dir: Directory to save output files
        visualization_dir: Directory to save visualizations
        report_file: File path to save the report

    Returns:
        Dictionary of analysis results
    """
    # Create and run pipeline
    pipeline = ShootingAnalysisPipeline(
        min_shots=min_shots,
        top_n=top_n,
        positions=positions,
        min_90s=min_90s,
        max_age=max_age,
        cache_enabled=True,
        save_to_db=save_to_db,
        output_dir=output_dir,
        visualization_dir=visualization_dir
    )

    results = pipeline.run(force_reload=force_reload, output_file=report_file)

    return results


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run shooting statistics analysis")
    parser.add_argument("--min-shots", type=int, default=DEFAULT_ANALYSIS_PARAMS["min_shots"])
    parser.add_argument("--top-n", type=int, default=DEFAULT_ANALYSIS_PARAMS["top_n"])
    parser.add_argument("--min-90s", type=int, default=DEFAULT_ANALYSIS_PARAMS["min_90s"])
    parser.add_argument("--max-age", type=int, default=DEFAULT_ANALYSIS_PARAMS["max_age"])
    parser.add_argument("--positions", nargs="+", default=["FW", "MF,FW"])
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--no-db", action="store_true", help="Don't save to database")
    parser.add_argument("--output-dir", type=str, default="reports/shooting")
    parser.add_argument("--viz-dir", type=str, default="visualizations/shooting")
    parser.add_argument("--report", type=str, help="Path to save report file")

    args = parser.parse_args()

    # Run analysis
    results = run_shooting_analysis(
        min_shots=args.min_shots,
        top_n=args.top_n,
        positions=args.positions,
        min_90s=args.min_90s,
        max_age=args.max_age,
        force_reload=args.force_reload,
        save_to_db=not args.no_db,
        output_dir=args.output_dir,
        visualization_dir=args.viz_dir,
        report_file=args.report or f"{args.output_dir}/shooting_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    # Print basic results
    if "shooting_efficiency" in results and not results["shooting_efficiency"].empty:
        print("\nTop 5 Players by Shooting Efficiency:")
        top_players = results["shooting_efficiency"][["Player", "Squad", "Pos", "Gls", "Sh", "shooting_efficiency_score"]].head(5)
        print(top_players.to_string(index=False))
