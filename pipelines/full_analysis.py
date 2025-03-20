import os
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
from src.utils.logging_setup import log_execution_time, log_data_stats

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Pipeline for comprehensive soccer player analysis."""

    def __init__(
        self,
        min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
        top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
        positions: Optional[List[str]] = None,
        min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
        max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
        cache_enabled: bool = True,
        save_to_db: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the analysis pipeline.

        Args:
            min_shots: Minimum number of shots for forward analysis
            top_n: Number of top players to return in each category
            positions: List of positions to filter for
            min_90s: Minimum number of 90-minute periods played
            max_age: Maximum player age to include
            cache_enabled: Whether to cache loaded data
            save_to_db: Whether to save results to database
            output_dir: Directory to save output files
        """
        self.min_shots = min_shots
        self.top_n = top_n
        self.positions = positions or DEFAULT_ANALYSIS_PARAMS["positions"]
        self.min_90s = min_90s
        self.max_age = max_age
        self.cache_enabled = cache_enabled
        self.save_to_db = save_to_db
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize components
        self.data_loader = DataLoader(cache_enabled=self.cache_enabled)
        self.db_manager = DatabaseManager() if self.save_to_db else None

        # Store results
        self.results: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Any] = {}

    def load_data(self, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all required data for analysis.

        Args:
            force_reload: Whether to force reload data from source

        Returns:
            Dictionary of loaded dataframes
        """
        logger.info("Loading data for analysis")
        start_time = datetime.now()

        # Load raw data
        data = {}
        data["passing"] = self.data_loader.get_data("passing", force_reload=force_reload)
        data["shooting"] = self.data_loader.get_data("shooting", force_reload=force_reload)
        data["possession"] = self.data_loader.get_data("possession", force_reload=force_reload)
        data["defense"] = self.data_loader.get_data("defense", force_reload=force_reload)
        data["shot_creation"] = self.data_loader.get_data("shot_creation", force_reload=force_reload)

        # Process data
        data["passing_processed"] = process_passing_stats(data["passing"])
        data["shooting_processed"] = process_shooting_stats(data["shooting"])
        data["defense_processed"] = process_defensive_stats(data["defense"])

        # Log data stats
        for name, df in data.items():
            log_data_stats(logger, df, name)

        log_execution_time(logger, start_time, "Data loading")
        return data

    def run_analyses(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Run all analyses on the loaded data.

        Args:
            data: Dictionary of dataframes to analyze

        Returns:
            Dictionary of analysis results
        """
        logger.info("Running player analyses")
        start_time = datetime.now()

        results = {}

        # Store raw/filtered data
        results["top_passers"] = data["passing_processed"].head(self.top_n)
        results["top_shooters"] = data["shooting_processed"].head(self.top_n)
        results["top_creators"] = data["shot_creation"].head(self.top_n)

        # Run specialized analyses
        results["playmakers"] = identify_playmakers(
            data["passing_processed"]
        ).head(self.top_n)

        results["clinical_forwards"] = find_clinical_forwards(
            data["shooting_processed"],
            min_shots=self.min_shots
        ).head(self.top_n)

        results["progressive_midfielders"] = analyze_progressive_midfielders(
            data["possession"]
        ).head(self.top_n)

        results["pressing_midfielders"] = identify_pressing_midfielders(
            data["defense_processed"]
        ).head(self.top_n)

        results["passing_quality"] = analyze_passing_quality(
            data["passing_processed"]
        ).head(self.top_n)

        results["complete_midfielders"] = find_complete_midfielders(
            data["passing_processed"],
            data["possession"],
            data["defense_processed"]
        ).head(self.top_n)

        log_execution_time(logger, start_time, "Analysis execution")
        return results

    def save_results(self) -> None:
        """Save analysis results to database and/or files."""
        if not self.results:
            logger.warning("No results to save")
            return

        start_time = datetime.now()

        # Save to database if enabled
        if self.save_to_db and self.db_manager:
            logger.info("Saving results to database")
            try:
                with self.db_manager:
                    for name, df in self.results.items():
                        if not df.empty:
                            self.db_manager.insert_dataframe(
                                df, name, metadata=self.metadata
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
                        file_path = os.path.join(self.output_dir, f"{name}_{timestamp}.csv")
                        df.to_csv(file_path, index=False)
                        logger.info(f"Saved {name} to {file_path}")

                # Save metadata
                metadata_path = os.path.join(self.output_dir, f"metadata_{timestamp}.csv")
                pd.DataFrame([self.metadata]).to_csv(metadata_path, index=False)
            except Exception as e:
                logger.error(f"Error saving to files: {str(e)}")

        log_execution_time(logger, start_time, "Saving results")

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive analysis report.

        Args:
            output_file: Optional file path to save the report

        Returns:
            Report content as string
        """
        logger.info("Generating analysis report")
        start_time = datetime.now()

        report = ["# Soccer Player Analysis Report\n"]
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Add parameters
        report.append("## Analysis Parameters\n")
        for param, value in self.metadata.items():
            if param != "analysis_date":
                report.append(f"- **{param}**: {value}")
        report.append("\n")

        # Add each analysis section
        for category, df in self.results.items():
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

    def run(self, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run the complete analysis pipeline.

        Args:
            force_reload: Whether to force reload data from source

        Returns:
            Dictionary of analysis results
        """
        logger.info("Starting analysis pipeline")
        pipeline_start = datetime.now()

        # Create metadata
        self.metadata = {
            "min_shots": self.min_shots,
            "top_n": self.top_n,
            "positions": str(self.positions),
            "min_90s": self.min_90s,
            "max_age": self.max_age,
            "analysis_date": datetime.now().isoformat()
        }

        # Load data
        data = self.load_data(force_reload=force_reload)

        # Run analyses
        self.results = self.run_analyses(data)

        # Add parameters to results
        self.results["parameters"] = pd.DataFrame([self.metadata])

        # Save results
        self.save_results()

        # Log total execution time
        log_execution_time(logger, pipeline_start, "Complete analysis pipeline")

        return self.results


def run_analysis_pipeline(
    min_shots: int = DEFAULT_ANALYSIS_PARAMS["min_shots"],
    top_n: int = DEFAULT_ANALYSIS_PARAMS["top_n"],
    positions: Optional[List[str]] = None,
    min_90s: int = DEFAULT_ANALYSIS_PARAMS["min_90s"],
    max_age: int = DEFAULT_ANALYSIS_PARAMS["max_age"],
    force_reload: bool = False,
    save_to_db: bool = True,
    output_dir: Optional[str] = None,
    report_file: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete analysis pipeline with the given parameters.

    This function provides a simple interface to run the full pipeline
    without having to create an AnalysisPipeline instance directly.

    Args:
        min_shots: Minimum number of shots for forward analysis
        top_n: Number of top players to return in each category
        positions: List of positions to filter for
        min_90s: Minimum number of 90-minute periods played
        max_age: Maximum player age to include
        force_reload: Whether to force reload data from source
        save_to_db: Whether to save results to database
        output_dir: Directory to save output files
        report_file: File path to save the report

    Returns:
        Dictionary of analysis results
    """
    # Create and run pipeline
    pipeline = AnalysisPipeline(
        min_shots=min_shots,
        top_n=top_n,
        positions=positions,
        min_90s=min_90s,
        max_age=max_age,
        cache_enabled=True,
        save_to_db=save_to_db,
        output_dir=output_dir
    )

    results = pipeline.run(force_reload=force_reload)

    # Generate report if requested
    if report_file:
        pipeline.generate_report(report_file)

    return results
