import os
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd

from config.settings import DEFAULT_ANALYSIS_PARAMS
from src.data.loaders import DataLoader
from src.db.operations import DatabaseManager
from src.utils.logging_setup import log_execution_time

logger = logging.getLogger(__name__)

class DailyUpdatePipeline:
    """Pipeline for daily updates of soccer player statistics."""

    def __init__(
        self,
        output_dir: Optional[str] = "reports",
        generate_report: bool = True
    ):
        """
        Initialize the daily update pipeline.

        Args:
            output_dir: Directory to save reports
            generate_report: Whether to generate report files
        """
        self.output_dir = output_dir
        self.generate_report = generate_report

        # Create output directory if needed
        if generate_report and output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize components
        self.data_loader = DataLoader(cache_enabled=False)  # Disable cache for updates
        self.db_manager = DatabaseManager()

    def run_update(self) -> Dict[str, int]:
        """
        Run the daily update process.

        Returns:
            Dictionary with statistics about the update
        """
        logger.info("Starting daily update pipeline")
        start_time = datetime.now()

        stats = {
            "tables_updated": 0,
            "rows_added": 0,
            "data_sources_processed": 0,
            "errors": 0
        }

        # Get all available data sources
        data_sources = self._get_data_sources()
        logger.info(f"Found {len(data_sources)} data sources to update")

        # Process each data source
        for source in data_sources:
            try:
                logger.info(f"Updating data from source: {source}")
                df = self.data_loader.get_data(source, force_reload=True)

                if df.empty:
                    logger.warning(f"Empty data received from source: {source}")
                    continue

                rows_added = self._update_database(source, df)
                stats["rows_added"] += rows_added
                stats["tables_updated"] += 1
                stats["data_sources_processed"] += 1

                logger.info(f"Added {rows_added} rows to {source} table")

                # Generate a report if requested
                if self.generate_report:
                    self._generate_source_report(source, df)

            except Exception as e:
                logger.error(f"Error updating {source}: {str(e)}")
                stats["errors"] += 1

        # Log execution summary
        log_execution_time(logger, start_time, "Daily update")
        logger.info(f"Update summary: {stats}")

        # Generate a final summary report
        if self.generate_report:
            self._generate_summary_report(stats)

        return stats

    def _get_data_sources(self) -> List[str]:
        """Get the list of data sources to update."""
        # Get all available sources from the data loader
        from config.urls import URLS
        return list(URLS.keys())

    def _update_database(self, source: str, df: pd.DataFrame) -> int:
        """
        Update the database with new data for a source.

        Args:
            source: Name of the data source
            df: DataFrame with new data

        Returns:
            Number of rows added
        """
        # Add metadata
        df_copy = df.copy()
        df_copy['update_date'] = datetime.now().isoformat()
        df_copy['source'] = source

        # Use a unique table name for raw data
        table_name = f"raw_{source}_data"

        try:
            with self.db_manager:
                # Check if the table exists
                table_exists = self.db_manager.table_exists(table_name)

                if table_exists:
                    # Get existing data to check for new records
                    existing_df = self.db_manager.execute_query(f"SELECT * FROM {table_name}")

                    # Identify new records (this is a simplified approach)
                    if "Player" in df_copy.columns and "Player" in existing_df.columns:
                        # Use Player names to identify new records
                        existing_players = set(existing_df["Player"].unique())
                        all_players = set(df_copy["Player"].unique())
                        new_players = all_players - existing_players

                        if new_players:
                            # Filter to only new players
                            new_records = df_copy[df_copy["Player"].isin(new_players)]

                            # Insert new records
                            if not new_records.empty:
                                self.db_manager.insert_dataframe(new_records, table_name)
                                return len(new_records)
                        return 0
                    else:
                        # If we can't identify by Player, just append all
                        self.db_manager.insert_dataframe(df_copy, table_name)
                        return len(df_copy)
                else:
                    # Create new table with all records
                    self.db_manager.insert_dataframe(df_copy, table_name)
                    return len(df_copy)
        except Exception as e:
            logger.error(f"Database update error for {source}: {str(e)}")
            return 0

    def _generate_source_report(self, source: str, df: pd.DataFrame) -> None:
        """
        Generate a report for a specific data source.

        Args:
            source: Name of the data source
            df: DataFrame with the data
        """
        if not self.output_dir:
            return

        try:
            # Create a directory for the source if it doesn't exist
            source_dir = os.path.join(self.output_dir, source)
            if not os.path.exists(source_dir):
                os.makedirs(source_dir)

            # Generate timestamp for the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(source_dir, f"{source}_update_{timestamp}.csv")

            # Save the data
            df.to_csv(report_path, index=False)
            logger.info(f"Generated report for {source} at {report_path}")

        except Exception as e:
            logger.error(f"Error generating report for {source}: {str(e)}")

    def _generate_summary_report(self, stats: Dict[str, int]) -> None:
        """
        Generate a summary report for the update.

        Args:
            stats: Dictionary with update statistics
        """
        if not self.output_dir:
            return

        try:
            # Generate timestamp for the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.output_dir, f"update_summary_{timestamp}.md")

            # Create report content
            report = [
                "# Soccer Data Update Summary\n",
                f"Update time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                "## Statistics\n"
            ]

            for key, value in stats.items():
                report.append(f"- **{key.replace('_', ' ').title()}**: {value}")

            report.append("\n## Updated Data Sources\n")

            # List all updated sources
            sources = self._get_data_sources()
            for source in sources:
                status = "✅ Updated" if stats["errors"] == 0 else "⚠️ Partial update"
                report.append(f"- {source}: {status}")

            # Write report to file
            with open(report_path, 'w') as f:
                f.write("\n".join(report))

            logger.info(f"Generated update summary report at {report_path}")

        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")


def run_daily_update(
    output_dir: Optional[str] = "reports/daily_updates",
    generate_report: bool = True
) -> Dict[str, int]:
    """
    Run the daily update process.

    Args:
        output_dir: Directory to save reports
        generate_report: Whether to generate report files

    Returns:
        Dictionary with update statistics
    """
    pipeline = DailyUpdatePipeline(
        output_dir=output_dir,
        generate_report=generate_report
    )

    return pipeline.run_update()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run daily soccer data update")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/daily_updates",
        help="Directory to save reports"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable report generation"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the update
    stats = run_daily_update(
        output_dir=args.output_dir,
        generate_report=not args.no_report
    )

    # Print summary
    print(f"Update complete: {stats['tables_updated']} tables updated, {stats['rows_added']} rows added")
