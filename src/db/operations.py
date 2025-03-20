import pandas as pd
import duckdb
import uuid
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from config.settings import DATABASE

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manager for database operations with context management.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the database file, uses config if None
        """
        self.db_path = db_path or DATABASE["path"]
        self.connection = None

    def __enter__(self):
        """Context manager entry point."""
        self.connection = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if the table exists, False otherwise
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        result = self.connection.execute(
            f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
        ).fetchone()[0]

        return result > 0

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        if_exists: str = 'append'
    ) -> bool:
        """
        Insert a DataFrame into a database table.

        Args:
            df: DataFrame to insert
            table_name: Name of the target table
            metadata: Additional metadata to include with each row
            if_exists: Action if table exists ('append', 'replace', 'fail')

        Returns:
            True if successful, False otherwise
        """
        if df.empty:
            logger.warning(f"Attempted to insert an empty DataFrame into {table_name}")
            return False

        if not self.connection:
            raise RuntimeError("Database connection not established")

        # Add metadata fields
        df_copy = df.copy()
        run_id = str(uuid.uuid4())
        current_time = datetime.utcnow()

        df_copy['run_id'] = run_id
        df_copy['created_at'] = current_time

        # Add any additional metadata
        if metadata:
            for key, value in metadata.items():
                df_copy[key] = value

        try:
            # Check if table exists and handle accordingly
            table_exists = self.table_exists(table_name)

            if table_exists:
                if if_exists == 'replace':
                    self.connection.execute(f"DROP TABLE {table_name}")
                    self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_copy")
                elif if_exists == 'append':
                    self.connection.execute(f"INSERT INTO {table_name} SELECT * FROM df_copy")
                elif if_exists == 'fail':
                    logger.error(f"Table {table_name} already exists and if_exists is set to 'fail'")
                    return False
            else:
                self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_copy")

            self.connection.commit()
            logger.info(f"Successfully inserted {len(df_copy)} rows into {table_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {str(e)}")
            return False

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.

        Args:
            query: SQL query to execute

        Returns:
            DataFrame with query results
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            result = self.connection.execute(query).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()

    def get_tables(self) -> List[str]:
        """
        Get a list of all tables in the database.

        Returns:
            List of table names
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        query = "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        result = self.connection.execute(query).fetchdf()
        return result['table_name'].tolist()

    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a backup of the database.

        Args:
            backup_path: Path for the backup file

        Returns:
            True if successful, False otherwise
        """
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = DATABASE.get("backup_dir", "./backups")
            backup_path = f"{backup_dir}/scouting_backup_{timestamp}.db"

        try:
            if self.connection:
                self.connection.close()
                self.connection = None

            # Create a connection to the backup database
            backup_conn = duckdb.connect(backup_path)

            # Attach the source database
            backup_conn.execute(f"ATTACH '{self.db_path}' AS source")

            # Get all tables from the source database
            tables = backup_conn.execute("SELECT table_name FROM source.information_schema.tables WHERE table_schema='main'").fetchdf()

            # Copy each table
            for table in tables['table_name']:
                backup_conn.execute(f"CREATE TABLE {table} AS SELECT * FROM source.{table}")

            backup_conn.close()

            # Reconnect to the original database
            self.connection = duckdb.connect(self.db_path)

            logger.info(f"Database successfully backed up to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup database: {str(e)}")
            if self.connection is None:
                self.connection = duckdb.connect(self.db_path)
            return False


# Simplified standalone function for quick use
def insert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    db_path: Optional[str] = None
) -> bool:
    """
    Insert a DataFrame into a database table using the context manager.

    Args:
        df: DataFrame to insert
        table_name: Name of the target table
        db_path: Path to the database file

    Returns:
        True if successful, False otherwise
    """
    try:
        with DatabaseManager(db_path) as db:
            return db.insert_dataframe(df, table_name)
    except Exception as e:
        logger.error(f"Error in insert_dataframe: {str(e)}")
        return False
