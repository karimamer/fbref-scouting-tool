"""
Data loading functions for the soccer analysis application.
"""
from typing import Dict, List, Optional, Union
import logging
import pandas as pd

from config.urls import URLS
from config.settings import DEFAULT_ANALYSIS_PARAMS

logger = logging.getLogger(__name__)

def read_from_html(
    url: str,
    fallback_url: Optional[str] = None,
    silent: bool = False
) -> pd.DataFrame:
    """
    Read data from an HTML source, with error handling and logging.

    Args:
        url: URL or file path to read from
        fallback_url: URL to use if primary URL fails
        silent: If True, suppress error messages

    Returns:
        DataFrame with the loaded data
    """
    try:
        logger.info(f"Loading data from {url}")
        df = pd.read_html(url)[0]

        # Process column names to handle multi-level headers
        column_lst = list(df.columns)
        for index in range(len(column_lst)):
            column_lst[index] = column_lst[index][1]

        df.columns = column_lst

        # Remove duplicate player rows
        df.drop(df[df["Player"] == "Player"].index, inplace=True)

        # Convert empty cells to "0" and set index
        df = df.fillna("0")
        df.set_index("Rk", drop=True, inplace=True)

        # Process competition and nation columns if they exist
        try:
            if "Comp" in df.columns:
                df["Comp"] = df["Comp"].apply(lambda x: " ".join(x.split()[1:]))
            if "Nation" in df.columns:
                df["Nation"] = df["Nation"].astype(str)
                df["Nation"] = df["Nation"].apply(lambda x: x.split()[-1])
        except ValueError as e:
            if not silent:
                logger.warning(f"Error processing columns in {url}: {str(e)}")

        # Convert numeric columns
        df = df.apply(pd.to_numeric, errors="ignore")
        return df

    except Exception as e:
        logger.error(f"Error loading data from {url}: {str(e)}")

        if fallback_url:
            logger.info(f"Attempting to load from fallback URL: {fallback_url}")
            return read_from_html(fallback_url, silent=silent)
        else:
            if not silent:
                logger.error(f"Failed to load data and no fallback provided")
            return pd.DataFrame()  # Return empty DataFrame on failure


class DataLoader:
    """
    Unified interface for loading and processing player statistics.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the data loader.

        Args:
            cache_enabled: If True, cache loaded dataframes to avoid repeat API calls
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_data(
        self,
        stat_type: str,
        url: Optional[str] = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load data for a specific stat type, using cache if available.

        Args:
            stat_type: Type of statistics to load (e.g., 'defense', 'passing')
            url: Optional custom URL to use instead of default
            force_reload: If True, bypass cache and load fresh data

        Returns:
            DataFrame with the requested statistics
        """
        # Check cache first if enabled and not forced to reload
        cache_key = f"{stat_type}_{url or URLS.get(stat_type, '')}"
        if self.cache_enabled and not force_reload and cache_key in self._cache:
            logger.debug(f"Using cached data for {stat_type}")
            return self._cache[cache_key].copy()

        # Determine URL to use
        data_url = url if url else URLS.get(stat_type)
        if not data_url:
            logger.error(f"No URL found for stat type: {stat_type}")
            return pd.DataFrame()

        # Load the data
        df = read_from_html(data_url)

        # Cache the result if enabled
        if self.cache_enabled:
            self._cache[cache_key] = df.copy()

        return df

    def get_all_stats(
        self,
        positions: Optional[List[str]] = None,
        min_90s: Optional[float] = None,
        max_age: Optional[int] = None,
        force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all statistics types with consistent filtering.

        Args:
            positions: List of position strings to filter for
            min_90s: Minimum number of 90-minute periods played
            max_age: Maximum player age to include
            force_reload: If True, bypass cache and load fresh data

        Returns:
            Dictionary with stat type as key and filtered DataFrame as value
        """
        # Use default parameters if none provided
        if positions is None:
            positions = DEFAULT_ANALYSIS_PARAMS["positions"]
        if min_90s is None:
            min_90s = DEFAULT_ANALYSIS_PARAMS["min_90s"]
        if max_age is None:
            max_age = DEFAULT_ANALYSIS_PARAMS["max_age"]

        # Load each stat type
        stats: Dict[str, pd.DataFrame] = {}
        for stat_type in URLS.keys():
            df = self.get_data(stat_type, force_reload=force_reload)

            # Apply consistent filtering across all stat types
            from src.data.processors import process_player_stats
            filtered_df = process_player_stats(
                df,
                positions=positions,
                min_90s=min_90s,
                max_age=max_age
            )

            stats[stat_type] = filtered_df

        return stats
