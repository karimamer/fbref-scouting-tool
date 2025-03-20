import unittest
from unittest.mock import patch
import pandas as pd

from src.data.loaders import read_from_html, DataLoader


class TestDataLoaders(unittest.TestCase):
    """Test cases for the data loaders module."""

    def setUp(self):
        """Set up test data."""
        # Create a mock dataframe for testing
        self.mock_df = pd.DataFrame({
            'Player': ['Player1', 'Player2', 'Player'],
            'Pos': ['FW', 'MF', 'Player'],
            'Age': ['24-104', '30-200', 'Age'],
            '90s': [10, 15, '90s'],
            'Gls': [5, 3, 'Gls']
        })

        # Create a processed dataframe (as it would be after processing)
        self.processed_df = self.mock_df.copy()
        self.processed_df.drop(self.processed_df[self.processed_df['Player'] == 'Player'].index, inplace=True)

    @patch('pandas.read_html')
    def test_read_from_html(self, mock_read_html):
        """Test read_from_html function."""
        # Setup the mock
        mock_read_html.return_value = [self.mock_df]

        # Call the function
        result = read_from_html('mock_url')

        # Verify function was called with the correct URL
        mock_read_html.assert_called_once_with('mock_url')

        # Check that the duplicate player row was dropped
        self.assertEqual(len(result), 2)
        self.assertNotIn('Player', result['Player'].values)

    @patch('src.data.loaders.read_from_html')
    def test_data_loader_get_data(self, mock_read_from_html):
        """Test DataLoader.get_data method."""
        # Setup the mock
        mock_read_from_html.return_value = self.processed_df

        # Create loader and get data
        loader = DataLoader(cache_enabled=True)
        result = loader.get_data('test_stat')

        # Verify the function was called correctly
        mock_read_from_html.assert_called_once()

        # Check result
        self.assertEqual(len(result), 2)

        # Check caching
        # Call again and ensure the function is not called a second time
        result2 = loader.get_data('test_stat')
        self.assertEqual(mock_read_from_html.call_count, 1)

        # Force reload should call the function again
        result3 = loader.get_data('test_stat', force_reload=True)
        self.assertEqual(mock_read_from_html.call_count, 2)

    @patch('src.data.loaders.read_from_html')
    def test_data_loader_cache_disabled(self, mock_read_from_html):
        """Test DataLoader with cache disabled."""
        # Setup the mock
        mock_read_from_html.return_value = self.processed_df

        # Create loader with caching disabled
        loader = DataLoader(cache_enabled=False)

        # Get data multiple times and check that the function is called each time
        result1 = loader.get_data('test_stat')
        self.assertEqual(mock_read_from_html.call_count, 1)

        result2 = loader.get_data('test_stat')
        self.assertEqual(mock_read_from_html.call_count, 2)

    @patch('src.data.loaders.read_from_html')
    @patch('src.data.loaders.process_player_stats')
    def test_get_all_stats(self, mock_process, mock_read_from_html):
        """Test get_all_stats method."""
        # Setup the mocks
        mock_read_from_html.return_value = self.processed_df
        mock_process.return_value = self.processed_df

        # Create loader and get all stats
        loader = DataLoader(cache_enabled=True)

        with patch.object(loader, 'get_data', return_value=self.processed_df) as mock_get_data:
            results = loader.get_all_stats(positions=['FW', 'MF'], min_90s=5, max_age=30)

            # Check that we got a dictionary of results
            self.assertIsInstance(results, dict)

            # Each value should be a dataframe
            for df in results.values():
                self.assertIsInstance(df, pd.DataFrame)

            # Process function should be called for each data source
            self.assertEqual(mock_process.call_count, len(results))


if __name__ == '__main__':
    unittest.main()
