"""
Unit tests for the metrics module.
"""
import unittest
import pandas as pd
import numpy as np

from src.analysis.metrics import normalize_metric, calculate_weighted_score


class TestMetrics(unittest.TestCase):
    """Test cases for the metrics module."""

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        self.test_df = pd.DataFrame({
            'Player': ['A', 'B', 'C', 'D', 'E'],
            'metric1': [10, 20, 30, 40, 50],
            'metric2': [5, 2, 8, 1, 9],
            'metric3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })

    def test_normalize_metric_robust(self):
        """Test normalize_metric with robust method."""
        # Test with a simple series
        series = pd.Series([10, 20, 30, 40, 50])
        result = normalize_metric(series, method='robust')

        # Check that values are normalized to 0-1 range
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

        # Check that order is preserved
        self.assertTrue(np.all(result.sort_values().index == series.sort_values().index))

    def test_normalize_metric_minmax(self):
        """Test normalize_metric with minmax method."""
        series = pd.Series([10, 20, 30, 40, 50])
        result = normalize_metric(series, method='minmax')

        # First value should be 0, last should be 1
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[4], 1.0)

    def test_normalize_metric_zscore(self):
        """Test normalize_metric with zscore method."""
        # Series with standard distribution
        series = pd.Series([1, 2, 3, 4, 5])
        result = normalize_metric(series, method='zscore')

        # Check that mean is 0 and std is 1 (approximately)
        self.assertAlmostEqual(result.mean(), 0, delta=0.001)
        self.assertAlmostEqual(result.std(), 1, delta=0.001)

    def test_normalize_metric_empty(self):
        """Test normalize_metric with empty series."""
        series = pd.Series([])
        result = normalize_metric(series)
        self.assertTrue(result.empty)

    def test_calculate_weighted_score(self):
        """Test calculate_weighted_score function."""
        # Define metrics and weights
        metrics = {
            'metric1': 0.5,
            'metric2': 0.3,
            'metric3': 0.2
        }

        # Calculate weighted score
        result_df = calculate_weighted_score(self.test_df, metrics, 'test_score')

        # Check that the score column was added
        self.assertIn('test_score', result_df.columns)

        # Check that all rows have a score
        self.assertEqual(len(result_df['test_score']), len(self.test_df))

        # Check that normalized columns were added
        for metric in metrics.keys():
            self.assertIn(f"{metric}_norm", result_df.columns)

    def test_calculate_weighted_score_missing_metrics(self):
        """Test calculate_weighted_score with missing metrics."""
        # Define metrics with one that doesn't exist
        metrics = {
            'metric1': 0.5,
            'metric2': 0.3,
            'missing_metric': 0.2
        }

        # Calculate weighted score
        result_df = calculate_weighted_score(self.test_df, metrics, 'test_score')

        # Check that the score column was added
        self.assertIn('test_score', result_df.columns)

        # Check that only existing metrics were normalized
        self.assertIn('metric1_norm', result_df.columns)
        self.assertIn('metric2_norm', result_df.columns)
        self.assertNotIn('missing_metric_norm', result_df.columns)

    def test_calculate_weighted_score_empty_df(self):
        """Test calculate_weighted_score with empty DataFrame."""
        empty_df = pd.DataFrame()
        metrics = {'metric1': 1.0}

        result_df = calculate_weighted_score(empty_df, metrics, 'test_score')
        self.assertTrue(result_df.empty)


if __name__ == '__main__':
    unittest.main()
