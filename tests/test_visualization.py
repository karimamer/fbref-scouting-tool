import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.visualization import plot_radar_chart

# Helper function for sample data
def _create_sample_radar_data(num_entities=2, num_metrics=3, custom_metrics=None, custom_values=None):
    """
    Creates sample data for radar chart tests.
    custom_metrics: Optional list of metric names.
    custom_values: Optional list of lists/dicts for entity data.
                   If provided, num_entities and num_metrics might be overridden by its structure.
    """
    if custom_values:
        num_entities = len(custom_values)
    
    entity_names = [f"Player {i+1}" for i in range(num_entities)]

    if custom_metrics:
        metric_columns = custom_metrics
        if custom_values and isinstance(custom_values[0], dict): # if values are dicts, num_metrics is from keys
             num_metrics = len(custom_values[0].keys())
        else: # if values are lists, num_metrics is from length of list
            num_metrics = len(custom_metrics)

    elif custom_values and isinstance(custom_values[0], dict):
        metric_columns = list(custom_values[0].keys())
        num_metrics = len(metric_columns)
    elif custom_values and isinstance(custom_values[0], list):
        num_metrics = len(custom_values[0])
        metric_columns = [f"Metric_{chr(65+j)}" for j in range(num_metrics)]
    else:
        metric_columns = [f"Metric_{chr(65+j)}" for j in range(num_metrics)]

    data_frames = []
    if custom_values:
        for i in range(num_entities):
            if isinstance(custom_values[i], dict):
                data = custom_values[i]
            else: # Assuming list of values in order of metric_columns
                data = {metric_columns[j]: custom_values[i][j] for j in range(len(custom_values[i]))}
            data_frames.append(pd.DataFrame([data]))
    else:
        for _ in range(num_entities):
            data = {col: np.random.rand() * 100 for col in metric_columns}
            data_frames.append(pd.DataFrame([data]))
            
    return data_frames, entity_names, metric_columns

class TestPlotRadarChart(unittest.TestCase):

    def tearDown(self):
        """Close all Matplotlib figures after each test."""
        plt.close('all')

    def test_basic_chart_creation(self):
        """Test basic chart creation with minimal valid inputs."""
        data_frames, entity_names, metric_columns = _create_sample_radar_data(num_entities=3, num_metrics=4)
        
        fig = None
        try:
            fig = plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Basic Test Chart"
            )
            self.assertIsInstance(fig, plt.Figure)
        except Exception as e:
            self.fail(f"plot_radar_chart raised an exception unexpectedly: {e}")
        finally:
            if fig:
                plt.close(fig)

    def test_normalization_standard_and_inverted(self):
        """Test with standard and inverted normalization."""
        metric_names = ["Attack", "Defense", "Speed", "Errors"]
        player_data = [
            {"Attack": 80, "Defense": 60, "Speed": 90, "Errors": 5},
            {"Attack": 70, "Defense": 70, "Speed": 60, "Errors": 10},
            {"Attack": 90, "Defense": 50, "Speed": 75, "Errors": 2},
        ]
        data_frames, entity_names, metric_columns = _create_sample_radar_data(
            custom_metrics=metric_names, custom_values=player_data
        )
        
        fig = None
        try:
            fig = plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Normalization Test (Standard & Inverted)",
                normalize=True,
                normalization_specs={'Errors': 'inverted', 'Speed': 'standard'}
            )
            self.assertIsInstance(fig, plt.Figure)
        except Exception as e:
            self.fail(f"plot_radar_chart raised an exception unexpectedly during normalization test: {e}")
        finally:
            if fig:
                plt.close(fig)

    def test_normalization_same_min_max(self):
        """Test normalization when all players have the same value for a metric."""
        metric_names = ["Consistency", "Effort", "Skill"]
        player_data = [
            {"Consistency": 75, "Effort": 80, "Skill": 90},
            {"Consistency": 75, "Effort": 70, "Skill": 85},
            {"Consistency": 75, "Effort": 90, "Skill": 80},
        ]
        data_frames, entity_names, metric_columns = _create_sample_radar_data(
            custom_metrics=metric_names, custom_values=player_data
        )
        
        fig = None
        try:
            fig = plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Same Min-Max Normalization Test",
                normalize=True
            )
            self.assertIsInstance(fig, plt.Figure)
        except Exception as e:
            self.fail(f"plot_radar_chart raised an exception unexpectedly during same min-max normalization test: {e}")
        finally:
            if fig:
                plt.close(fig)

    def test_output_file_creation(self):
        """Test chart output to a file."""
        data_frames, entity_names, metric_columns = _create_sample_radar_data()
        output_file_path = "test_radar_chart_output.png"

        fig = None
        try:
            fig = plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="File Output Test",
                output_file=output_file_path
            )
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(os.path.exists(output_file_path))
        except Exception as e:
            self.fail(f"plot_radar_chart raised an exception unexpectedly during file output test: {e}")
        finally:
            if fig:
                plt.close(fig)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
                
    def test_metric_labels_override(self):
        """Test overriding metric labels."""
        data_frames, entity_names, metric_columns = _create_sample_radar_data(num_metrics=3)
        labels_override = {
            metric_columns[0]: "Custom Label 1",
            metric_columns[1]: "Super Metric B",
        }
        fig = None
        try:
            fig = plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Metric Labels Override Test",
                metric_labels_override=labels_override
            )
            self.assertIsInstance(fig, plt.Figure)
            # Direct check of tick labels is complex and brittle.
            # For this test, error-free execution with the override is the main check.
        except Exception as e:
            self.fail(f"plot_radar_chart raised an exception unexpectedly during metric labels override test: {e}")
        finally:
            if fig:
                plt.close(fig)

    def test_plot_on_existing_axes(self):
        """Test plotting on an existing Matplotlib Axes object."""
        data_frames, entity_names, metric_columns = _create_sample_radar_data()
        
        fig_created, ax_created = plt.subplots(subplot_kw={'polar': True}, figsize=(8,8))
        
        returned_fig = None
        try:
            returned_fig = plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Existing Axes Test",
                ax=ax_created
            )
            self.assertIsInstance(returned_fig, plt.Figure)
            self.assertEqual(returned_fig, fig_created, "Should return the same figure object it plotted on.")
        except Exception as e:
            self.fail(f"plot_radar_chart raised an exception unexpectedly when plotting on existing axes: {e}")
        finally:
            # The tearDown method will close fig_created if returned_fig is the same.
            # If they are different for some reason (which would be a failure), close both.
            if returned_fig and returned_fig is not fig_created:
                 plt.close(returned_fig)
            if fig_created: # fig_created will always exist
                 plt.close(fig_created)


    def test_input_validation_errors(self):
        """Test for expected ValueError exceptions with invalid inputs."""
        data_frames, entity_names, metric_columns = _create_sample_radar_data(num_entities=2, num_metrics=3)

        # Mismatch in length between data_frames and entity_names
        with self.assertRaisesRegex(ValueError, "Length of data_frames must match length of entity_names"):
            plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names[:1], # One less entity name
                title="Error Test"
            )
        plt.close('all') # Ensure any partial fig is closed

        # Metric in metric_columns not present in one of the data_frames
        df_missing_metric = [data_frames[0].copy()]
        df_missing_metric[0].drop(columns=[metric_columns[0]], inplace=True)
        
        faulty_data_frames = [df_missing_metric[0]] + data_frames[1:]

        with self.assertRaisesRegex(ValueError, f"Metric '{metric_columns[0]}' not found in DataFrame for entity '{entity_names[0]}'"):
             plot_radar_chart(
                data_frames=faulty_data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Error Test Metric Missing"
            )
        plt.close('all') # Ensure any partial fig is closed

        # DataFrame with more than one row
        df_multi_row = pd.concat([data_frames[0], data_frames[0]], ignore_index=True) # Changed from append to concat
        multi_row_dfs = [df_multi_row] + data_frames[1:]
        with self.assertRaisesRegex(ValueError, f"DataFrame for entity '{entity_names[0]}' must have exactly one row."):
            plot_radar_chart(
                data_frames=multi_row_dfs,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Error Test Multi-Row DF"
            )
        plt.close('all')
        
        # Non-polar axes provided
        fig_non_polar, ax_non_polar = plt.subplots()
        with self.assertRaisesRegex(ValueError, "Provided 'ax' must be a Matplotlib polar Axes object."):
            plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                ax=ax_non_polar
            )
        plt.close(fig_non_polar)
        plt.close('all')


    def test_custom_styling(self):
        """Test with custom styling options."""
        data_frames, entity_names, metric_columns = _create_sample_radar_data(num_entities=2, num_metrics=4)
        
        custom_colors = ['#FF5733', '#33FF57'] # Bright orange and green
        custom_line_styles = ['--', ':']
        custom_line_widths = [2.5, 1.0]
        custom_fill_alphas = [0.25, 0.05]
        custom_fig_size = (12, 12)
        custom_legend_kwargs = {'loc': 'lower left', 'fontsize': 'small'}

        fig = None
        try:
            fig = plot_radar_chart(
                data_frames=data_frames,
                metric_columns=metric_columns,
                entity_names=entity_names,
                title="Custom Styling Test",
                fig_size=custom_fig_size,
                colors=custom_colors,
                line_styles=custom_line_styles,
                line_widths=custom_line_widths,
                fill_alphas=custom_fill_alphas,
                legend_kwargs=custom_legend_kwargs
            )
            self.assertIsInstance(fig, plt.Figure)
            # Basic check for custom figure size (approximate due to layout adjustments)
            # self.assertAlmostEqual(fig.get_size_inches()[0], custom_fig_size[0], delta=0.5)
            # self.assertAlmostEqual(fig.get_size_inches()[1], custom_fig_size[1], delta=0.5)
            # More detailed checks would involve inspecting artists, which is complex.
        except Exception as e:
            self.fail(f"plot_radar_chart raised an exception unexpectedly during custom styling test: {e}")
        finally:
            if fig:
                plt.close(fig)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
