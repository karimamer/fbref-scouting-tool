import os
import warnings # Added for deprecation warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union

def create_radar_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Comparison",
    max_players: int = 5,
    output_file: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    DEPRECATED: Create a radar chart comparing multiple players across different metrics. 
    Use plot_radar_chart instead for more flexibility and consistent normalization.

    Parameters:
    -----------
    df: DataFrame containing player data
    metrics: List of metrics to include in the comparison
    player_column: Column name that contains player names
    title: Title for the chart
    max_players: Maximum number of players to include
    output_file: If provided, save the figure to this path
    normalize: Whether to normalize metrics to 0-1 scale

    Returns:
    --------
    matplotlib Figure object
    """
    warnings.warn(
        "create_radar_comparison is deprecated and will be removed in a future version. "
        "Use plot_radar_chart instead.",
        FutureWarning,
        stacklevel=2  # Ensures the warning points to the caller of this function
    )
    # Select top players and required columns
    players_df = df.head(max_players).copy()

    # Ensure all metrics exist
    missing_metrics = [m for m in metrics if m not in players_df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metrics in dataframe: {missing_metrics}")

    # Normalize metrics if requested
    if normalize:
        for metric in metrics:
            min_val = players_df[metric].min()
            max_val = players_df[metric].max()
            if max_val > min_val:
                players_df[f"{metric}_norm"] = (players_df[metric] - min_val) / (max_val - min_val)
            else:
                players_df[f"{metric}_norm"] = 0.5  # Default if all values are the same
        plotting_metrics = [f"{m}_norm" for m in metrics]
    else:
        plotting_metrics = metrics

    # Number of metrics
    num_metrics = len(plotting_metrics)

    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(players_df)))

    # Plot each player
    for i, (_, player) in enumerate(players_df.iterrows()):
        values = [player[m] for m in plotting_metrics]
        values += values[:1]  # Close the circle

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=player[player_column])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Set labels
    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    plt.xticks(angles[:-1], metric_labels, size=12)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Add title
    plt.title(title, size=15)

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

def create_scatter_comparison(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    player_column: str = "Player",
    title: str = "Player Comparison",
    labeled_players: int = 10,
    output_file: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot comparing players on two metrics.

    Parameters:
    -----------
    df: DataFrame containing player data
    x_metric: Metric to plot on x-axis
    y_metric: Metric to plot on y-axis
    color_by: Column to use for point coloring
    size_by: Column to use for point sizing
    player_column: Column name that contains player names
    title: Title for the chart
    labeled_players: Number of players to label in the plot
    output_file: If provided, save the figure to this path

    Returns:
    --------
    matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare plot parameters
    plot_kwargs = {
        'x': x_metric,
        'y': y_metric,
        'data': df,
        'alpha': 0.7
    }

    # Add color grouping if specified
    if color_by and color_by in df.columns:
        plot_kwargs['hue'] = color_by

    # Add size variation if specified
    if size_by and size_by in df.columns:
        plot_kwargs['size'] = size_by
        plot_kwargs['sizes'] = (20, 200)

    # Create scatter plot
    sns.scatterplot(**plot_kwargs)

    # Add player labels for top players
    importance_metric = y_metric  # Default to y-axis metric for importance

    for _, player in df.sort_values(importance_metric, ascending=False).head(labeled_players).iterrows():
        plt.text(
            player[x_metric] + (df[x_metric].max() - df[x_metric].min()) * 0.01,
            player[y_metric] + (df[y_metric].max() - df[y_metric].min()) * 0.01,
            player[player_column],
            fontsize=9
        )

    # Format chart
    plt.title(title, fontsize=16)
    plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

def create_bar_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Comparison",
    max_players: int = 10,
    sort_by: Optional[str] = None,
    output_file: Optional[str] = None,
    horizontal: bool = True,
    stacked: bool = False
) -> plt.Figure:
    """
    Create a bar chart comparing players across metrics.

    Parameters:
    -----------
    df: DataFrame containing player data
    metrics: List of metrics to include in the comparison
    player_column: Column name that contains player names
    title: Title for the chart
    max_players: Maximum number of players to include
    sort_by: Metric to sort players by
    output_file: If provided, save the figure to this path
    horizontal: Whether to create horizontal bar chart
    stacked: Whether to create stacked bar chart

    Returns:
    --------
    matplotlib Figure object
    """
    # Select data
    players_df = df.head(max_players).copy()

    # Sort if requested
    if sort_by:
        if sort_by in players_df.columns:
            players_df = players_df.sort_values(sort_by, ascending=False)
        else:
            print(f"Warning: Sort column '{sort_by}' not found in dataframe")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8) if horizontal else (10, 10))

    # Determine plot orientation
    plot_func = players_df.plot.barh if horizontal else players_df.plot.bar

    # Create bar chart
    plot_func(
        x=player_column,
        y=metrics,
        ax=ax,
        stacked=stacked,
        width=0.8,
        alpha=0.8
    )

    # Format chart
    plt.title(title, fontsize=16)
    if horizontal:
        plt.xlabel(', '.join([m.replace('_', ' ').title() for m in metrics]), fontsize=12)
        plt.ylabel("Players", fontsize=12)
    else:
        plt.xlabel("Players", fontsize=12)
        plt.ylabel(', '.join([m.replace('_', ' ').title() for m in metrics]), fontsize=12)

    plt.legend(title='Metrics')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x' if horizontal else 'y')
    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

def create_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    player_column: str = "Player",
    title: str = "Player Metrics Heatmap",
    max_players: int = 15,
    output_file: Optional[str] = None,
    normalize: bool = True,
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Create a heatmap of multiple metrics for multiple players.

    Parameters:
    -----------
    df: DataFrame containing player data
    metrics: List of metrics to include in the heatmap
    player_column: Column name that contains player names
    title: Title for the chart
    max_players: Maximum number of players to include
    output_file: If provided, save the figure to this path
    normalize: Whether to normalize metrics for better comparison
    cmap: Colormap to use

    Returns:
    --------
    matplotlib Figure object
    """
    # Select data
    players_df = df.head(max_players).copy()

    # Set player as index
    players_df = players_df.set_index(player_column)

    # Select only the metrics we want to display
    metrics_df = players_df[metrics].copy()

    # Normalize if requested
    if normalize:
        for col in metrics_df.columns:
            metrics_df[col] = (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        metrics_df,
        annot=True,
        cmap=cmap,
        fmt=".2f" if normalize else ".1f",
        linewidths=0.5,
        ax=ax
    )

    # Format chart
    plt.title(title, fontsize=16)
    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

def create_dashboard(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "visualizations",
    prefix: str = ""
) -> List[str]:
    """
    Create a full dashboard of visualizations from analysis results.

    Parameters:
    -----------
    results: Dictionary of analysis results
    output_dir: Directory to save visualizations
    prefix: Prefix for output filenames

    Returns:
    --------
    List of created visualization filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    created_files = []

    # 1. Versatility radar chart
    if "versatile_players" in results and not results["versatile_players"].empty:
        versatile_df = results["versatile_players"].head(5) # This selects top 5 players as before
        
        # Define metrics for the radar chart, equivalent to the old 'metrics' list
        metric_columns_for_radar = ["passing_score", "possession_score", "defensive_score"]
        if "shooting_score" in versatile_df.columns: # Check if optional metric is present
            metric_columns_for_radar.append("shooting_score")
        
        # Ensure all selected metrics are actually present in the DataFrame columns
        # This prevents errors if a defined metric is missing from versatile_df
        metric_columns_for_radar = [m for m in metric_columns_for_radar if m in versatile_df.columns]

        # Determine the player name column. The old call didn't specify player_column,
        # so create_radar_comparison would have used its default "Player".
        player_name_col = "Player" 
        
        if not metric_columns_for_radar:
            print(f"Warning: No valid metrics found for versatility radar chart ({prefix}versatility_radar.png). Skipping.")
        elif player_name_col not in versatile_df.columns:
            # If "Player" column is not found, try "Name" as a common alternative.
            if "Name" in versatile_df.columns:
                player_name_col = "Name"
            else:
                print(f"Warning: Player column '{player_name_col}' or 'Name' not found in versatile_players DataFrame for {prefix}versatility_radar.png. Skipping chart.")
                # To prevent further errors, we effectively skip this chart if no player name column is found.
                # This is done by not proceeding to the data_frames_list creation.
        
        # Proceed only if we have metrics and a valid player name column
        if metric_columns_for_radar and player_name_col in versatile_df.columns:
            # Prepare data for plot_radar_chart:
            # 1. Create a list of single-row DataFrames, one for each player, containing the selected metrics.
            # 2. Create a list of player names corresponding to these DataFrames.
            data_frames_list = []
            entity_names_list = []

            for _, row in versatile_df.iterrows():
                # Extract the player's data for the specified radar metrics.
                player_metrics_data = row[metric_columns_for_radar]
                # Create a single-row DataFrame. Using .values and specifying columns ensures correct structure.
                player_data_df = pd.DataFrame([player_metrics_data.values], columns=metric_columns_for_radar)
                data_frames_list.append(player_data_df)
                entity_names_list.append(row[player_name_col])
            
            if data_frames_list: # Ensure data was actually prepared
                output_file = os.path.join(output_dir, f"{prefix}versatility_radar.png")
                
                # Call the new plot_radar_chart function
                plot_radar_chart(
                    data_frames=data_frames_list,
                    metric_columns=metric_columns_for_radar, 
                    entity_names=entity_names_list,
                    title="Top Players by Versatility", # Same title as before
                    output_file=output_file
                    # normalize=True is the default in plot_radar_chart.
                    # create_radar_comparison also defaulted to normalize=True.
                )
                created_files.append(output_file)
            else:
                # This condition might be hit if versatile_df was empty or became empty after filtering,
                # though the initial checks should largely prevent this.
                print(f"Warning: No data to plot for versatility radar chart ({prefix}versatility_radar.png) after processing. Skipping.")

    # 2. Progressive actions comparison
    prog_metrics = ["overall_progressors", "top_carriers", "top_passers"]
    for metric in prog_metrics:
        if metric in results and not results[metric].empty:
            df = results[metric].head(10)
            if "total_progression_score" in df.columns:
                sort_col = "total_progression_score"
            else:
                sort_col = None

            output_file = os.path.join(output_dir, f"{prefix}{metric}_comparison.png")
            create_bar_comparison(
                df,
                metrics=["carrying_progression_score", "passing_progression_score", "receiving_progression_score"]
                if all(col in df.columns for col in ["carrying_progression_score", "passing_progression_score", "receiving_progression_score"])
                else ["PrgC", "PrgP", "PrgR"],
                title=f"Top {metric.replace('_', ' ').title()}",
                sort_by=sort_col,
                output_file=output_file,
                stacked=True
            )
            created_files.append(output_file)

    # 3. Possession Impact scatter
    if "possession_impact" in results and not results["possession_impact"].empty:
        xpi_df = results["possession_impact"].copy()

        # Create position column for coloring
        if "Pos" in xpi_df.columns:
            xpi_df["Position"] = xpi_df["Pos"].apply(
                lambda x: "Defender" if "DF" in x
                else "Midfielder" if "MF" in x
                else "Forward" if "FW" in x
                else "Other"
            )
            color_by = "Position"
        else:
            color_by = None

        # Choose size variable if available
        size_by = "touches_90" if "touches_90" in xpi_df.columns else None

        output_file = os.path.join(output_dir, f"{prefix}possession_impact.png")
        create_scatter_comparison(
            xpi_df,
            x_metric="90s",
            y_metric="xPI",
            color_by=color_by,
            size_by=size_by,
            title="Expected Possession Impact (xPI)",
            output_file=output_file
        )
        created_files.append(output_file)

    # 4. Midfielder clusters
    if "midfielder_clusters" in results and "cluster" in results["midfielder_clusters"].columns:
        mf_df = results["midfielder_clusters"].copy()

        if len(mf_df) >= 10 and "PrgP" in mf_df.columns and "PrgC" in mf_df.columns:
            output_file = os.path.join(output_dir, f"{prefix}midfielder_clusters.png")
            create_scatter_comparison(
                mf_df,
                x_metric="PrgP",
                y_metric="PrgC",
                color_by="cluster",
                title="Midfielder Clusters: Progressive Passes vs Carries",
                output_file=output_file
            )
            created_files.append(output_file)

    # 5. Heatmap of top versatile players
    if "versatile_players" in results and not results["versatile_players"].empty:
        versatile_df = results["versatile_players"].head(15)

        # Select metrics for heatmap
        heatmap_metrics = [
            col for col in ["passing_score", "possession_score", "defensive_score",
                           "shooting_score", "versatility_score", "adjusted_versatility"]
            if col in versatile_df.columns
        ]

        if len(heatmap_metrics) >= 3:
            output_file = os.path.join(output_dir, f"{prefix}versatility_heatmap.png")
            create_heatmap(
                versatile_df,
                metrics=heatmap_metrics,
                title="Top Players Versatility Breakdown",
                output_file=output_file
            )
            created_files.append(output_file)

    return created_files


def plot_radar_chart(
    data_frames: List[pd.DataFrame],
    metric_columns: List[str],
    entity_names: List[str],
    title: str = "Radar Chart Comparison",
    normalize: bool = True,
    normalization_specs: Optional[Dict[str, str]] = None,
    fig_size: Tuple[int, int] = (10, 10),
    colors: Optional[List[str]] = None,
    line_styles: Optional[List[str]] = None,
    line_widths: Optional[List[float]] = None,
    fill_alphas: Optional[List[float]] = None,
    metric_labels_override: Optional[Dict[str, str]] = None,
    legend_kwargs: Optional[Dict] = None,
    output_file: Optional[str] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Generate a radar chart comparing multiple entities across specified metrics.

    This function allows for flexible customization of the radar chart, including normalization options
    per metric (e.g., inverting scales for metrics where lower is better).

    Parameters:
    -----------
    data_frames : List[pd.DataFrame]
        A list of pandas DataFrames. Each DataFrame represents one entity (e.g., player, team)
        and *must* contain exactly one row of data. The columns of these DataFrames should
        correspond to the metrics being plotted.
        Example: `[pd.DataFrame({'MetricA': [10], 'MetricB': [20]}), pd.DataFrame({'MetricA': [15], 'MetricB': [25]})]`
    metric_columns : List[str]
        A list of strings specifying the column names from the DataFrames in `data_frames`
        to be used as axes (metrics) for the radar chart. The order in this list determines
        the order of axes on the chart.
    entity_names : List[str]
        A list of strings representing the names of the entities being compared. The order of
        names in this list must correspond to the order of DataFrames in `data_frames`.
        These names will be used in the legend.
    title : str, optional
        The title for the radar chart. Defaults to "Radar Chart Comparison".
    normalize : bool, optional
        If True (default), metric values are normalized to a 0-1 scale across all entities for each metric.
        This is crucial when comparing metrics with different units or scales. Normalization is
        performed by scaling based on the minimum and maximum values found for that metric
        across *all* provided `data_frames`. If False, raw values are used (not recommended
        for metrics with vastly different scales).
    normalization_specs : Optional[Dict[str, str]], optional
        A dictionary to specify custom normalization behavior for individual metrics. This is
        applied only if `normalize` is True.
        Keys should be metric column names (from `metric_columns`).
        Values should be strings indicating the normalization type:
        - 'standard': Standard min-max normalization: `(value - min) / (max - min)`. This is the
                      default for metrics not specified or if `normalization_specs` is None.
        - 'inverted': Inverted min-max normalization: `1 - ((value - min) / (max - min))`.
                      Useful for metrics where lower values are better (e.g., rank, distance error).
        Example: `{'Rank': 'inverted', 'Score': 'standard'}`. Defaults to None.
    fig_size : Tuple[int, int], optional
        Tuple specifying the figure size (width, height) in inches if a new figure is created
        (i.e., if `ax` is None). Defaults to (10, 10).
    colors : Optional[List[str]], optional
        A list of color strings for each entity's plot. If None, uses Matplotlib's default
        color cycle. The list length should match the number of entities.
    line_styles : Optional[List[str]], optional
        A list of line style strings (e.g., '-', '--', ':') for each entity's plot. If None,
        uses a solid line for all. The list length should match the number of entities.
    line_widths : Optional[List[float]], optional
        A list of line widths for each entity's plot. If None, uses a default width (e.g., 1.5).
        The list length should match the number of entities.
    fill_alphas : Optional[List[float]], optional
        A list of alpha (transparency) values (0.0 to 1.0) for the fill of each entity's plot.
        If None, uses a default alpha (e.g., 0.1). The list length should match the number of entities.
    metric_labels_override : Optional[Dict[str, str]], optional
        A dictionary to override the display names for metrics on the chart axes.
        Keys should be metric column names, and values are the desired display names.
        Example: `{'shots_pg': 'Shots per Game'}`. Defaults to None (uses title-cased column names).
    legend_kwargs : Optional[Dict], optional
        A dictionary of keyword arguments to be passed to `ax.legend()`.
        Example: `{'loc': 'lower center', 'ncol': 3}`. Defaults to a standard legend configuration.
    output_file : Optional[str], optional
        The file path to save the figure. If None, the figure is not saved. Defaults to None.
    dpi : int, optional
        Dots per inch for the saved figure. Used if `output_file` is provided. Defaults to 300.
    ax : Optional[plt.Axes], optional
        A Matplotlib Axes object with a polar projection to plot on. If None (default),
        a new figure and polar axes are created.

    Returns:
    --------
    plt.Figure
        The Matplotlib Figure object containing the radar chart.

    Raises:
    -------
    ValueError
        If input parameters are inconsistent (e.g., mismatched lengths of `data_frames` and
        `entity_names`; DataFrames with incorrect shape (not single-row); missing `metric_columns`
        in any DataFrame; or if a provided `ax` is not a Matplotlib polar Axes object).
    """
    # Input Validation
    if len(data_frames) != len(entity_names):
        raise ValueError("Length of data_frames must match length of entity_names.")
    if colors and len(colors) != len(data_frames):
        raise ValueError("Length of colors must match length of data_frames if provided.")
    if line_styles and len(line_styles) != len(data_frames):
        raise ValueError("Length of line_styles must match length of data_frames if provided.")
    if line_widths and len(line_widths) != len(data_frames):
        raise ValueError("Length of line_widths must match length of data_frames if provided.")
    if fill_alphas and len(fill_alphas) != len(data_frames):
        raise ValueError("Length of fill_alphas must match length of data_frames if provided.")

    for i, df_entity in enumerate(data_frames): # Renamed df to df_entity for clarity
        if not isinstance(df_entity, pd.DataFrame):
            raise ValueError(f"Item at index {i} in data_frames is not a pandas DataFrame.")
        if df_entity.shape[0] != 1:
            raise ValueError(f"DataFrame for entity '{entity_names[i]}' must have exactly one row. Found shape {df_entity.shape}.")
        for metric in metric_columns:
            if metric not in df_entity.columns:
                raise ValueError(f"Metric '{metric}' not found in DataFrame for entity '{entity_names[i]}'. Available columns: {list(df_entity.columns)}")

    # Figure and Axes
    if ax:
        if not hasattr(ax, 'name') or ax.name != 'polar': # More robust check for polar axes
            raise ValueError("Provided 'ax' must be a Matplotlib polar Axes object.")
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))

    num_metrics = len(metric_columns)
    processed_values_all_entities = []

    # Data Preparation
    if normalize:
        metric_data_for_scaling = {metric: [] for metric in metric_columns}
        for df_entity in data_frames:
            for metric in metric_columns:
                metric_data_for_scaling[metric].append(df_entity[metric].iloc[0])
        
        min_max_per_metric = {}
        for metric, values in metric_data_for_scaling.items():
            # Ensure values are numeric before min/max, handle potential non-numeric data if necessary
            numeric_values = pd.to_numeric(values, errors='coerce')
            numeric_values = numeric_values[~np.isnan(numeric_values)] # Remove NaNs that were non-numeric
            if len(numeric_values) == 0: # All values were non-numeric or NaN
                 min_val, max_val = 0, 1 # Default scaling if no valid numeric data
            else:
                min_val = np.min(numeric_values)
                max_val = np.max(numeric_values)
            min_max_per_metric[metric] = (min_val, max_val)

    for df_entity in data_frames:
        entity_values = []
        for metric in metric_columns:
            value = df_entity[metric].iloc[0]
            if pd.isna(value): # Handle NaN values before normalization
                entity_values.append(0.0) # Or some other placeholder like np.nan then handle in plotting
                continue

            if normalize:
                min_val, max_val = min_max_per_metric[metric]
                
                if max_val == min_val:
                    normalized_value = 0.5 
                else:
                    normalized_value = (value - min_val) / (max_val - min_val)

                if normalization_specs and metric in normalization_specs:
                    if normalization_specs[metric] == 'inverted':
                        normalized_value = 1 - normalized_value
                    # elif normalization_specs[metric] != 'standard':
                        # Can add warning or error for unknown spec, or just default to standard
                entity_values.append(normalized_value)
            else:
                entity_values.append(value)
        processed_values_all_entities.append(entity_values)

    # Plotting
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Default styling
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        colors = [default_colors[i % len(default_colors)] for i in range(len(data_frames))]
    
    if line_styles is None: line_styles = ['-'] * len(data_frames)
    if line_widths is None: line_widths = [1.5] * len(data_frames)
    if fill_alphas is None: fill_alphas = [0.1] * len(data_frames)

    for i, values in enumerate(processed_values_all_entities):
        plot_values = values + values[:1]
        # Ensure no NaNs are passed to plot, replace with a value (e.g., 0) or skip point
        plot_values_cleaned = [0 if pd.isna(v) else v for v in plot_values]

        ax.plot(angles, plot_values_cleaned, color=colors[i], linestyle=line_styles[i], linewidth=line_widths[i], label=entity_names[i])
        ax.fill(angles, plot_values_cleaned, color=colors[i], alpha=fill_alphas[i])

    # Labels and Legend
    if metric_labels_override:
        x_tick_labels = [metric_labels_override.get(m, m.replace('_', ' ').title()) for m in metric_columns]
    else:
        x_tick_labels = [m.replace('_', ' ').title() for m in metric_columns]
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(x_tick_labels)
    
    if normalize:
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_ylim(0, 1.05)
    # Else: let matplotlib auto-scale for non-normalized data or define custom logic

    ax.set_title(title, size=15, y=1.1)

    legend_params = legend_kwargs if legend_kwargs is not None else {'loc': 'upper right', 'bbox_to_anchor': (0.1, 0.1)}
    ax.legend(**legend_params)

    if output_file:
        try:
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving figure to {output_file}: {e}")

    return fig
