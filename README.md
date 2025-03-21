# Fast Soccer Analysis Scripts

A modular Python package for analyzing soccer player statistics and identifying talent.

## 📋 Overview

This project provides tools for:

- Loading and processing soccer statistics from different sources
- Analyzing player performance across various metrics
- Identifying promising players based on different attributes
- Creating comprehensive scouting reports
- Storing and tracking player data over time
- Advanced metrics analysis with visualizations
- **Enhanced shooting analysis capabilities**

## 🏗️ Project Structure

```
fast-soccer-analysis/
│
├── config/                 # Configuration settings
│   ├── settings.py         # General settings and parameters
│   └── urls.py             # Data source URLs
│
├── src/                    # Core functionality
│   ├── data/               # Data handling
│   │   ├── loaders.py      # Data loading functions
│   │   ├── processors.py   # Data processing functions
│   │   └── shooting_processors.py  # Enhanced shooting data processors
│   │
│   ├── db/                 # Database operations
│   │   └── operations.py   # Database functions
│   │
│   ├── analysis/           # Analysis algorithms
│   │   ├── basic/          # Basic analysis functions
│   │   │   ├── playmakers.py  # Playmaker identification
│   │   │   ├── forwards.py    # Forward analysis
│   │   │   └── midfielders.py # Midfielder analysis
│   │   │
│   │   ├── advanced/       # Advanced analysis modules
│   │   │   ├── versatility.py     # Versatility score calculations
│   │   │   ├── progression.py     # Progressive action analysis
│   │   │   ├── possession_impact.py  # xPI calculations
│   │   │   └── clustering.py      # Positional clustering analysis
│   │   │
│   │   ├── shooting_analyzer.py  # Advanced shooting analysis
│   │   └── metrics.py      # Metrics calculation utilities
│   │
│   └── utils/              # Utility functions
│       ├── normalization.py  # Metric normalization helpers
│       ├── logging_setup.py  # Logging configuration
│       ├── visualization.py  # General visualization utilities
│       └── shooting_visualizations.py  # Shooting-specific visualizations
│
├── pipelines/              # Analysis pipelines
│   ├── full_analysis.py    # Complete analysis pipeline
│   ├── advanced_analysis.py  # Advanced analysis pipeline
│   ├── shooting_pipeline.py  # Shooting analysis pipeline
│   └── daily_update.py     # Daily data update pipeline
│
├── visualizations/         # Generated visualization outputs
│   └── shooting/           # Shooting-specific visualizations
│
└── main.py                 # Main application entry point
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Dependencies listed in `pyproject.toml`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fast-soccer-analysis.git
   cd fast-soccer-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and dependencies:
   ```bash
   uv pip install -e .
   ```

## 📊 Usage

### Basic Usage

Run a complete analysis with default parameters:

```bash
python main.py
```

### Advanced Analysis

Run advanced analysis with player versatility scores, progression metrics, and clustering:

```bash
python main.py --analysis-type advanced
```

### Shooting Analysis

Run in-depth shooting analysis with shooting efficiency, profiles, and shot quality metrics:

```bash
python main.py --analysis-type shooting
```

### Run All Analyses

Run all available analysis types together:

```bash
python main.py --analysis-type all
```

### Command Line Options

The main script supports various options:

```bash
python main.py --analysis-type shooting --min-shots 15 --top-n 20 --positions FW --min-90s 10 --max-age 25 --report-file report.md
```

Options:
- `--analysis-type`: Type of analysis to run (basic, advanced, shooting, all)
- `--min-shots`: Minimum shots for forward analysis (default: 20)
- `--top-n`: Number of top players to return (default: 20)
- `--positions`: Positions to analyze (default: ["MF", "FW, MF", "MF,DF"])
- `--min-90s`: Minimum 90-minute periods played (default: 5)
- `--max-age`: Maximum player age to include (default: 30)
- `--force-reload`: Force data reload from source
- `--no-save`: Don't save to database
- `--no-visualizations`: Skip creating visualizations for advanced analysis
- `--report-file`: Path to save the report

### Using as a Module

```python
from pipelines.full_analysis import run_analysis_pipeline
from pipelines.advanced_analysis import run_advanced_analysis
from pipelines.shooting_pipeline import run_shooting_analysis

# Run basic analysis with custom parameters
basic_results = run_analysis_pipeline(
    min_shots=15,
    top_n=20,
    positions=["MF", "FW"],
    min_90s=10,
    max_age=23,
    save_to_db=True,
    report_file="reports/young_midfielders.md"
)

# Run advanced analysis
advanced_results = run_advanced_analysis(
    min_shots=15,
    top_n=20,
    positions=["MF", "FW"],
    min_90s=10,
    max_age=23,
    save_to_db=True,
    create_visualizations=True
)

# Run shooting analysis
shooting_results = run_shooting_analysis(
    min_shots=15,
    top_n=20,
    positions=["FW"],
    min_90s=8,
    max_age=25,
    create_visualizations=True,
    output_dir="visualizations/shooting"
)

# Access specific results
playmakers = basic_results["playmakers"]
print(f"Top playmaker: {playmakers.iloc[0]['Player']}")

versatile_players = advanced_results["versatile_players"]
print(f"Most versatile player: {versatile_players.iloc[0]['Player']}")

clinical_forwards = shooting_results["shooting_efficiency"]
print(f"Most efficient shooter: {clinical_forwards.iloc[0]['Player']}")
```

### Daily Updates

Run daily data updates to keep your database current:

```bash
python -m pipelines.daily_update --output-dir reports/updates
```

## 🔍 Analysis Types

### Basic Analysis

- **Playmakers**: Creative midfielders based on progressive passing and chance creation
- **Clinical Forwards**: Efficient forwards based on shooting and conversion metrics
- **Progressive Midfielders**: Players who excel at moving the ball forward
- **Pressing Midfielders**: Players who excel in defensive actions and pressing
- **Complete Midfielders**: Well-rounded midfielders who contribute in multiple areas
- **Passing Quality**: Players with exceptional passing metrics

### Advanced Analysis

- **Player Versatility**: Players who excel across multiple skill areas (passing, possession, defense, shooting)
- **Progressive Actions**: Breakdown of how players move the ball forward (carrying, passing, receiving)
- **Expected Possession Impact (xPI)**: Comprehensive metric quantifying a player's contribution to team possession
- **Positional Clustering**: Groups players by statistical profiles rather than listed positions

### Shooting Analysis

- **Shooting Efficiency**: Comprehensive analysis of shooting effectiveness and conversion quality
- **Shooting Profiles**: Classification of players into shooting style categories (Volume Shooter, Clinical Finisher, etc.)
- **Shot Creation Specialists**: Players who excel at both scoring and creating shots
- **Finishing Skill**: Analysis of players who outperform their expected goals metrics
- **Shot Quality**: Evaluation of shot selection based on location, accuracy, and expected value

## 📊 Visualizations

The analysis automatically generates visualizations including:

- Radar charts comparing player strengths
- Scatter plots showing relationships between metrics
- Bar charts for direct player comparisons
- Heatmaps for comprehensive metric evaluation
- Finishing skill plots comparing goals to expected goals
- Shot quality distributions
- Shot distance histograms

These visualizations are saved in the `visualizations/` directory and can be referenced in reports.

## 🗄️ Database

Player data and analysis results are stored in a DuckDB database (`scouting.db` by default). This provides:

- Efficient storage of player statistics
- Tracking player development over time
- Persistent storage of analysis results (basic, advanced, and shooting)
- Fast querying capabilities

Analysis results are stored in separate tables with appropriate prefixes.

## 📈 Extending the System

### Adding New Analysis Types

1. Create a new analysis function in the appropriate module:
   - Basic analysis in `src/analysis/basic/`
   - Advanced analysis in `src/analysis/advanced/`
   - Shooting analysis in `src/analysis/shooting_analyzer.py`
2. Update the relevant pipelines to include your new analysis
3. Add appropriate weights and parameters in `config/settings.py`

### Adding New Visualizations

1. Create a new visualization function in the appropriate module:
   - General visualizations in `src/utils/visualization.py`
   - Shooting visualizations in `src/utils/shooting_visualizations.py`
2. Update the visualization dashboard to include your new chart
3. Reference the visualization in your reports

### Adding New Data Sources

1. Add the URL to `config/urls.py`
2. Create a processor function in the appropriate module:
   - General processing in `src/data/processors.py`
   - Shooting-specific processing in `src/data/shooting_processors.py`
3. Update the data loader to handle the new source

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [FBref](https://fbref.com/) for providing soccer statistics
