# Fast Soccer Analysis Scripts

A modular Python package for analyzing soccer player statistics and identifying talent.

## 📋 Overview

This project provides tools for:

- Loading and processing soccer statistics from different sources
- Analyzing player performance across various metrics
- Identifying promising players based on different attributes
- Creating comprehensive scouting reports
- Storing and tracking player data over time

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
│   ├── db/                 # Database operations
│   ├── analysis/           # Analysis algorithms
│   └── utils/              # Utility functions
│
├── pipelines/              # Analysis pipelines
│   ├── full_analysis.py    # Complete analysis pipeline
│   └── daily_update.py     # Daily data update pipeline
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

### Command Line Options

The main script supports various options:

```bash
python main.py --min-shots 15 --top-n 20 --positions MF FW --min-90s 10 --max-age 25 --report-file report.md
```

Options:
- `--min-shots`: Minimum shots for forward analysis (default: 20)
- `--top-n`: Number of top players to return (default: 20)
- `--positions`: Positions to analyze (default: ["MF", "FW, MF", "MF,DF"])
- `--min-90s`: Minimum 90-minute periods played (default: 5)
- `--max-age`: Maximum player age (default: 30)
- `--force-reload`: Force data reload from source
- `--no-save`: Don't save to database
- `--report-file`: Path to save the report

### Using as a Module

```python
from pipelines.full_analysis import run_analysis_pipeline

# Run analysis with custom parameters
results = run_analysis_pipeline(
    min_shots=15,
    top_n=20,
    positions=["MF", "FW"],
    min_90s=10,
    max_age=23,
    save_to_db=True,
    report_file="reports/young_midfielders.md"
)

# Access specific results
playmakers = results["playmakers"]
print(f"Top playmaker: {playmakers.iloc[0]['Player']}")
```

### Daily Updates

Run daily data updates to keep your database current:

```bash
python -m pipelines.daily_update --output-dir reports/updates
```

## 🔍 Analysis Types

The system provides several specialized analyses:

- **Playmakers**: Creative midfielders based on progressive passing and chance creation
- **Clinical Forwards**: Efficient forwards based on shooting and conversion metrics
- **Progressive Midfielders**: Players who excel at moving the ball forward
- **Pressing Midfielders**: Players who excel in defensive actions and pressing
- **Complete Midfielders**: Well-rounded midfielders who contribute in multiple areas
- **Passing Quality**: Players with exceptional passing metrics

## 🗄️ Database

Player data and analysis results are stored in a DuckDB database (`scouting.db` by default). This provides:

- Efficient storage of player statistics
- Tracking player development over time
- Persistent storage of analysis results
- Fast querying capabilities

## 📈 Extending the System

### Adding New Analysis Types

1. Create a new analysis function in `src/analysis/player_scout.py` or a new module
2. Update the relevant pipelines to include your new analysis
3. Add appropriate weights and parameters in `config/settings.py`

### Adding New Data Sources

1. Add the URL to `config/urls.py`
2. Create a processor function in `src/data/processors.py` if needed
3. Update the data loader to handle the new source

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [FBref](https://fbref.com/) for providing soccer statistics
- [pandas](https://pandas.pydata.org/) for data manipulation
- [DuckDB](https://duckdb.org/) for database functionality
