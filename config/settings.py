import os

# Database settings
DATABASE = {
    "path": os.getenv("DB_PATH", "scouting.db"),
    "backup_dir": os.getenv("DB_BACKUP_DIR", "./backups"),
}

# Default analysis parameters
DEFAULT_ANALYSIS_PARAMS = {
    "min_90s": 5,
    "max_age": 23,
    "min_shots": 20,
    "top_n": 20,
    "positions": ["MF", "MF,DF"],
}

# Visualization settings
VISUALIZATION_DIR = "visualizations"
REPORTS_DIR = "reports"

# Logging configuration
LOGGING = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.getenv("LOG_FILE", "soccer_analysis.log"),
}

# Column mappings and transformations
COLUMN_MAPPINGS = {
    "passing": {
        "rename": {
            8: "total_cmp",
            10: "total_Cmp%",
        }
    },
    "defense": {
        "rename": {
            # Map duplicate column names
            "Tkl_duplicate": "Tkl_challenge",
        }
    }
}

# Performance thresholds for player filtering
PLAYER_THRESHOLDS = {
    "defense": {"Tkl%": 50.0},
    "shooting": {"Gls": 5},
}

# Analysis weighting factors
ANALYSIS_WEIGHTS = {
    "playmaker": {
        "PrgP_90_norm": 0.35,
        "KP_90_norm": 0.30,
        "total_Cmp%_norm": 0.20,
        "Ast_90_norm": 0.15,
    },
    "forward": {
        "conversion_rate_norm": 0.30,
        "SoT%_norm": 0.25,
        "xG_difference_norm": 0.25,
        "Gls_90_norm": 0.20,
    },
    "progressive": {
        "PrgDist_norm": 0.35,
        "PrgC_norm": 0.30,
        "1/3_norm": 0.20,
        "PrgR_norm": 0.15,
    },
    "pressing": {
        "Tkl_90_norm": 0.35,
        "Int_90_norm": 0.30,
        "Tkl%_norm": 0.20,
        "Att3rd_90_norm": 0.15,
    },
    "complete_midfielder": {
        "progression_score_norm": 0.40,
        "pressing_score_norm": 0.30,
        "playmaker_score_norm": 0.30,
    }
}

ANALYSIS_WEIGHTS.update({
    # Add weights for versatility analysis
    "versatility": {
        "passing_component": 0.30,
        "possession_component": 0.30,
        "defensive_component": 0.30,
        "shooting_component": 0.10
    },

    # Add weights for possession impact analysis
    "possession_impact": {
        "touches_weight": 0.05,
        "carries_weight": 0.10,
        "dribbles_weight": 0.15,
        "progression_weight": 0.40,
        "retention_weight": 0.30
    },

    # Add weights for progression analysis components
    "progression_components": {
        "carrying_weight": 0.40,
        "passing_weight": 0.40,
        "receiving_weight": 0.20
    }
})

# Add specific parameters for advanced analysis
ADVANCED_ANALYSIS_PARAMS = {
    "visualization_dir": VISUALIZATION_DIR,
    "cluster_count": 5,  # Number of clusters for player clustering
    "value_analysis": {
        "age_penalty_factor": 0.3,  # Penalty factor for older players in value analysis
        "bargain_threshold": 0.90  # Percentile threshold to identify bargain players
    }
}
