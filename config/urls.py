"""
URLs for data sources used in the soccer analysis application.
"""

# Base URL for FBRef data
FBREF_BASE = "https://fbref.com/en/comps/Big5"

# Specific statistics URLs
URLS = {
    "defense": f"{FBREF_BASE}/defense/players/Big-5-European-Leagues-Stats",
    "possession": f"{FBREF_BASE}/possession/players/Big-5-European-Leagues-Stats",
    "passing": f"{FBREF_BASE}/passing/players/Big-5-European-Leagues-Stats",
    "shooting": f"{FBREF_BASE}/shooting/players/Big-5-European-Leagues-Stats",
    "shot_creation": f"{FBREF_BASE}/gca/players/Big-5-European-Leagues-Stats",
    "keepers": f"{FBREF_BASE}/keepers/players/Big-5-European-Leagues-Stats",
    "advanced_keepers": f"{FBREF_BASE}/keepersadv/players/Big-5-European-Leagues-Stats",
    "standard": f"{FBREF_BASE}/stats/players/Big-5-European-Leagues-Stats",
}

# API endpoints for other data sources could be added here
