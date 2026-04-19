# CONTRACT: data_sources/fbref_scraper.py
# Classes: FBrefScraper
# Methods: __init__, __repr__, get_league_stats, get_team_shooting_stats

from config import *

import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class FBrefScraper:
    """Scrapes advanced statistics from fbref.com. Fully defensive — never raises."""

    def __init__(self, registry) -> None:
        """Args: registry — SourceRegistry instance."""
        self._registry = registry

    def get_league_stats(self, league_url_segment: str, season_year: int) -> pd.DataFrame:
        """
        Scrape match schedule/results with xG from FBref.

        Tries pd.read_html() first (faster), falls back to BeautifulSoup
        table parsing if that fails.

        Args:
            league_url_segment: FBref URL segment (e.g. '20-Bundesliga').
            season_year: Season start year (e.g. 2024 for 2024-25).

        Returns:
            pd.DataFrame: Columns — home_team, away_team, home_xg, away_xg,
                match_date, score. Empty DataFrame on any failure.
        """
        if not self._registry.before_call("fbref"):
            return pd.DataFrame()

        url = f"https://fbref.com/en/comps/{league_url_segment}/schedule/"
        try:
            time.sleep(SCRAPER_DELAY_SECONDS)
            resp = requests.get(url, headers=SCRAPER_HEADERS, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            self._registry.after_call("fbref", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("FBrefScraper.get_league_stats(%r) failed: %s", league_url_segment, exc)
            self._registry.after_call("fbref", False, str(exc))
            return pd.DataFrame()

        # Try pd.read_html first
        try:
            tables = pd.read_html(resp.text)
            for table in tables:
                cols = [str(c).lower() for c in table.columns]
                # Look for a table that has home/away team columns
                if any("home" in c for c in cols) and any("away" in c for c in cols):
                    # Normalise column names
                    table.columns = [str(c).lower().replace(" ", "_") for c in table.columns]
                    # Try to extract relevant columns
                    col_map = {}
                    for col in table.columns:
                        if "home" in col and "xg" not in col:
                            col_map["home_team"] = col
                        elif "away" in col and "xg" not in col:
                            col_map["away_team"] = col
                        elif "home" in col and "xg" in col:
                            col_map["home_xg"] = col
                        elif "away" in col and "xg" in col:
                            col_map["away_xg"] = col
                        elif "date" in col:
                            col_map["match_date"] = col
                        elif "score" in col:
                            col_map["score"] = col

                    if "home_team" in col_map and "away_team" in col_map:
                        result = pd.DataFrame()
                        result["home_team"] = table[col_map["home_team"]].astype(str)
                        result["away_team"] = table[col_map["away_team"]].astype(str)
                        result["home_xg"] = pd.to_numeric(table.get(col_map.get("home_xg", ""), pd.Series()), errors="coerce").fillna(0.0)
                        result["away_xg"] = pd.to_numeric(table.get(col_map.get("away_xg", ""), pd.Series()), errors="coerce").fillna(0.0)
                        result["match_date"] = table.get(col_map.get("match_date", ""), pd.Series(dtype=str)).astype(str)
                        result["score"] = table.get(col_map.get("score", ""), pd.Series(dtype=str)).astype(str)
                        # Drop rows with NaN team names
                        result = result[result["home_team"].notna() & (result["home_team"] != "nan")]
                        if not result.empty:
                            return result
        except Exception as exc:
            logger.debug("FBrefScraper pd.read_html failed, trying BeautifulSoup: %s", exc)

        # BeautifulSoup fallback
        try:
            soup = BeautifulSoup(resp.text, "lxml")
            table_tag = soup.find("table", id=lambda x: x and "sched" in x.lower())
            if not table_tag:
                table_tag = soup.find("table")
            if not table_tag:
                return pd.DataFrame()

            headers = [th.get_text(strip=True).lower().replace(" ", "_") for th in table_tag.find_all("th")]
            rows = []
            for tr in table_tag.find("tbody").find_all("tr"):
                cells = tr.find_all(["td", "th"])
                if len(cells) < 3:
                    continue
                row = [c.get_text(strip=True) for c in cells]
                rows.append(row)

            if not rows:
                return pd.DataFrame()

            max_cols = max(len(r) for r in rows)
            while len(headers) < max_cols:
                headers.append(f"col_{len(headers)}")

            df = pd.DataFrame(rows, columns=headers[:max_cols])
            return df
        except Exception as exc:
            logger.warning("FBrefScraper BeautifulSoup fallback failed: %s", exc)
            return pd.DataFrame()

    def get_team_shooting_stats(self, league_url_segment: str) -> pd.DataFrame:
        """
        Scrape per-team shooting statistics from FBref.

        Args:
            league_url_segment: FBref URL segment (e.g. '20-Bundesliga').

        Returns:
            pd.DataFrame: Columns — team, shots_per_90, shots_on_target_per_90,
                npxg_per_90. Empty DataFrame on any failure.
        """
        if not self._registry.before_call("fbref"):
            return pd.DataFrame()

        url = f"https://fbref.com/en/comps/{league_url_segment}/shooting/"
        try:
            time.sleep(SCRAPER_DELAY_SECONDS)
            resp = requests.get(url, headers=SCRAPER_HEADERS, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            self._registry.after_call("fbref", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("FBrefScraper.get_team_shooting_stats(%r) failed: %s", league_url_segment, exc)
            self._registry.after_call("fbref", False, str(exc))
            return pd.DataFrame()

        try:
            tables = pd.read_html(resp.text)
            for table in tables:
                table.columns = [str(c).lower().replace(" ", "_") for c in table.columns]
                if "squad" in table.columns or "team" in table.columns:
                    team_col = "squad" if "squad" in table.columns else "team"
                    result = pd.DataFrame()
                    result["team"] = table[team_col].astype(str)
                    result["shots_per_90"] = pd.to_numeric(table.get("sh/90", pd.Series()), errors="coerce").fillna(0.0)
                    result["shots_on_target_per_90"] = pd.to_numeric(table.get("sot/90", pd.Series()), errors="coerce").fillna(0.0)
                    result["npxg_per_90"] = pd.to_numeric(table.get("npxg/90", pd.Series()), errors="coerce").fillna(0.0)
                    result = result[result["team"].notna() & (result["team"] != "nan")]
                    if not result.empty:
                        return result
            return pd.DataFrame()
        except Exception as exc:
            logger.warning("FBrefScraper.get_team_shooting_stats parse error: %s", exc)
            return pd.DataFrame()

    def __repr__(self) -> str:
        return "FBrefScraper(source=fbref.com)"
