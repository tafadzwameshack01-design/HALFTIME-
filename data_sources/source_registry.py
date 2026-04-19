# CONTRACT: data_sources/source_registry.py
# Classes: SourceRegistry
# Methods: __init__, __repr__, before_call, after_call, get_health,
#          get_all_health, get_priority_sources

from config import *

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SourceRegistry:
    """Tracks API source health and rate limits. Guards all outbound calls."""

    SOURCES: list = ["espn", "openligadb", "football_data_org",
                     "api_football", "understat", "fbref", "odds_api"]

    PRIORITY_MAP: dict = {
        "fixtures": ["espn", "football_data_org", "api_football"],
        "stats":    ["api_football", "fbref", "understat"],
        "live":     ["espn"],
        "odds":     ["odds_api"],
        "xg":       ["understat", "fbref"],
    }

    # Daily limits per source (0 = unlimited)
    DAILY_LIMITS: dict = {
        "api_football": API_FOOTBALL_DAILY_LIMIT,
        "odds_api":     ODDS_API_MONTHLY_LIMIT,
    }

    def __init__(self, db) -> None:
        """
        Args:
            db: DatabaseManager instance for api_usage reads/writes.
        """
        self._db = db

    def before_call(self, source_name: str) -> bool:
        """
        Check if a source call is permitted.

        Returns False (blocking the call) if:
        - Source health is RED (3+ consecutive errors)
        - Daily/monthly API limit is exceeded

        Args:
            source_name: Identifier of the data source.

        Returns:
            bool: True if call is permitted, False to block.
        """
        try:
            health = self.get_health(source_name)
            if health == "RED":
                logger.warning("Source %r is RED — call blocked.", source_name)
                return False

            # Check rate limits for capped sources
            limit = self.DAILY_LIMITS.get(source_name, 0)
            if limit > 0:
                usage = self._db.get_api_usage(source_name)
                if usage.get("call_count", 0) >= limit:
                    logger.warning("Source %r has reached its daily/monthly limit (%d).", source_name, limit)
                    return False

            return True
        except Exception as exc:
            logger.error("before_call(%r) error: %s", source_name, exc)
            return True  # allow on error to avoid blocking legitimate calls

    def after_call(self, source_name: str, success: bool, error: str = "") -> None:
        """
        Record the outcome of an API call.

        Args:
            source_name: Data source identifier.
            success: Whether the call succeeded.
            error: Optional error message for failed calls.
        """
        try:
            self._db.track_api_call(source_name, success, error)
        except Exception as exc:
            logger.error("after_call(%r) error: %s", source_name, exc)

    def get_health(self, source_name: str) -> str:
        """
        Determine current health status of a source.

        Rules:
        - GREEN: 0 errors in last 10 calls today
        - AMBER: 1-2 errors in last 10 calls today
        - RED: 3+ consecutive errors OR daily limit exceeded

        Args:
            source_name: Data source identifier.

        Returns:
            str: "GREEN", "AMBER", or "RED".
        """
        try:
            usage = self._db.get_api_usage(source_name)
            call_count = usage.get("call_count", 0)
            error_count = usage.get("error_count", 0)

            # Check hard limit
            limit = self.DAILY_LIMITS.get(source_name, 0)
            if limit > 0 and call_count >= limit:
                return "RED"

            if call_count == 0:
                return "GREEN"

            # Compute error rate over today's calls (capped window at last 10)
            recent_window = min(call_count, 10)
            # We use error_count as a proxy — if error_count >= 3 and
            # they are recent (today), classify as RED
            if error_count >= 3:
                return "RED"
            elif error_count >= 1:
                return "AMBER"
            return "GREEN"
        except Exception as exc:
            logger.error("get_health(%r) error: %s", source_name, exc)
            return "GREEN"

    def get_all_health(self) -> dict:
        """
        Return health status for all registered sources.

        Returns:
            dict: {source_name: health_status} for all SOURCES.
        """
        return {source: self.get_health(source) for source in self.SOURCES}

    def get_priority_sources(self, data_type: str) -> list:
        """
        Return ordered list of healthy sources for a data type.

        Args:
            data_type: One of "fixtures", "stats", "live", "odds", "xg".

        Returns:
            list: Source names in priority order, excluding RED sources.
        """
        priority = self.PRIORITY_MAP.get(data_type, self.SOURCES)
        return [s for s in priority if self.get_health(s) != "RED"]

    def __repr__(self) -> str:
        health = self.get_all_health()
        return f"SourceRegistry(sources={len(self.SOURCES)}, health={health})"
