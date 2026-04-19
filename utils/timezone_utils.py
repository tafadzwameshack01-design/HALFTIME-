# CONTRACT: utils/timezone_utils.py
# Functions: to_utc, from_utc, now_utc, get_todays_fixture_window_utc,
#            match_is_today, minutes_until_kickoff, match_is_live,
#            is_first_half, estimated_current_minute

from config import *

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _get_tz(league_key: str):
    """Return a timezone object for the given league key using zoneinfo with pytz fallback."""
    if league_key not in LEAGUES:
        raise ValueError(f"Unknown league_key {league_key!r}.")
    tz_str = LEAGUES[league_key]["timezone"]
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            return ZoneInfo(tz_str)
        except ZoneInfoNotFoundError:
            logger.warning("zoneinfo could not find %r — falling back to pytz.", tz_str)
    except ImportError:
        logger.warning("zoneinfo not available — falling back to pytz.")
    import pytz
    return pytz.timezone(tz_str)


def _make_aware(dt: datetime, tz) -> datetime:
    """Attach timezone info to a naive datetime."""
    try:
        from zoneinfo import ZoneInfo
        if isinstance(tz, ZoneInfo):
            return dt.replace(tzinfo=tz)
    except ImportError:
        pass
    import pytz
    if hasattr(tz, "localize"):
        return tz.localize(dt)
    return dt.replace(tzinfo=tz)


def to_utc(dt: datetime, league_key: str) -> datetime:
    """Convert a league-local datetime to UTC."""
    try:
        if dt.tzinfo is None:
            tz = _get_tz(league_key)
            dt = _make_aware(dt, tz)
        return dt.astimezone(timezone.utc)
    except Exception as exc:
        logger.error("to_utc failed for league=%r: %s", league_key, exc)
        return dt.replace(tzinfo=timezone.utc)


def from_utc(dt: datetime, league_key: str) -> datetime:
    """Convert a UTC datetime to the league's local timezone."""
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        tz = _get_tz(league_key)
        return dt.astimezone(tz)
    except Exception as exc:
        logger.error("from_utc failed for league=%r: %s", league_key, exc)
        return dt


def now_utc() -> datetime:
    """Return the current UTC-aware datetime."""
    return datetime.now(timezone.utc)


def get_todays_fixture_window_utc() -> tuple:
    """Return (window_start, window_end) UTC: 05:00 today to 03:00 tomorrow."""
    today = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)
    return today.replace(hour=5), today + timedelta(days=1, hours=3)


def match_is_today(kickoff_utc: datetime) -> bool:
    """Return True if kickoff falls within today's fixture window."""
    try:
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
        window_start, window_end = get_todays_fixture_window_utc()
        return window_start <= kickoff_utc <= window_end
    except Exception as exc:
        logger.error("match_is_today error: %s", exc)
        return False


def minutes_until_kickoff(kickoff_utc: datetime) -> float:
    """Return minutes from now until kickoff (negative = past)."""
    try:
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
        return (kickoff_utc - now_utc()).total_seconds() / 60.0
    except Exception as exc:
        logger.error("minutes_until_kickoff error: %s", exc)
        return 0.0


def match_is_live(kickoff_utc: datetime, current_minute: int = 0) -> bool:
    """Return True if match is currently in progress (0-95 minutes elapsed)."""
    try:
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
        elapsed = (now_utc() - kickoff_utc).total_seconds() / 60.0
        if elapsed < 0:
            return False
        minute = current_minute if current_minute > 0 else elapsed
        return 0 <= minute <= 95
    except Exception as exc:
        logger.error("match_is_live error: %s", exc)
        return False


def is_first_half(kickoff_utc: datetime) -> bool:
    """Return True if estimated elapsed time is between 0 and 47 minutes."""
    try:
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
        elapsed = (now_utc() - kickoff_utc).total_seconds() / 60.0
        return 0.0 <= elapsed <= 47.0
    except Exception as exc:
        logger.error("is_first_half error: %s", exc)
        return False


def estimated_current_minute(kickoff_utc: datetime) -> int:
    """Return estimated match minute capped at 45."""
    try:
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
        elapsed = (now_utc() - kickoff_utc).total_seconds() / 60.0
        if elapsed < 0:
            return 0
        return min(45, int(elapsed))
    except Exception as exc:
        logger.error("estimated_current_minute error: %s", exc)
        return 0
