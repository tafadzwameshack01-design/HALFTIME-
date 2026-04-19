# CONTRACT: utils/helpers.py
# Classes: InsufficientDataError
# Functions: safe_divide, flatten_dict, chunks, retry_with_backoff,
#            build_match_id, parse_espn_datetime, normalize_team_name,
#            get_season_stage, is_rivalry

from config import *

import hashlib
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """Raised when dataset is too small for reliable model fitting."""
    pass


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers safely, returning default on zero-division."""
    try:
        if denominator == 0 or denominator is None:
            return default
        result = numerator / denominator
        if result != result or abs(result) == float("inf"):
            return default
        return float(result)
    except (TypeError, ZeroDivisionError):
        return default


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Recursively flatten a nested dictionary."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunks(lst: list, n: int) -> list:
    """Split list into chunks of at most n items."""
    if n < 1:
        raise ValueError(f"Chunk size must be >= 1, got {n!r}.")
    return [lst[i: i + n] for i in range(0, len(lst), n)]


def retry_with_backoff(func: Callable, max_retries: int = 3, backoff_base: float = 2.0) -> Any:
    """Call func with exponential backoff on failure. Returns None after all retries."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning("retry_with_backoff: attempt %d/%d failed — %s. Waiting %.1fs.",
                           attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    logger.warning("retry_with_backoff: all %d attempts exhausted. Last error: %s", max_retries, last_exc)
    return None


def build_match_id(home_team_id: str, away_team_id: str, kickoff_utc: str) -> str:
    """Build a deterministic 16-char match ID from SHA-256 hash."""
    date_part = kickoff_utc[:10] if kickoff_utc else "0000-00-00"
    raw = f"{home_team_id}|{away_team_id}|{date_part}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_espn_datetime(espn_date_str: str) -> datetime:
    """Parse ESPN ISO-8601 datetime string to UTC-aware datetime."""
    if not espn_date_str:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        normalised = espn_date_str.replace("Z", "+00:00")
        if "." in normalised:
            dot_idx = normalised.index(".")
            plus_idx = normalised.index("+", dot_idx)
            normalised = normalised[:dot_idx] + normalised[plus_idx:]
        return datetime.fromisoformat(normalised)
    except Exception as exc:
        logger.warning("parse_espn_datetime could not parse %r: %s", espn_date_str, exc)
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def normalize_team_name(name: str) -> str:
    """Normalise team name: strip accents, lowercase, underscores."""
    if not name:
        return ""
    try:
        s = name.strip()
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = s.lower()
        s = re.sub(r"[\s\-]+", "_", s)
        s = re.sub(r"[^\w]", "", s)
        return s
    except Exception as exc:
        logger.warning("normalize_team_name(%r) failed: %s", name, exc)
        return name.lower().replace(" ", "_")


def get_season_stage(match_date: datetime, league_key: str) -> int:
    """Return season stage: 0=early (Aug-Oct), 1=mid (Nov-Feb), 2=late (Mar-May)."""
    try:
        month = match_date.month
        if month in (8, 9, 10):
            return 0
        elif month in (11, 12, 1, 2):
            return 1
        else:
            return 2
    except Exception as exc:
        logger.warning("get_season_stage failed: %s", exc)
        return 1


def is_rivalry(h2h_records: list) -> bool:
    """Return True if mean red cards across H2H records exceeds 2."""
    if not h2h_records:
        return False
    try:
        total_reds = 0.0
        for record in h2h_records:
            if "red_cards" in record:
                total_reds += float(record["red_cards"] or 0)
            else:
                total_reds += float(record.get("home_red_cards") or 0) + float(record.get("away_red_cards") or 0)
        return (total_reds / len(h2h_records)) > 2.0
    except Exception as exc:
        logger.warning("is_rivalry error: %s", exc)
        return False
