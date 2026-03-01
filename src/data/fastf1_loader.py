"""
FastF1 wrapper with caching for weather and timing (fallback source).

Uses FastF1 library to load session data; caches to data/cache (or config
data.fastf1_cache_dir). Use when OpenF1 does not have data for a session.

Usage:
    from src.data.fastf1_loader import get_session_weather, get_session_results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# Lazy import so FastF1 is only required when this loader is used
_fastf1 = None


def _get_fastf1():
    global _fastf1
    if _fastf1 is None:
        try:
            import fastf1
            _fastf1 = fastf1
        except ImportError:
            log.warning("FastF1 not installed; fastf1_loader will no-op.")
    return _fastf1


def get_cache_dir(config: dict[str, Any] | None = None) -> Path:
    if config:
        return Path(config.get("data", {}).get("fastf1_cache_dir", "data/cache"))
    return Path("data/cache")


def get_session_weather(
    year: int,
    round_number: int,
    session_type: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load session weather from FastF1 (cached). Returns one row with air_temp_c, track_temp_c, etc."""
    ff1 = _get_fastf1()
    if ff1 is None:
        return pd.DataFrame()
    cache_dir = get_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        ff1.Cache.enable_cache(str(cache_dir))
    except Exception:
        pass
    _st = {"qualifying": "Q", "race": "R", "sprint": "Sprint"}.get(session_type.lower(), session_type)
    try:
        session = ff1.get_session(year, round_number, _st)
        session.load()
    except Exception as e:
        log.debug("FastF1 session load failed %s %s %s: %s", year, round_number, session_type, e)
        return pd.DataFrame()
    weather = getattr(session, "weather_data", None)
    if weather is None or (isinstance(weather, pd.DataFrame) and weather.empty):
        return pd.DataFrame()
    if isinstance(weather, pd.DataFrame):
        # Take last row (end of session) or first
        row = weather.iloc[-1] if len(weather) > 0 else None
        if row is None:
            return pd.DataFrame()
        out = pd.DataFrame([{
            "year": year,
            "round": round_number,
            "session_type": session_type.lower(),
            "air_temp_c": row.get("AirTemp") if hasattr(row, "get") else getattr(row, "AirTemp", None),
            "track_temp_c": row.get("TrackTemp") if hasattr(row, "get") else getattr(row, "TrackTemp", None),
            "humidity_pct": None,
            "wind_speed_ms": None,
            "rainfall": None,
        }])
        return out
    return pd.DataFrame()


def get_session_results(
    year: int,
    round_number: int,
    session_type: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load session results (positions, drivers) from FastF1. Returns DataFrame with driver_code, finish_position, etc."""
    ff1 = _get_fastf1()
    if ff1 is None:
        return pd.DataFrame()
    cache_dir = get_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        ff1.Cache.enable_cache(str(cache_dir))
    except Exception:
        pass
    _st = {"qualifying": "Q", "race": "R", "sprint": "Sprint"}.get(session_type.lower(), session_type)
    try:
        session = ff1.get_session(year, round_number, _st)
        session.load()
    except Exception as e:
        log.debug("FastF1 session load failed: %s", e)
        return pd.DataFrame()
    results = getattr(session, "results", None)
    if results is None or (isinstance(results, pd.DataFrame) and results.empty):
        return pd.DataFrame()
    if isinstance(results, pd.DataFrame):
        out = results.copy()
        out["year"] = year
        out["round"] = round_number
        out["session_type"] = session_type.lower()
        if "DriverNumber" in out.columns and "Abbreviation" not in out.columns:
            try:
                from src.data.schema import normalise_driver_number
                out["driver_code"] = out["DriverNumber"].map(lambda x: normalise_driver_number(int(x)) if pd.notna(x) else None)
            except Exception:
                pass
        return out
    return pd.DataFrame()
