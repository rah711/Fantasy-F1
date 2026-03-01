"""
OpenF1 API client for weather, overtakes, and pitstops (2023+).

Rate-limited to 3 requests per second (configurable). Resolves session_key
from year/round/session_type and fetches session-level and driver-level data.

Usage:
    from src.data.openf1_loader import load_openf1_weather, fetch_sessions
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from ratelimit import limits, sleep_and_retry

from src.data.schema import normalise_circuit, normalise_driver_number
from src.utils.logging import get_logger

log = get_logger(__name__)

# Default base URL (config can override)
OPENF1_BASE = "https://api.openf1.org/v1"
ONE_SECOND = 1
CALLS_PER_SECOND = 3


@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=ONE_SECOND)
def _rate_limited_get(session: requests.Session, url: str) -> requests.Response:
    return session.get(url, timeout=30)


def _get_session_params(config: dict[str, Any] | None) -> tuple[str, int]:
    base = OPENF1_BASE
    rate = CALLS_PER_SECOND
    if config:
        base = config.get("data", {}).get("openf1_base_url", base)
        rate = config.get("data", {}).get("openf1_rate_limit", rate)
    return base, rate


def fetch_sessions(
    year: int,
    config: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch all sessions for a year from OpenF1. Returns DataFrame with session_key, session_name, circuit_short_name, year, meeting_key."""
    base, _ = _get_session_params(config)
    sess = session or requests.Session()
    url = f"{base}/sessions?year={year}"
    try:
        r = _rate_limited_get(sess, url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("OpenF1 sessions fetch failed: %s", e)
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Infer round from first date_start per meeting (calendar order)
    if "meeting_key" in df.columns and "date_start" in df.columns:
        first_date = df.groupby("meeting_key")["date_start"].min().reset_index()
        first_date = first_date.sort_values("date_start").reset_index(drop=True)
        first_date["round"] = first_date.index + 1
        df = df.merge(first_date[["meeting_key", "round"]], on="meeting_key", how="left")
    return df


def session_key_for(
    year: int,
    round_number: int,
    session_type: str,
    config: dict[str, Any] | None = None,
    sessions_df: pd.DataFrame | None = None,
) -> int | None:
    """Resolve OpenF1 session_key for (year, round, session_type). session_type: 'qualifying'|'sprint'|'race'."""
    if sessions_df is not None and not sessions_df.empty:
        df = sessions_df
    else:
        df = fetch_sessions(year, config=config)
    if df.empty:
        return None
    # OpenF1 session_name: "Race", "Sprint", "Qualifying"
    name_map = {"race": "Race", "sprint": "Sprint", "qualifying": "Qualifying"}
    want = name_map.get(session_type.lower(), session_type)
    sub = df[(df["round"] == round_number) & (df["session_name"] == want)]
    if sub.empty:
        return None
    return int(sub.iloc[0]["session_key"])


def load_openf1_weather(
    year: int,
    round_number: int,
    session_type: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch weather for one session. Returns DataFrame with session_key, air_temp, track_temp, etc."""
    base, _ = _get_session_params(config)
    sess = requests.Session()
    sk = session_key_for(year, round_number, session_type, config=config)
    if sk is None:
        return pd.DataFrame()
    url = f"{base}/weather?session_key={sk}"
    try:
        r = _rate_limited_get(sess, url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("OpenF1 weather fetch failed session_key=%s: %s", sk, e)
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    df["year"] = year
    df["round"] = round_number
    df["session_type"] = session_type.lower()
    return df


def load_openf1_overtakes(
    year: int,
    round_number: int,
    session_type: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch overtakes for one session. Returns DataFrame with driver_number (or driver_code), overtakes count."""
    base, _ = _get_session_params(config)
    sess = requests.Session()
    sk = session_key_for(year, round_number, session_type, config=config)
    if sk is None:
        return pd.DataFrame()
    url = f"{base}/overtakes?session_key={sk}"
    try:
        r = _rate_limited_get(sess, url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("OpenF1 overtakes fetch failed: %s", e)
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    df["year"] = year
    df["round"] = round_number
    df["session_type"] = session_type.lower()
    return df


def load_openf1_pitstops(
    year: int,
    round_number: int,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch pitstops for the race. Returns DataFrame with driver_number, duration_ms, etc. aggregated per constructor."""
    base, _ = _get_session_params(config)
    sess = requests.Session()
    sk = session_key_for(year, round_number, "race", config=config)
    if sk is None:
        return pd.DataFrame()
    url = f"{base}/pitstops?session_key={sk}"
    try:
        r = _rate_limited_get(sess, url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("OpenF1 pitstops fetch failed: %s", e)
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    df["year"] = year
    df["round"] = round_number
    return df


def enrich_sessions_with_openf1(
    base_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Left-join OpenF1 weather and overtakes onto base_df (with year, round, session_type, driver_code).

    Adds columns: air_temp_c, track_temp_c, humidity_pct, wind_speed_ms, rainfall, overtakes (if missing).
    Rate-limited; use for moderate-sized base_df.
    """
    if base_df.empty or "year" not in base_df.columns or "round" not in base_df.columns or "session_type" not in base_df.columns:
        return base_df
    out = base_df.copy()
    keys = ["year", "round", "session_type"]
    for grp_id, grp in base_df.groupby(keys, dropna=False):
        year, round_num, st = grp_id
        weather = load_openf1_weather(year, round_num, st, config=config)
        if not weather.empty and "air_temperature" in weather.columns:
            idx = grp.index
            out.loc[idx, "air_temp_c"] = weather["air_temperature"].iloc[0] if "air_temperature" in weather.columns else None
            out.loc[idx, "track_temp_c"] = weather["track_temperature"].iloc[0] if "track_temperature" in weather.columns else None
            out.loc[idx, "humidity_pct"] = weather["humidity"].iloc[0] if "humidity" in weather.columns else None
            out.loc[idx, "wind_speed_ms"] = weather["wind_speed"].iloc[0] if "wind_speed" in weather.columns else None
            out.loc[idx, "rainfall"] = weather["rainfall"].iloc[0] if "rainfall" in weather.columns else None
        overtakes = load_openf1_overtakes(year, round_num, st, config=config)
        if not overtakes.empty and "driver_number" in overtakes.columns:
            for _, row in overtakes.iterrows():
                try:
                    code = normalise_driver_number(int(row["driver_number"]))
                except (KeyError, TypeError, ValueError):
                    continue
                mask = (out["year"] == year) & (out["round"] == round_num) & (out["session_type"] == st) & (out["driver_code"] == code)
                count = row.get("count") or row.get("overtakes") or len(overtakes)
                out.loc[mask, "overtakes"] = count
        time.sleep(1.0 / CALLS_PER_SECOND)
    return out
