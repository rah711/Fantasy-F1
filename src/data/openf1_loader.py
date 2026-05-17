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

import json
from src.data.schema import normalise_circuit, normalise_constructor, normalise_driver_number
from src.utils.logging import get_logger

log = get_logger(__name__)

# Default base URL (config can override)
OPENF1_BASE = "https://api.openf1.org/v1"
ONE_SECOND = 1
CALLS_PER_SECOND = 3


@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=ONE_SECOND)
def _rate_limited_get(session: requests.Session, url: str) -> requests.Response:
    # Retry-on-429 with exponential backoff. OpenF1 can throttle harder than
    # our configured rate limit, especially during bulk fetches.
    delay = 2.0
    for attempt in range(5):
        resp = session.get(url, timeout=30)
        if resp.status_code != 429:
            return resp
        retry_after = resp.headers.get("Retry-After")
        try:
            sleep_for = float(retry_after) if retry_after else delay
        except ValueError:
            sleep_for = delay
        log.info("OpenF1 429 — sleeping %.1fs before retry %d/5", sleep_for, attempt + 1)
        time.sleep(min(sleep_for, 30.0))
        delay = min(delay * 2, 30.0)
    return resp


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


# ============================================================
# 2026+ base sessions: pull race/quali/sprint results from OpenF1
# (since Kaggle is static and TracingInsights ends at 2025)
# ============================================================

_BASE_SESSIONS_CACHE = Path("data/cache/openf1_sessions")


def _cache_read(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _cache_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fetch_meetings(
    year: int,
    config: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Race meetings for the year, sorted by date with `round` numbers assigned.

    Skips pre-season testing meetings. Returns DataFrame with meeting_key,
    round, location, circuit_short_name, country_code, date_start.
    """
    cdir = Path(cache_dir) if cache_dir else _BASE_SESSIONS_CACHE / str(year)
    cache_path = cdir / "meetings.json"
    data = _cache_read(cache_path)
    if data is None:
        base, _ = _get_session_params(config)
        sess = requests.Session()
        url = f"{base}/meetings?year={year}"
        try:
            r = _rate_limited_get(sess, url)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("OpenF1 meetings fetch failed for %s: %s", year, e)
            return pd.DataFrame()
        if data:
            _cache_write(cache_path, data)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "meeting_name" in df.columns:
        mask = ~df["meeting_name"].astype(str).str.contains("Testing|Pre-Season", case=False, na=False)
        df = df[mask].copy()
    if df.empty:
        return df
    df = df.sort_values("date_start").reset_index(drop=True)
    df["round"] = df.index + 1
    return df


def _fetch_session_result(session_key: int, cache_dir: Path, config: dict[str, Any] | None = None) -> list:
    cache_path = cache_dir / f"session_result_{session_key}.json"
    cached = _cache_read(cache_path)
    if cached is not None:
        return cached
    base, _ = _get_session_params(config)
    sess = requests.Session()
    try:
        r = _rate_limited_get(sess, f"{base}/session_result?session_key={session_key}")
        r.raise_for_status()
        data = r.json() or []
    except Exception as e:
        log.warning("OpenF1 session_result fetch failed for %s: %s", session_key, e)
        return []
    if data:
        _cache_write(cache_path, data)
    return data


def _fetch_drivers(session_key: int, cache_dir: Path, config: dict[str, Any] | None = None) -> list:
    cache_path = cache_dir / f"drivers_{session_key}.json"
    cached = _cache_read(cache_path)
    if cached is not None:
        return cached
    base, _ = _get_session_params(config)
    sess = requests.Session()
    try:
        r = _rate_limited_get(sess, f"{base}/drivers?session_key={session_key}")
        r.raise_for_status()
        data = r.json() or []
    except Exception as e:
        log.warning("OpenF1 drivers fetch failed for %s: %s", session_key, e)
        return []
    if data:
        _cache_write(cache_path, data)
    return data


def load_openf1_race_sessions(
    year: int,
    config: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Pull race + qualifying + sprint session results from OpenF1 for `year`.

    Returns base sessions DataFrame compatible with the pipeline's expected
    schema: year, round, circuit_id, session_type, driver_code, constructor_id,
    finish_position, status, points_official, finish_time_ms.

    Disk-cached so subsequent runs are nearly instant (race results are
    immutable once a session is over).
    """
    cdir = Path(cache_dir) if cache_dir else _BASE_SESSIONS_CACHE / str(year)
    cdir.mkdir(parents=True, exist_ok=True)

    meetings = fetch_meetings(year, config=config, cache_dir=cdir)
    if meetings.empty:
        log.info("No OpenF1 meetings for year %d", year)
        return pd.DataFrame()

    sessions_df = fetch_sessions(year, config=config)
    if sessions_df.empty:
        log.info("No OpenF1 sessions for year %d", year)
        return pd.DataFrame()

    name_map = {"race": "Race", "qualifying": "Qualifying", "sprint": "Sprint"}
    rows: list[dict[str, Any]] = []

    for _, meeting in meetings.iterrows():
        mk = int(meeting["meeting_key"])
        rnd = int(meeting["round"])
        location = str(meeting.get("location") or "")
        circuit_short = str(meeting.get("circuit_short_name") or "")
        try:
            circuit_id = normalise_circuit(location)
        except KeyError:
            try:
                circuit_id = normalise_circuit(circuit_short)
            except KeyError:
                log.warning("Couldn't normalize circuit for meeting %s (%s / %s)", mk, location, circuit_short)
                circuit_id = location.lower().replace(" ", "_") or f"unknown_{mk}"

        meeting_sessions = sessions_df[sessions_df["meeting_key"] == mk]
        if meeting_sessions.empty:
            continue

        for stype_canon, stype_name in name_map.items():
            sub = meeting_sessions[meeting_sessions["session_name"] == stype_name]
            if sub.empty:
                continue
            session_key = int(sub.iloc[0]["session_key"])

            results = _fetch_session_result(session_key, cdir, config=config)
            drivers = _fetch_drivers(session_key, cdir, config=config)
            if not results or not drivers:
                continue

            d_map = {int(d["driver_number"]): d for d in drivers if d.get("driver_number") is not None}

            for r in results:
                dnum = r.get("driver_number")
                if dnum is None:
                    continue
                drv = d_map.get(int(dnum))
                if not drv:
                    continue

                team_raw = drv.get("team_name") or ""
                try:
                    constructor_id = normalise_constructor(team_raw)
                except KeyError:
                    log.warning("Unknown team for meeting %s: %s — using raw fallback", mk, team_raw)
                    constructor_id = team_raw.lower().replace(" ", "_") if team_raw else None

                if r.get("dsq"):
                    status = "DSQ"
                elif r.get("dnf"):
                    status = "DNF"
                elif r.get("dns"):
                    status = "DNS"
                else:
                    status = "Finished"

                duration = r.get("duration")
                try:
                    finish_time_ms = int(float(duration) * 1000) if duration is not None else None
                except (TypeError, ValueError):
                    finish_time_ms = None

                pos = r.get("position")
                try:
                    finish_position = int(pos) if pos is not None else None
                except (TypeError, ValueError):
                    finish_position = None

                rows.append({
                    "year": int(year),
                    "round": rnd,
                    "circuit_id": circuit_id,
                    "session_type": stype_canon,
                    "driver_code": drv.get("name_acronym"),
                    "constructor_id": constructor_id,
                    "driver_number": int(dnum),
                    "finish_position": finish_position,
                    "status": status,
                    "points_official": r.get("points"),
                    "finish_time_ms": finish_time_ms,
                    "number_of_laps": r.get("number_of_laps"),
                })

    log.info("OpenF1 race sessions for %d: %d rows across %d meetings", year, len(rows), len(meetings))
    return pd.DataFrame(rows)
