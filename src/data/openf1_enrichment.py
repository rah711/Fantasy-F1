"""
Batch enrichment of sessions data using OpenF1 API (2023+).

Fills in grid_position, status (DNF/DSQ), positions_gained, overtakes,
weather, and pit stop data that TracingInsights archive doesn't provide.
Results are cached locally so the API is only hit once per session.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.data.openf1_loader import (
    CALLS_PER_SECOND,
    _rate_limited_get,
    fetch_sessions,
)
from src.data.schema import DRIVER_NUMBER_TO_CODE
from src.utils.logging import get_logger

log = get_logger(__name__)

CACHE_DIR_DEFAULT = Path("data/cache/openf1_enrichment")

# Extended driver number mapping for historical drivers not in the 2026 schema
_EXTRA_DRIVER_NUMBERS: dict[int, str] = {
    3: "RIC",      # Ricciardo (2023-2024)
    22: "TSU",     # Tsunoda (2023-2025)
    20: "MAG",     # Magnussen (2023-2024)
    24: "ZHO",     # Zhou (2023-2024)
    2: "SAR",      # Sargeant (2023-2024)
    21: "DEV",     # De Vries (2023)
    6: "LAT",      # Latifi (historical)
    5: "VET",      # Vettel (2022)
    47: "MSC",     # Mick Schumacher (2022)
    7: "DOO",      # Doohan (2024-2025)
    38: "DOO",     # Doohan alternate number
    34: "BEA",     # Bearman alternate number
    50: "BOR",     # Bortoleto alternate number
    61: "ANT",     # Antonelli alternate number
    33: "VER",     # Verstappen used 33 in older seasons
    77: "BOT",     # Bottas
    99: "GIO",     # Giovinazzi (historical)
    9: "GAS",      # Gasly used 10 in AlphaTauri; 10 is already mapped
}


def _number_to_code(num: int) -> str | None:
    """Resolve driver number to 3-letter code, trying both maps."""
    if num in DRIVER_NUMBER_TO_CODE:
        return DRIVER_NUMBER_TO_CODE[num]
    if num in _EXTRA_DRIVER_NUMBERS:
        return _EXTRA_DRIVER_NUMBERS[num]
    return None


def _session_cache_path(cache_dir: Path, session_key: int) -> Path:
    return cache_dir / f"session_{session_key}.json"


def _fetch_and_cache(sess: requests.Session, url: str, cache_path: Path) -> list | None:
    """Fetch JSON from OpenF1 API with local file caching."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    try:
        r = _rate_limited_get(sess, url)
        if r.status_code == 404:
            cache_path.write_text("[]")
            return []
        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        log.debug("OpenF1 fetch failed %s: %s", url, e)
        return None


def _get_grid_positions(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> dict[int, int]:
    """Get starting grid positions from the qualifying/race starting position endpoint.

    Uses the /position endpoint at the start of the session — the first recorded
    position for each driver is their grid position.
    """
    cache_path = cache_dir / f"position_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/position?session_key={session_key}", cache_path)
    if not data:
        return {}
    df = pd.DataFrame(data)
    if df.empty or "driver_number" not in df.columns or "position" not in df.columns:
        return {}
    first_pos = df.sort_values("date").groupby("driver_number")["position"].first()
    return {int(k): int(v) for k, v in first_pos.items()}


def _get_final_positions(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> dict[int, int]:
    """Get final race positions from the /position endpoint (last recorded)."""
    cache_path = cache_dir / f"position_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/position?session_key={session_key}", cache_path)
    if not data:
        return {}
    df = pd.DataFrame(data)
    if df.empty or "driver_number" not in df.columns or "position" not in df.columns:
        return {}
    last_pos = df.sort_values("date").groupby("driver_number")["position"].last()
    return {int(k): int(v) for k, v in last_pos.items()}


def _count_overtakes(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> dict[int, int]:
    """Count overtakes per driver from position changes during the session."""
    cache_path = cache_dir / f"position_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/position?session_key={session_key}", cache_path)
    if not data:
        return {}
    df = pd.DataFrame(data)
    if df.empty or "driver_number" not in df.columns or "position" not in df.columns:
        return {}
    df = df.sort_values(["driver_number", "date"])
    overtakes: dict[int, int] = {}
    for drv, grp in df.groupby("driver_number"):
        positions = grp["position"].values
        gains = sum(1 for i in range(1, len(positions)) if positions[i] < positions[i - 1])
        overtakes[int(drv)] = gains
    return overtakes


def _get_race_control_dnfs(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> set[int]:
    """Identify DNF/retired drivers from race control messages."""
    cache_path = cache_dir / f"race_control_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/race_control?session_key={session_key}", cache_path)
    if not data:
        return set()
    dnf_drivers: set[int] = set()
    for msg in data:
        message_text = str(msg.get("message", "")).upper()
        driver_num = msg.get("driver_number")
        if driver_num is None:
            continue
        if any(kw in message_text for kw in ["RETIRED", "STOPPED", "OUT OF THE RACE", "DID NOT FINISH", "WILL NOT"]):
            dnf_drivers.add(int(driver_num))
    return dnf_drivers


def _get_dsq_drivers(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> set[int]:
    """Identify DSQ drivers from race control messages."""
    cache_path = cache_dir / f"race_control_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/race_control?session_key={session_key}", cache_path)
    if not data:
        return set()
    dsq_drivers: set[int] = set()
    for msg in data:
        message_text = str(msg.get("message", "")).upper()
        driver_num = msg.get("driver_number")
        if driver_num and "DISQUALIF" in message_text:
            dsq_drivers.add(int(driver_num))
    return dsq_drivers


def _get_weather(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> dict[str, Any]:
    """Get median weather conditions for a session."""
    cache_path = cache_dir / f"weather_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/weather?session_key={session_key}", cache_path)
    if not data:
        return {}
    df = pd.DataFrame(data)
    result: dict[str, Any] = {}
    if "air_temperature" in df.columns:
        result["air_temp_c"] = float(df["air_temperature"].median())
    if "track_temperature" in df.columns:
        result["track_temp_c"] = float(df["track_temperature"].median())
    if "humidity" in df.columns:
        result["humidity_pct"] = float(df["humidity"].median())
    if "wind_speed" in df.columns:
        result["wind_speed_ms"] = float(df["wind_speed"].median())
    if "rainfall" in df.columns:
        result["rainfall"] = bool(df["rainfall"].any())
    return result


def _get_pit_stops(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> dict[int, dict]:
    """Get pit stop data per driver (fastest and average duration in ms)."""
    cache_path = cache_dir / f"pit_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/pit?session_key={session_key}", cache_path)
    if not data:
        return {}
    df = pd.DataFrame(data)
    if df.empty or "driver_number" not in df.columns or "pit_duration" not in df.columns:
        return {}
    df["pit_duration"] = pd.to_numeric(df["pit_duration"], errors="coerce")
    result: dict[int, dict] = {}
    for drv, grp in df.groupby("driver_number"):
        durations = grp["pit_duration"].dropna()
        if not durations.empty:
            result[int(drv)] = {
                "fastest_ms": float(durations.min() * 1000),
                "avg_ms": float(durations.mean() * 1000),
            }
    return result


def _get_fastest_lap(sess: requests.Session, base_url: str, session_key: int, cache_dir: Path) -> int | None:
    """Get driver number with fastest lap from /laps endpoint."""
    cache_path = cache_dir / f"laps_{session_key}.json"
    data = _fetch_and_cache(sess, f"{base_url}/laps?session_key={session_key}&is_pit_out_lap=false", cache_path)
    if not data:
        return None
    df = pd.DataFrame(data)
    if df.empty or "driver_number" not in df.columns or "lap_duration" not in df.columns:
        return None
    df["lap_duration"] = pd.to_numeric(df["lap_duration"], errors="coerce")
    valid = df.dropna(subset=["lap_duration"])
    if valid.empty:
        return None
    fastest_idx = valid["lap_duration"].idxmin()
    return int(valid.loc[fastest_idx, "driver_number"])


def enrich_sessions_batch(
    sessions_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Enrich sessions DataFrame with OpenF1 data for 2023+ years.

    Backfills: grid_position, status (DNF/DSQ), positions_gained, overtakes,
    is_fastest_lap, weather fields, pit stop data.
    """
    if sessions_df.empty:
        return sessions_df

    out = sessions_df.copy()
    cache_dir = Path(cache_dir or CACHE_DIR_DEFAULT)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://api.openf1.org/v1"
    if config:
        base_url = config.get("data", {}).get("openf1_base_url", base_url)

    if years is None:
        years = [2023, 2024, 2025]

    sess = requests.Session()
    total_enriched = 0

    for year in years:
        year_sessions = fetch_sessions(year, config=config, session=sess)
        if year_sessions.empty:
            log.warning("No OpenF1 sessions for %d", year)
            continue

        year_mask = out["year"] == year
        if not year_mask.any():
            continue

        rounds_in_data = sorted(out.loc[year_mask, "round"].dropna().astype(int).unique())
        log.info("Enriching %d: %d rounds to process", year, len(rounds_in_data))

        for rnd in rounds_in_data:
            for st in ["qualifying", "race", "sprint"]:
                row_mask = year_mask & (out["round"] == rnd) & (out["session_type"] == st)
                if not row_mask.any():
                    continue

                name_map = {"race": "Race", "sprint": "Sprint", "qualifying": "Qualifying"}
                want = name_map.get(st, st)
                sk_rows = year_sessions[(year_sessions["round"] == rnd) & (year_sessions["session_name"] == want)]
                if sk_rows.empty:
                    continue
                sk = int(sk_rows.iloc[0]["session_key"])

                session_cache = cache_dir / str(year)
                session_cache.mkdir(parents=True, exist_ok=True)

                # Grid positions (first recorded position = grid)
                if st in ("race", "sprint"):
                    grid_map = _get_grid_positions(sess, base_url, sk, session_cache)
                    for drv_num, grid_pos in grid_map.items():
                        code = _number_to_code(drv_num)
                        if code is None:
                            continue
                        dmask = row_mask & (out["driver_code"] == code)
                        if dmask.any():
                            current_grid = out.loc[dmask, "grid_position"]
                            if current_grid.isna().all():
                                out.loc[dmask, "grid_position"] = grid_pos

                # Overtakes
                if st in ("race", "sprint"):
                    ov_map = _count_overtakes(sess, base_url, sk, session_cache)
                    for drv_num, ov_count in ov_map.items():
                        code = _number_to_code(drv_num)
                        if code is None:
                            continue
                        dmask = row_mask & (out["driver_code"] == code)
                        if dmask.any():
                            current_ov = out.loc[dmask, "overtakes"]
                            if current_ov.isna().all() or (current_ov == 0).all():
                                out.loc[dmask, "overtakes"] = ov_count

                # DNF/DSQ status
                if st in ("race", "sprint"):
                    dnf_drivers = _get_race_control_dnfs(sess, base_url, sk, session_cache)
                    dsq_drivers = _get_dsq_drivers(sess, base_url, sk, session_cache)
                    for drv_num in dnf_drivers:
                        code = _number_to_code(drv_num)
                        if code is None:
                            continue
                        dmask = row_mask & (out["driver_code"] == code)
                        if dmask.any() and out.loc[dmask, "status"].iloc[0] == "Finished":
                            out.loc[dmask, "status"] = "DNF"
                    for drv_num in dsq_drivers:
                        code = _number_to_code(drv_num)
                        if code is None:
                            continue
                        dmask = row_mask & (out["driver_code"] == code)
                        if dmask.any():
                            out.loc[dmask, "status"] = "DSQ"

                # Fastest lap
                if st == "race":
                    fl_driver = _get_fastest_lap(sess, base_url, sk, session_cache)
                    if fl_driver is not None:
                        fl_code = _number_to_code(fl_driver)
                        if fl_code:
                            out.loc[row_mask, "is_fastest_lap"] = False
                            dmask = row_mask & (out["driver_code"] == fl_code)
                            if dmask.any():
                                out.loc[dmask, "is_fastest_lap"] = True

                # Weather
                weather = _get_weather(sess, base_url, sk, session_cache)
                for col, val in weather.items():
                    if col in out.columns:
                        current = out.loc[row_mask, col]
                        if current.isna().all():
                            out.loc[row_mask, col] = val

                # Pit stops (race only, per-constructor aggregation)
                if st == "race":
                    pit_data = _get_pit_stops(sess, base_url, sk, session_cache)
                    for drv_num, pit_info in pit_data.items():
                        code = _number_to_code(drv_num)
                        if code is None:
                            continue
                        dmask = row_mask & (out["driver_code"] == code)
                        if dmask.any():
                            if out.loc[dmask, "fastest_pitstop_ms"].isna().all():
                                out.loc[dmask, "fastest_pitstop_ms"] = pit_info["fastest_ms"]
                            if out.loc[dmask, "avg_pitstop_ms"].isna().all():
                                out.loc[dmask, "avg_pitstop_ms"] = pit_info["avg_ms"]

                total_enriched += row_mask.sum()

    # Recompute positions_gained where we now have grid_position
    has_grid = out["grid_position"].notna() & out["finish_position"].notna()
    if has_grid.any():
        out.loc[has_grid, "positions_gained"] = (
            out.loc[has_grid, "grid_position"].astype(float) - out.loc[has_grid, "finish_position"].astype(float)
        )

    log.info("OpenF1 enrichment complete: %d row-sessions processed", total_enriched)
    return out
