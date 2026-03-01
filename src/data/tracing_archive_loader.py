"""
Load historical sessions (2018+) from TracingInsights seasonal GitHub archives.

This loader fetches per-season repositories from TracingInsights-Archive and
builds unified rows from `drivers.json` files in each race weekend session
folder (Qualifying, Sprint, Race).

The output can be concatenated with Kaggle sessions so model training can use
more history than 2024+.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from src.data.schema import normalise_circuit, normalise_constructor
from src.utils.logging import get_logger

log = get_logger(__name__)

GITHUB_API_BASE = "https://api.github.com/repos/TracingInsights-Archive"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/TracingInsights-Archive"

SESSION_DIR_TO_TYPE = {
    "Qualifying": "qualifying",
    "Sprint": "sprint",
    "Race": "race",
}


def _get_json(url: str, timeout: int = 25) -> dict | list | None:
    """HTTP GET JSON helper with graceful failure."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _norm_event_name(name: str) -> str:
    s = name.lower().replace("formula 1", "")
    for ch in ["-", "_", "(", ")", "'", ".", ",", ":"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def _event_round_map(year: int, fastf1_cache_dir: str | Path) -> dict[str, int]:
    """Build event_name -> round_number map from FastF1 schedule for a year."""
    try:
        import fastf1

        cache_dir = Path(fastf1_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        log.warning("FastF1 schedule unavailable for %s: %s", year, e)
        return {}

    mapping: dict[str, int] = {}
    for _, row in schedule.iterrows():
        event_name = str(row.get("EventName", "")).strip()
        round_num = row.get("RoundNumber")
        try:
            if event_name and pd.notna(round_num):
                mapping[_norm_event_name(event_name)] = int(round_num)
        except (TypeError, ValueError):
            continue
    return mapping


def _round_for_event(event_name: str, round_map: dict[str, int]) -> int | None:
    """Resolve round number for event name using exact/partial normalized matching."""
    k = _norm_event_name(event_name)
    if k in round_map:
        return round_map[k]
    for key, val in round_map.items():
        if k in key or key in k:
            return val
    return None


def _safe_constructor_id(team_name: str) -> str:
    """Normalise constructor with fallback to snake_case string."""
    try:
        return normalise_constructor(team_name)
    except Exception:
        return team_name.strip().lower().replace("-", " ").replace(" ", "_")


def _safe_circuit_id(event_name: str) -> str:
    """Normalise circuit with fallback to event slug."""
    try:
        return normalise_circuit(event_name)
    except Exception:
        slug = (
            event_name.lower()
            .replace("grand prix", "")
            .replace("prix", "")
            .replace("-", " ")
            .replace(" ", "_")
        )
        return "_".join([p for p in slug.split("_") if p])


def _year_cache_path(cache_dir: Path, year: int) -> Path:
    return cache_dir / f"sessions_{year}.parquet"


def load_tracing_archive_year(
    year: int,
    cache_dir: str | Path = "data/raw/tracing_archive",
    fastf1_cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load one season from TracingInsights-Archive/<year> into unified rows.

    Rows are built from session-level `drivers.json` where ordering represents
    classification order for that session.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _year_cache_path(cache_dir, year)

    if cache_path.exists() and not refresh:
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    root_url = f"{GITHUB_API_BASE}/{year}/contents?ref=main"
    root_items = _get_json(root_url)
    if not isinstance(root_items, list):
        log.warning("No archive root for year %s", year)
        return pd.DataFrame()

    gp_dirs = [
        item["name"]
        for item in root_items
        if item.get("type") == "dir" and "prix" in str(item.get("name", "")).lower()
    ]
    if not gp_dirs:
        log.warning("No race weekend folders found for year %s", year)
        return pd.DataFrame()

    round_map = _event_round_map(year, fastf1_cache_dir)
    rows: list[dict[str, Any]] = []

    for gp_name in gp_dirs:
        round_num = _round_for_event(gp_name, round_map)
        if round_num is None:
            continue

        circuit_id = _safe_circuit_id(gp_name)

        for session_dir, session_type in SESSION_DIR_TO_TYPE.items():
            url = (
                f"{GITHUB_RAW_BASE}/{year}/main/"
                f"{quote(gp_name)}/{quote(session_dir)}/drivers.json"
            )
            data = _get_json(url)
            if not isinstance(data, dict):
                continue

            drivers = data.get("drivers")
            if not isinstance(drivers, list):
                continue

            for idx, entry in enumerate(drivers, start=1):
                driver_code = str(entry.get("driver", "")).strip().upper()
                team_name = str(entry.get("team", "")).strip()
                if not driver_code or not team_name:
                    continue

                constructor_id = _safe_constructor_id(team_name)
                finish_position = idx
                grid_position = idx if session_type == "qualifying" else None

                rows.append(
                    {
                        "year": year,
                        "round": int(round_num),
                        "circuit_id": circuit_id,
                        "session_type": session_type,
                        "driver_code": driver_code,
                        "constructor_id": constructor_id,
                        "grid_position": grid_position,
                        "finish_position": finish_position,
                        "status": "Finished",
                        "points_official": None,
                        "finish_time_ms": None,
                        "fastest_lap_rank": None,
                        "q1_time_ms": None,
                        "q2_time_ms": None,
                        "q3_time_ms": None,
                        "positions_gained": (
                            (grid_position - finish_position)
                            if grid_position is not None and finish_position is not None
                            else None
                        ),
                        "overtakes": None,
                        "is_fastest_lap": False,
                        "is_dotd": False,
                        "fastest_pitstop_ms": None,
                        "avg_pitstop_ms": None,
                        "air_temp_c": None,
                        "track_temp_c": None,
                        "humidity_pct": None,
                        "wind_speed_ms": None,
                        "rainfall": None,
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["year", "round", "session_type", "driver_code"]).reset_index(drop=True)
        try:
            out.to_parquet(cache_path, index=False)
        except Exception as e:
            log.warning("Could not cache archive year %s: %s", year, e)

    log.info("Tracing archive %s rows: %s", year, len(out))
    return out


def load_tracing_archive_sessions(
    start_year: int = 2018,
    end_year: int = 2023,
    cache_dir: str | Path = "data/raw/tracing_archive",
    fastf1_cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load multiple seasons from TracingInsights archive and concatenate."""
    if end_year < start_year:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        df = load_tracing_archive_year(
            year=year,
            cache_dir=cache_dir,
            fastf1_cache_dir=fastf1_cache_dir,
            refresh=refresh,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
