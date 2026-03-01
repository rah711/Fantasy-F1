"""
Load Driver of the Day (DOTD) and pitstop data from TracingInsights GitHub.

Downloads DOTD JSON and PitStops-Archive JSONs from GitHub raw URLs and caches
them locally in data/raw/tracing_insights/. Returns DataFrames keyed by year/round
for left-join in the pipeline.

Usage:
    from src.data.tracing_loader import load_tracing_dotd, load_tracing_pitstops
    dotd_df = load_tracing_dotd(tracing_dir, year=2024)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.logging import get_logger

log = get_logger(__name__)

# Default GitHub raw base URLs (TracingInsights-style archives)
DOTD_BASE_URL = "https://raw.githubusercontent.com/TracingInsights-Archive/{year}/main/dotd.json"
PITSTOPS_ARCHIVE_URL = "https://raw.githubusercontent.com/TracingInsights-Archive/{year}/main/pitstops.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fetch_json_cached(
    url: str,
    cache_path: Path,
    session: requests.Session | None = None,
) -> dict | list | None:
    """Fetch JSON from URL; save to cache_path if successful."""
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            log.debug("Cache read failed %s: %s", cache_path, e)

    sess = session or requests.Session()
    try:
        r = sess.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        _ensure_dir(cache_path.parent)
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=0)
        return data
    except Exception as e:
        log.warning("Fetch failed %s: %s", url, e)
        return None


def load_tracing_dotd(
    tracing_dir: str | Path,
    year: int | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load DOTD data. Returns DataFrame with year, round, driver_code (or driver_id), is_dotd.

    Caches per-year dotd.json in tracing_dir. If year is None, loads all available cached years.
    """
    if config and not tracing_dir:
        tracing_dir = config.get("data", {}).get("tracing_insights_dir", "data/raw/tracing_insights")
    tracing_dir = Path(tracing_dir)
    _ensure_dir(tracing_dir)

    years = [year] if year is not None else [2020, 2021, 2022, 2023, 2024, 2025]
    rows = []
    for y in years:
        url = DOTD_BASE_URL.format(year=y)
        cache_path = tracing_dir / f"{y}_dotd.json"
        data = _fetch_json_cached(url, cache_path)
        if data is None:
            continue
        if isinstance(data, list):
            for item in data:
                round_num = item.get("round") or item.get("race_round") or item.get("roundNumber")
                driver = item.get("driver_code") or item.get("driverCode") or item.get("driver") or item.get("winner")
                if round_num is not None and driver:
                    rows.append({"year": y, "round": int(round_num), "driver_code": str(driver).upper()[:3], "is_dotd": True})
        elif isinstance(data, dict):
            for round_key, driver in data.items():
                try:
                    round_num = int(round_key)
                except (TypeError, ValueError):
                    continue
                if driver:
                    rows.append({"year": y, "round": round_num, "driver_code": str(driver).upper()[:3], "is_dotd": True})
    if not rows:
        return pd.DataFrame(columns=["year", "round", "driver_code", "is_dotd"])
    return pd.DataFrame(rows).drop_duplicates(subset=["year", "round", "driver_code"])


def load_tracing_pitstops(
    tracing_dir: str | Path,
    year: int | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load pitstop data. Returns DataFrame with year, round, constructor_id/driver_code, fastest_pitstop_ms, avg_pitstop_ms.

    Caches per-year pitstops.json in tracing_dir.
    """
    if config and not tracing_dir:
        tracing_dir = config.get("data", {}).get("tracing_insights_dir", "data/raw/tracing_insights")
    tracing_dir = Path(tracing_dir)
    _ensure_dir(tracing_dir)

    years = [year] if year is not None else [2020, 2021, 2022, 2023, 2024, 2025]
    rows = []
    for y in years:
        url = PITSTOPS_ARCHIVE_URL.format(year=y)
        cache_path = tracing_dir / f"{y}_pitstops.json"
        data = _fetch_json_cached(url, cache_path)
        if data is None:
            continue
        if isinstance(data, list):
            for item in data:
                round_num = item.get("round") or item.get("race_round") or item.get("roundNumber")
                team = item.get("constructor_id") or item.get("team") or item.get("constructor")
                fastest_ms = item.get("fastest_pitstop_ms") or item.get("fastest_ms") or item.get("duration_ms")
                if fastest_ms is not None and isinstance(fastest_ms, (int, float)):
                    fastest_ms = float(fastest_ms)
                elif fastest_ms is not None and isinstance(fastest_ms, str) and ":" in fastest_ms:
                    parts = fastest_ms.replace(",", ".").split(":")
                    if len(parts) == 2:
                        fastest_ms = int(parts[0]) * 60 * 1000 + float(parts[1]) * 1000
                    else:
                        fastest_ms = None
                avg_ms = item.get("avg_pitstop_ms") or item.get("average_ms")
                if round_num is not None and team:
                    rows.append({
                        "year": y,
                        "round": int(round_num),
                        "constructor_id": str(team).lower().replace(" ", "_"),
                        "fastest_pitstop_ms": fastest_ms,
                        "avg_pitstop_ms": float(avg_ms) if avg_ms is not None else None,
                    })
        elif isinstance(data, dict):
            for round_key, stops in data.items():
                try:
                    round_num = int(round_key)
                except (TypeError, ValueError):
                    continue
                if isinstance(stops, list):
                    for s in stops:
                        team = s.get("constructor_id") or s.get("team") or s.get("constructor")
                        fastest_ms = s.get("fastest_pitstop_ms") or s.get("duration_ms")
                        if team:
                            rows.append({
                                "year": y,
                                "round": round_num,
                                "constructor_id": str(team).lower().replace(" ", "_"),
                                "fastest_pitstop_ms": float(fastest_ms) if fastest_ms is not None else None,
                                "avg_pitstop_ms": s.get("avg_pitstop_ms"),
                            })
    if not rows:
        return pd.DataFrame(columns=["year", "round", "constructor_id", "fastest_pitstop_ms", "avg_pitstop_ms"])
    return pd.DataFrame(rows)
