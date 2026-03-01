"""
Load and unify Kaggle F1 CSV data into the canonical session format.

Reads CSVs from data/raw/kaggle/ (or config data.kaggle_dir), joins via
race_id / driver/team, normalises names via schema, and outputs a DataFrame
with UNIFIED_COLUMNS. Handles \\N nulls and converts qualifying time strings
(e.g. "1:39.103") to milliseconds.

Usage:
    from src.data.kaggle_loader import load_kaggle_sessions
    df = load_kaggle_sessions(kaggle_dir="data/raw/kaggle")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import (
    UNIFIED_COLUMNS,
    normalise_circuit,
    normalise_constructor,
    normalise_driver,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# CSV null sentinel used by Kaggle/Ergast
NA_VALUES = ["\\N", ""]


def _qualifying_time_to_ms(s: Any) -> float | None:
    """Convert qualifying time string 'M:SS.mmm' or 'M:SS.mm' to milliseconds."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    raw = str(s).strip()
    if "+" in raw or "Pole" in raw or raw.upper() in ("DNF", "DNS", "NC"):
        return None
    parts = raw.replace(",", ".").split(":")
    if len(parts) == 3:
        # H:MM:SS.mmm
        h, m, sec = parts
        return int(h) * 3600 * 1000 + int(m) * 60 * 1000 + float(sec) * 1000
    if len(parts) == 2:
        # M:SS.mmm
        m, sec = parts
        return int(m) * 60 * 1000 + float(sec) * 1000
    return None


def _race_status_from_time(race_time: Any) -> str:
    """Infer status from race_time column (e.g. 'DNF', '+1 lap', '1:35:17.520')."""
    if race_time is None or (isinstance(race_time, float) and pd.isna(race_time)):
        return "DNF"
    s = str(race_time).strip()
    if s.upper() == "DNF":
        return "DNF"
    if s.upper() in ("DNS", "DSQ", "EXCLUDED", "DISQUALIFIED"):
        return "DSQ"
    if s.startswith("+") and ("lap" in s.lower() or "laps" in s.lower()):
        return "Finished"
    if s.replace(".", "").replace(":", "").isdigit() or ":" in s:
        return "Finished"
    return "Finished"


def _to_int(value: Any) -> int | None:
    """Best-effort int conversion for values like 1, '1', ' 1 '."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _assign_round_from_circuit_order(df: pd.DataFrame, circuit_col: str = "circuit") -> pd.DataFrame:
    """Assign round numbers by first appearance order of circuit names."""
    out = df.copy()
    if circuit_col not in out.columns:
        return out
    order = pd.Series(out[circuit_col].astype(str)).drop_duplicates().tolist()
    mapping = {name: idx + 1 for idx, name in enumerate(order)}
    out["round"] = out[circuit_col].astype(str).map(mapping)
    return out


def _infer_year_from_path(csv_path: Path) -> int | None:
    """Infer season year from filename like f1_2024_race_results.csv."""
    name = csv_path.stem.lower()
    if "2024" in name:
        return 2024
    if "2023" in name:
        return 2023
    if "2025" in name:
        return 2025
    if "2022" in name:
        return 2022
    if "2021" in name:
        return 2021
    if "2020" in name:
        return 2020
    return None


def _normalise_race_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column name variants to the names used downstream (position, driver_name, team, circuit, race_time, points, fastest_lap)."""
    aliases = [
        ("position", ["position", "pos", "Position", "Pos", "finish_position"]),
        ("driver_name", ["driver_name", "driver", "Driver", "DriverName"]),
        ("team", ["team", "Team", "constructor", "Constructor", "team_name"]),
        ("circuit", ["circuit", "Circuit", "circuit_name", "track", "Track", "grand_prix"]),
        ("race_time", ["race_time", "time", "Time", "result", "Time/Retired"]),
        ("points", ["points", "Points", "pts"]),
        ("fastest_lap", ["fastest_lap", "FastestLap", "fastest lap", "Set Fastest Lap"]),
        ("grid_position", ["grid_position", "Starting Grid", "starting_grid", "Grid", "grid"]),
        ("race_id", ["race_id", "round", "Round", "race", "grand_prix_id"]),
    ]
    out = df.copy()
    for canonical, candidates in aliases:
        if canonical in out.columns:
            continue
        for c in candidates:
            if c in out.columns:
                out = out.rename(columns={c: canonical})
                break
    return out


def _normalise_qualifying_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map qualifying/sprint-qualifying CSV columns to canonical names."""
    out = df.copy()
    aliases = [
        ("qualifying_position", ["qualifying_position", "Position", "position", "Pos", "pos"]),
        ("driver_name", ["driver_name", "Driver", "driver"]),
        ("team", ["team", "Team", "constructor"]),
        ("circuit", ["circuit", "Circuit", "Track", "track"]),
        ("q1_time", ["q1_time", "Q1", "q1"]),
        ("q2_time", ["q2_time", "Q2", "q2"]),
        ("q3_time", ["q3_time", "Q3", "q3"]),
        ("race_id", ["race_id", "round", "Round", "race", "grand_prix_id"]),
    ]
    for canonical, candidates in aliases:
        if canonical in out.columns:
            continue
        for c in candidates:
            if c in out.columns:
                out = out.rename(columns={c: canonical})
                break
    return out


def _load_csv_patterns(kaggle_dir: Path, patterns: list[str], year: int | None) -> list[pd.DataFrame]:
    """Load all matching CSVs for a year, deduplicating paths."""
    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for path in kaggle_dir.glob(pattern):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            y = _infer_year_from_path(path)
            if year is not None and y != year:
                continue
            if y is None:
                continue
            try:
                df = pd.read_csv(path, na_values=NA_VALUES)
            except Exception as e:
                log.warning("Skip %s: %s", path.name, e)
                continue
            df["year"] = y
            frames.append(df)
    return frames


def load_kaggle_race_results(kaggle_dir: Path, year: int | None) -> pd.DataFrame:
    """Load race results CSVs (e.g. f1_2024_race_results.csv, f1-race-result-2025.csv)."""
    kaggle_dir = Path(kaggle_dir)
    if not kaggle_dir.exists():
        log.warning("Kaggle dir does not exist: %s", kaggle_dir)
        return pd.DataFrame()

    # Match f1_2024_race_results.csv, F1_2025_RaceResults.csv, f1-race-result-2025.csv, etc.
    patterns = [
        "f1_*_race_results*.csv", "f1*race*result*.csv",
        "*race*result*2025*.csv", "*2025*race*.csv",
        "*Race*Result*2025*.csv", "*2025*Race*.csv", "F1_*_Race*.csv",
    ]
    frames = []
    for df in _load_csv_patterns(kaggle_dir, patterns, year):
        y = int(df["year"].iloc[0]) if not df.empty else None
        if y is None:
            continue
        df = _normalise_race_results_columns(df)
        if "race_id" in df.columns:
            df["round"] = df["race_id"]
        elif "round" not in df.columns and "circuit" in df.columns:
            df = _assign_round_from_circuit_order(df, circuit_col="circuit")
        elif "round" not in df.columns:
            df["round"] = range(1, len(df) + 1)  # fallback
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_kaggle_qualifying(kaggle_dir: Path, year: int | None) -> pd.DataFrame:
    """Load qualifying CSVs (e.g. f1_qualifying_results_2024.csv, *qualifying*2025*.csv)."""
    kaggle_dir = Path(kaggle_dir)
    if not kaggle_dir.exists():
        return pd.DataFrame()

    patterns = [
        "f1_*qualifying*.csv", "f1*qualifying*.csv",
        "*qualifying*2025*.csv", "*2025*qualifying*.csv",
        "*Qualifying*2025*.csv", "*2025*Qualifying*.csv", "F1_*_Qualifying*.csv",
    ]
    frames = []
    for df in _load_csv_patterns(kaggle_dir, patterns, year):
        df = _normalise_qualifying_columns(df)
        if "race_id" in df.columns:
            df["round"] = df["race_id"]
        elif "round" not in df.columns and "circuit" in df.columns:
            df = _assign_round_from_circuit_order(df, circuit_col="circuit")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_kaggle_sprint_results(kaggle_dir: Path, year: int | None) -> pd.DataFrame:
    """Load sprint results CSVs (e.g. F1_2025_SprintResults.csv)."""
    kaggle_dir = Path(kaggle_dir)
    if not kaggle_dir.exists():
        return pd.DataFrame()

    patterns = [
        "*sprint*result*.csv", "*Sprint*Result*.csv",
        "*2025*Sprint*.csv", "F1_*_SprintResults*.csv",
    ]
    frames = []
    for df in _load_csv_patterns(kaggle_dir, patterns, year):
        # Sprint results schema is close to race results
        df = _normalise_race_results_columns(df)
        if "race_id" in df.columns:
            df["round"] = df["race_id"]
        elif "round" not in df.columns and "circuit" in df.columns:
            df = _assign_round_from_circuit_order(df, circuit_col="circuit")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_kaggle_sprint_qualifying(kaggle_dir: Path, year: int | None) -> pd.DataFrame:
    """Load sprint qualifying CSVs (e.g. F1_2025_SprintQualifyingResults.csv)."""
    kaggle_dir = Path(kaggle_dir)
    if not kaggle_dir.exists():
        return pd.DataFrame()

    patterns = [
        "*sprint*qualifying*.csv", "*Sprint*Qualifying*.csv",
        "*2025*SprintQualifying*.csv", "F1_*_SprintQualifying*.csv",
    ]
    frames = []
    for df in _load_csv_patterns(kaggle_dir, patterns, year):
        df = _normalise_qualifying_columns(df)
        if "race_id" in df.columns:
            df["round"] = df["race_id"]
        elif "round" not in df.columns and "circuit" in df.columns:
            df = _assign_round_from_circuit_order(df, circuit_col="circuit")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalise_driver_safe(name: Any) -> str | None:
    try:
        return normalise_driver(str(name).strip()) if name is not None and not (isinstance(name, float) and pd.isna(name)) else None
    except KeyError:
        return None


def _normalise_constructor_safe(team: Any) -> str | None:
    try:
        return normalise_constructor(str(team).strip()) if team is not None and not (isinstance(team, float) and pd.isna(team)) else None
    except KeyError:
        return None


def _normalise_circuit_safe(circuit: Any) -> str | None:
    try:
        return normalise_circuit(str(circuit).strip()) if circuit is not None and not (isinstance(circuit, float) and pd.isna(circuit)) else None
    except KeyError:
        return None


def load_kaggle_sessions(
    kaggle_dir: str | Path,
    year: int | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load all Kaggle session data and return one DataFrame with UNIFIED_COLUMNS.

    Args:
        kaggle_dir: Path to data/raw/kaggle (or use config['data']['kaggle_dir'] if config given).
        year: If set, only load this season; otherwise load all available.
        config: Optional config dict (used for kaggle_dir if kaggle_dir not provided).

    Returns:
        DataFrame with columns matching UNIFIED_COLUMNS. One row per driver per session
        (qualifying, sprint qualifying+results, and race where available).
    """
    if config and not kaggle_dir:
        kaggle_dir = config.get("data", {}).get("kaggle_dir", "data/raw/kaggle")
    kaggle_dir = Path(kaggle_dir)

    race_df = load_kaggle_race_results(kaggle_dir, year)
    qual_df = load_kaggle_qualifying(kaggle_dir, year)
    sprint_df = load_kaggle_sprint_results(kaggle_dir, year)
    sprint_qual_df = load_kaggle_sprint_qualifying(kaggle_dir, year)

    rows = []

    # Build lookup for sprint-grid positions from sprint qualifying.
    sprint_grid_map: dict[tuple[int, int, str], int] = {}
    if not sprint_qual_df.empty and "qualifying_position" in sprint_qual_df.columns:
        for _, r in sprint_qual_df.iterrows():
            code = _normalise_driver_safe(r.get("driver_name"))
            if not code:
                continue
            yr = _to_int(r.get("year"))
            rnd = _to_int(r.get("round"))
            pos = _to_int(r.get("qualifying_position"))
            if yr is None or rnd is None or pos is None:
                continue
            sprint_grid_map[(yr, rnd, code)] = pos

    # Qualifying rows
    if not qual_df.empty and "qualifying_position" in qual_df.columns:
        for _, r in qual_df.iterrows():
            driver_code = _normalise_driver_safe(r.get("driver_name"))
            constructor_id = _normalise_constructor_safe(r.get("team"))
            circuit_id = _normalise_circuit_safe(r.get("circuit"))
            if not driver_code or not constructor_id or not circuit_id:
                continue
            q1_ms = _qualifying_time_to_ms(r.get("q1_time"))
            q2_ms = _qualifying_time_to_ms(r.get("q2_time"))
            q3_ms = _qualifying_time_to_ms(r.get("q3_time"))
            finish_position = _to_int(r.get("qualifying_position"))
            rows.append({
                "year": int(r["year"]),
                "round": int(r["round"]),
                "circuit_id": circuit_id,
                "session_type": "qualifying",
                "driver_code": driver_code,
                "constructor_id": constructor_id,
                "grid_position": finish_position,
                "finish_position": finish_position,
                "status": "",
                "points_official": None,
                "finish_time_ms": None,
                "fastest_lap_rank": None,
                "q1_time_ms": q1_ms,
                "q2_time_ms": q2_ms,
                "q3_time_ms": q3_ms,
                "positions_gained": None,
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
            })

    # Race rows
    if not race_df.empty and "position" in race_df.columns:
        for _, r in race_df.iterrows():
            driver_code = _normalise_driver_safe(r.get("driver_name"))
            constructor_id = _normalise_constructor_safe(r.get("team"))
            circuit_id = _normalise_circuit_safe(r.get("circuit"))
            if not driver_code or not constructor_id or not circuit_id:
                continue
            finish_position = _to_int(r.get("position"))
            grid_position = _to_int(r.get("grid_position"))
            if grid_position is None:
                grid_position = finish_position
            status = _race_status_from_time(r.get("race_time"))
            is_fl = str(r.get("fastest_lap", "")).strip().lower() == "yes"
            rows.append({
                "year": int(r["year"]),
                "round": int(r["round"]),
                "circuit_id": circuit_id,
                "session_type": "race",
                "driver_code": driver_code,
                "constructor_id": constructor_id,
                "grid_position": grid_position,
                "finish_position": finish_position,
                "status": status,
                "points_official": r.get("points"),
                "finish_time_ms": None,
                "fastest_lap_rank": None,
                "q1_time_ms": None,
                "q2_time_ms": None,
                "q3_time_ms": None,
                "positions_gained": (grid_position - finish_position) if grid_position is not None and finish_position is not None else None,
                "overtakes": None,
                "is_fastest_lap": is_fl,
                "is_dotd": False,
                "fastest_pitstop_ms": None,
                "avg_pitstop_ms": None,
                "air_temp_c": None,
                "track_temp_c": None,
                "humidity_pct": None,
                "wind_speed_ms": None,
                "rainfall": None,
            })

    # Sprint rows
    if not sprint_df.empty and "position" in sprint_df.columns:
        for _, r in sprint_df.iterrows():
            driver_code = _normalise_driver_safe(r.get("driver_name"))
            constructor_id = _normalise_constructor_safe(r.get("team"))
            circuit_id = _normalise_circuit_safe(r.get("circuit"))
            if not driver_code or not constructor_id or not circuit_id:
                continue
            year_i = _to_int(r.get("year"))
            round_i = _to_int(r.get("round"))
            if year_i is None or round_i is None:
                continue
            finish_position = _to_int(r.get("position"))
            grid_position = _to_int(r.get("grid_position"))
            if grid_position is None:
                grid_position = sprint_grid_map.get((year_i, round_i, driver_code))
            status = _race_status_from_time(r.get("race_time"))
            is_fl = str(r.get("fastest_lap", "")).strip().lower() == "yes"
            rows.append({
                "year": year_i,
                "round": round_i,
                "circuit_id": circuit_id,
                "session_type": "sprint",
                "driver_code": driver_code,
                "constructor_id": constructor_id,
                "grid_position": grid_position,
                "finish_position": finish_position,
                "status": status,
                "points_official": r.get("points"),
                "finish_time_ms": None,
                "fastest_lap_rank": None,
                "q1_time_ms": None,
                "q2_time_ms": None,
                "q3_time_ms": None,
                "positions_gained": (grid_position - finish_position) if grid_position is not None and finish_position is not None else None,
                "overtakes": None,
                "is_fastest_lap": is_fl,
                "is_dotd": False,
                "fastest_pitstop_ms": None,
                "avg_pitstop_ms": None,
                "air_temp_c": None,
                "track_temp_c": None,
                "humidity_pct": None,
                "wind_speed_ms": None,
                "rainfall": None,
            })

    if not rows:
        log.warning("No Kaggle rows produced; check kaggle_dir and CSV columns.")
        return pd.DataFrame(columns=list(UNIFIED_COLUMNS.keys()))

    out = pd.DataFrame(rows)
    log.info("Kaggle loader produced %d rows", len(out))
    return out
