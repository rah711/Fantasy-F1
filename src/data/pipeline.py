"""
Orchestrator: load Kaggle base, left-join TracingInsights and OpenF1,
compute fantasy points, validate joins (report mismatch counts, never drop rows),
and save sessions.parquet.

Modes: full (all years), incremental (from last run), 2026_only (future use).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.config import load_config
from src.data.kaggle_loader import load_kaggle_sessions
from src.data.openf1_enrichment import enrich_sessions_batch
from src.data.scoring import compute_fantasy_points
from src.data.tracing_archive_loader import load_tracing_archive_sessions
from src.data.tracing_loader import load_tracing_dotd, load_tracing_pitstops
from src.utils.logging import get_logger

log = get_logger(__name__)

Mode = Literal["full", "incremental", "2026_only"]


def run_pipeline(
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    mode: Mode = "full",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load all sources, left-join, compute fantasy points, validate, and save parquet.

    Args:
        config_path: Path to config.yaml (optional if config provided).
        output_path: Where to write sessions.parquet (default: data/processed/sessions.parquet).
        mode: "full" = all years from Kaggle, "incremental" = from last run, "2026_only" = 2026 only.
        config: Pre-loaded config dict (optional).

    Returns:
        DataFrame with unified columns plus fantasy_points_driver and constructor_bonus.
    """
    if config is None:
        config = load_config(str(config_path) if config_path else None)

    kaggle_dir = config.get("data", {}).get("kaggle_dir", "data/raw/kaggle")
    tracing_dir = config.get("data", {}).get("tracing_insights_dir", "data/raw/tracing_insights")
    tracing_archive_dir = config.get("data", {}).get("tracing_archive_dir", "data/raw/tracing_archive")
    fastf1_cache_dir = config.get("data", {}).get("fastf1_cache_dir", "data/cache")
    start_year = config.get("data", {}).get("historical_start_year", 2020)

    year_filter = None
    if mode == "2026_only":
        year_filter = 2026
    elif mode == "full":
        year_filter = None

    # 1. Kaggle base
    base = load_kaggle_sessions(kaggle_dir, year=year_filter, config=config)
    if base.empty:
        log.warning("Kaggle loader returned no rows.")
        base = pd.DataFrame(columns=list({
            "year", "round", "circuit_id", "session_type", "driver_code", "constructor_id",
            "grid_position", "finish_position", "status", "points_official", "finish_time_ms",
            "fastest_lap_rank", "q1_time_ms", "q2_time_ms", "q3_time_ms", "positions_gained",
            "overtakes", "is_fastest_lap", "is_dotd", "fastest_pitstop_ms", "avg_pitstop_ms",
            "air_temp_c", "track_temp_c", "humidity_pct", "wind_speed_ms", "rainfall",
        }))

    # 1b. Augment with TracingInsights historical archives.
    # Load for years before Kaggle coverage AND for years where Kaggle is incomplete
    # (fewer rounds than expected). This ensures 2025 gets full-season coverage
    # from TracingInsights even when the Kaggle dataset only has partial data.
    archive_end_year = config.get("season", {}).get("year", 2026) - 1
    archive_start_year = max(2018, int(start_year))
    if archive_start_year <= archive_end_year:
        archive_df = load_tracing_archive_sessions(
            start_year=archive_start_year,
            end_year=archive_end_year,
            cache_dir=tracing_archive_dir,
            fastf1_cache_dir=fastf1_cache_dir,
            refresh=False,
        )
        if not archive_df.empty:
            base = pd.concat([archive_df, base], ignore_index=True)
            # When both sources have rows for the same (year, round, session, driver),
            # prefer Kaggle rows (which appear last after concat) since they have
            # richer metadata (grid positions, times, etc.).
            dedupe_keys = ["year", "round", "session_type", "driver_code"]
            base = base.drop_duplicates(subset=dedupe_keys, keep="last")
            log.info(
                "Augmented base with archive rows: +%d (total %d)",
                len(archive_df),
                len(base),
            )

    n_base = len(base)
    log.info("Base rows after archive merge: %d", n_base)

    # 2. Left-join DOTD (TracingInsights)
    dotd = load_tracing_dotd(tracing_dir, year=year_filter, config=config)
    if not dotd.empty and "driver_code" in dotd.columns:
        base = base.merge(
            dotd[["year", "round", "driver_code", "is_dotd"]],
            on=["year", "round", "driver_code"],
            how="left",
            suffixes=("", "_dotd"),
        )
        if "is_dotd_dotd" in base.columns:
            base["is_dotd"] = base["is_dotd_dotd"].fillna(base.get("is_dotd", False))
            base = base.drop(columns=["is_dotd_dotd"], errors="ignore")
        matched = base["is_dotd"].notna().sum() if "is_dotd" in base.columns else 0
        log.info("DOTD join: %d rows matched", matched)
    else:
        if "is_dotd" not in base.columns:
            base["is_dotd"] = False

    # 3. Left-join pitstops (TracingInsights) — constructor-level, so merge on year, round, constructor_id
    pitstops = load_tracing_pitstops(tracing_dir, year=year_filter, config=config)
    if not pitstops.empty and "constructor_id" in pitstops.columns:
        base = base.merge(
            pitstops[["year", "round", "constructor_id", "fastest_pitstop_ms", "avg_pitstop_ms"]],
            on=["year", "round", "constructor_id"],
            how="left",
            suffixes=("", "_tracing"),
        )
        if "fastest_pitstop_ms_tracing" in base.columns:
            base["fastest_pitstop_ms"] = base["fastest_pitstop_ms_tracing"].fillna(base.get("fastest_pitstop_ms"))
            base = base.drop(columns=["fastest_pitstop_ms_tracing"], errors="ignore")
        if "avg_pitstop_ms_tracing" in base.columns:
            base["avg_pitstop_ms"] = base["avg_pitstop_ms_tracing"].fillna(base.get("avg_pitstop_ms"))
            base = base.drop(columns=["avg_pitstop_ms_tracing"], errors="ignore")
        matched = base["fastest_pitstop_ms"].notna().sum() if "fastest_pitstop_ms" in base.columns else 0
        log.info("Pitstops join: %d rows with pitstop data", matched)
    else:
        if "fastest_pitstop_ms" not in base.columns:
            base["fastest_pitstop_ms"] = pd.NA
        if "avg_pitstop_ms" not in base.columns:
            base["avg_pitstop_ms"] = pd.NA

    # 4. OpenF1 enrichment: backfill grid, DNF status, overtakes, weather, pit stops for 2023+
    openf1_cache = config.get("data", {}).get("openf1_cache_dir", "data/cache/openf1_enrichment")
    openf1_years = [y for y in range(2023, archive_end_year + 1) if y <= 2025]
    if openf1_years:
        log.info("Running OpenF1 enrichment for years %s", openf1_years)
        base = enrich_sessions_batch(base, config=config, cache_dir=openf1_cache, years=openf1_years)

    # 5. Compute fantasy points
    base = compute_fantasy_points(config, base)

    # 6. Validate: never drop rows; report mismatch counts
    log.info("Pipeline output rows: %d (no rows dropped)", len(base))
    if n_base != len(base):
        log.warning("Row count changed from %d to %d (expected no change with left joins)", n_base, len(base))

    # 7. Save parquet
    if output_path is None:
        output_path = Path(config.get("data", {}).get("processed_dir", "data/processed")) / "sessions.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output_path, index=False)
    log.info("Saved %s", output_path)

    return base
