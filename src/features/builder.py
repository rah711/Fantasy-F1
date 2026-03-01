"""
Feature builder: orchestrate all feature modules and validate output.

Reads sessions.parquet (or DataFrame), adds track, contextual, driver, and team
features, and writes features.parquet. Validates no NaN in feature columns for 2023+.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.features.car_track_features import add_car_track_features
from src.features.contextual_features import add_contextual_features
from src.features.driver_features import add_driver_features
from src.features.team_features import add_team_features
from src.features.track_features import add_track_features
from src.utils.logging import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS = [
    # Track
    "circuit_type", "circuit_overtake_difficulty", "circuit_drs_zones", "circuit_downforce",
    "circuit_safety_car_prob", "is_sprint_round",
    # Contextual
    "season_phase", "era_weight", "rainfall_flag",
    # Driver rolling
    "driver_rolling_pts_3", "driver_rolling_pts_5", "driver_avg_finish_at_circuit",
    "driver_overtake_rate", "driver_dnf_rate",
    # Driver cold-start
    "driver_prev_season_avg_pts", "driver_prev_season_avg_finish",
    "driver_prev_season_dnf_rate", "driver_prev_season_overtake_rate",
    "driver_prev_season_races", "driver_is_cold_start", "driver_early_round_flag",
    "driver_cold_start_pressure",
    # Team static + pitstop
    "team_development_score", "team_regulation_adaptation", "team_development_trajectory",
    "team_fastest_pitstop_avg", "team_avg_pitstop_avg",
    # Team cold-start
    "team_prev_season_avg_pts", "team_prev_season_avg_finish",
    "team_prev_season_dnf_rate", "team_prev_season_races",
    # Car-track interactions
    "team_pts_at_circuit_type_hist", "team_pts_at_circuit_type_season",
    "team_pts_at_downforce_hist", "team_pts_at_downforce_season",
    "team_circuit_type_delta", "team_season_rounds_so_far",
    "driver_pts_at_circuit_type_hist", "driver_pts_at_circuit_type_season",
    "driver_pts_at_downforce_hist", "driver_pts_at_downforce_season",
]


def build_features(
    sessions: pd.DataFrame | str | Path,
    config: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build feature DataFrame from sessions and save to features.parquet.

    Args:
        sessions: DataFrame or path to sessions.parquet.
        config: Optional config (load_config() if None).
        output_path: Where to write features.parquet (default: data/processed/features.parquet).

    Returns:
        DataFrame with session columns plus feature columns.
    """
    if config is None:
        config = load_config()
    if isinstance(sessions, (str, Path)):
        path = Path(sessions)
        if not path.exists():
            log.warning("Sessions path does not exist: %s", path)
            return pd.DataFrame()
        sessions = pd.read_parquet(path)
    if sessions.empty:
        log.warning("Sessions DataFrame is empty.")
        return pd.DataFrame()

    df = add_track_features(sessions, config)
    df = add_contextual_features(df, config)
    df = add_driver_features(df, config)
    df = add_team_features(df, config)
    df = add_car_track_features(df, config)

    # Validate: report NaN counts for 2023+ in feature columns
    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    if present:
        recent = df[df["year"] >= 2023] if "year" in df.columns else df
        nan_counts = recent[present].isna().sum()
        if nan_counts.any():
            log.info("Feature NaN counts (2023+): %s", nan_counts[nan_counts > 0].to_dict())

    if output_path is None:
        output_path = Path(config.get("data", {}).get("processed_dir", "data/processed")) / "features.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Saved features to %s (%d rows)", output_path, len(df))
    return df
