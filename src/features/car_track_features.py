"""
Car-track interaction features.

Captures how well each constructor and driver performs on different circuit types
and downforce levels. Splits into:
  - Historical (prior seasons): stable prior but reflects a different car
  - Current season: noisy early on but reflects the actual 2026 car
The model learns to weight them; current-season signal strengthens each round.

All features use shift(1) to prevent data leakage.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def add_car_track_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add constructor-x-circuit and driver-x-circuit interaction features.

    Requires circuit_type and circuit_downforce to already be present (from track_features).
    """
    new_cols = [
        "team_pts_at_circuit_type_hist",
        "team_pts_at_circuit_type_season",
        "team_pts_at_downforce_hist",
        "team_pts_at_downforce_season",
        "team_circuit_type_delta",
        "team_season_rounds_so_far",
        "driver_pts_at_circuit_type_hist",
        "driver_pts_at_circuit_type_season",
        "driver_pts_at_downforce_hist",
        "driver_pts_at_downforce_season",
    ]
    if df.empty or "constructor_id" not in df.columns:
        return df

    out = df.copy()
    for c in new_cols:
        out[c] = float("nan")

    has_session = "session_type" in out.columns
    has_ctype = "circuit_type" in out.columns
    has_df_level = "circuit_downforce" in out.columns

    if not has_session or not has_ctype:
        return out

    race = out[out["session_type"] == "race"].copy()
    if race.empty:
        return out

    race = race.sort_values(["year", "round", "driver_code"])
    pts = pd.to_numeric(race["fantasy_points_driver"], errors="coerce").fillna(0)
    race["_pts"] = pts

    # --- Constructor x circuit_type ---
    race["team_pts_at_circuit_type_hist"] = _cross_avg_historical(
        race, group_cols=["constructor_id", "circuit_type"], val_col="_pts"
    )
    race["team_pts_at_circuit_type_season"] = _cross_avg_current_season(
        race, group_cols=["constructor_id", "circuit_type"], val_col="_pts"
    )
    team_overall = _cross_avg_all(race, group_cols=["constructor_id"], val_col="_pts")
    team_ct = _cross_avg_all(race, group_cols=["constructor_id", "circuit_type"], val_col="_pts")
    race["team_circuit_type_delta"] = team_ct - team_overall

    race["team_season_rounds_so_far"] = (
        race.groupby(["constructor_id", "year"]).cumcount()
    )

    # --- Constructor x downforce ---
    if has_df_level:
        race["team_pts_at_downforce_hist"] = _cross_avg_historical(
            race, group_cols=["constructor_id", "circuit_downforce"], val_col="_pts"
        )
        race["team_pts_at_downforce_season"] = _cross_avg_current_season(
            race, group_cols=["constructor_id", "circuit_downforce"], val_col="_pts"
        )

    # --- Driver x circuit_type ---
    race["driver_pts_at_circuit_type_hist"] = _cross_avg_historical(
        race, group_cols=["driver_code", "circuit_type"], val_col="_pts"
    )
    race["driver_pts_at_circuit_type_season"] = _cross_avg_current_season(
        race, group_cols=["driver_code", "circuit_type"], val_col="_pts"
    )

    # --- Driver x downforce ---
    if has_df_level:
        race["driver_pts_at_downforce_hist"] = _cross_avg_historical(
            race, group_cols=["driver_code", "circuit_downforce"], val_col="_pts"
        )
        race["driver_pts_at_downforce_season"] = _cross_avg_current_season(
            race, group_cols=["driver_code", "circuit_downforce"], val_col="_pts"
        )

    # Write back using the preserved original index
    for c in new_cols:
        if c in race.columns:
            out.loc[race.index, c] = race[c]

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cross_avg_all(race: pd.DataFrame, group_cols: list[str], val_col: str) -> pd.Series:
    """Expanding mean of val_col within group, shifted by 1 to prevent leakage."""
    return race.groupby(group_cols)[val_col].transform(
        lambda s: s.shift(1).expanding().mean()
    )


def _cross_avg_historical(race: pd.DataFrame, group_cols: list[str], val_col: str) -> pd.Series:
    """Average of val_col from prior seasons only (year < this row's year).

    Aggregates per (group + year), then shifts by 1 year and takes expanding mean.
    So for year 2023, uses the averages from 2020, 2021, 2022.
    """
    yearly_agg = race.groupby(group_cols + ["year"])[val_col].mean().reset_index()
    yearly_agg = yearly_agg.rename(columns={val_col: "_hist_avg"})
    yearly_agg["_hist_avg"] = yearly_agg.groupby(group_cols)["_hist_avg"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    merged = race[group_cols + ["year"]].merge(
        yearly_agg[group_cols + ["year", "_hist_avg"]],
        on=group_cols + ["year"],
        how="left",
    )
    return merged["_hist_avg"].values


def _cross_avg_current_season(race: pd.DataFrame, group_cols: list[str], val_col: str) -> pd.Series:
    """Expanding mean of val_col within (group + year), shifted by 1.

    Only uses data from the same season so it reflects this year's car.
    """
    return race.groupby(group_cols + ["year"])[val_col].transform(
        lambda s: s.shift(1).expanding().mean()
    )
