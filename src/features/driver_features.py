"""
Driver-level features: rolling fantasy pts (3/5 race), historical avg finish per circuit,
overtake rate, DNF rate. All with shift(1) to prevent leakage.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def add_driver_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add driver features using expanding windows and shift(1).

    Expects df sorted by (driver_code, year, round) with fantasy_points_driver, finish_position, status.
    Adds: driver_rolling_pts_3, driver_rolling_pts_5, driver_avg_finish_at_circuit,
          driver_overtake_rate, driver_dnf_rate, and cold-start carryover features.
    All rolling/expanding signals are shifted by 1 so no leakage.
    """
    if df.empty or "driver_code" not in df.columns:
        return df
    out = df.copy()
    cold_cols = [
        "driver_prev_season_avg_pts",
        "driver_prev_season_avg_finish",
        "driver_prev_season_dnf_rate",
        "driver_prev_season_overtake_rate",
        "driver_prev_season_races",
        "driver_is_cold_start",
        "driver_early_round_flag",
        "driver_cold_start_pressure",
    ]
    if "session_type" not in out.columns:
        out["driver_rolling_pts_3"] = float("nan")
        out["driver_rolling_pts_5"] = float("nan")
        out["driver_avg_finish_at_circuit"] = float("nan")
        out["driver_overtake_rate"] = float("nan")
        out["driver_dnf_rate"] = float("nan")
        out["driver_skill_residual"] = float("nan")
        for c in cold_cols:
            out[c] = float("nan")
        return out

    race = out[out["session_type"] == "race"].copy()
    if race.empty:
        out["driver_rolling_pts_3"] = float("nan")
        out["driver_rolling_pts_5"] = float("nan")
        out["driver_avg_finish_at_circuit"] = float("nan")
        out["driver_overtake_rate"] = float("nan")
        out["driver_dnf_rate"] = float("nan")
        out["driver_skill_residual"] = float("nan")
        for c in cold_cols:
            out[c] = float("nan")
        return out

    race = race.sort_values(["driver_code", "year", "round"])
    pts = pd.to_numeric(race["fantasy_points_driver"], errors="coerce").fillna(0)
    finish = pd.to_numeric(race.get("finish_position"), errors="coerce").fillna(20)
    ov = pd.to_numeric(race.get("overtakes"), errors="coerce").fillna(0)
    dnf = race["status"].astype(str).str.upper().isin(("DNF", "DSQ")).astype(int)

    race["driver_rolling_pts_3"] = (
        race.assign(_pts=pts).groupby("driver_code")["_pts"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    )
    race["driver_rolling_pts_5"] = (
        race.assign(_pts=pts).groupby("driver_code")["_pts"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    )
    race["driver_avg_finish_at_circuit"] = (
        race.assign(_finish=finish)
        .groupby(["driver_code", "circuit_id"])["_finish"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    race["driver_overtake_rate"] = (
        race.assign(_ov=ov).groupby("driver_code")["_ov"].transform(lambda s: s.shift(1).expanding().mean())
    )
    race["driver_dnf_rate"] = (
        race.assign(_dnf=dnf).groupby("driver_code")["_dnf"].transform(lambda s: s.shift(1).expanding().mean())
    )

    # Driver-vs-teammate skill residual: avg (teammate_finish - driver_finish)
    # across the driver's historical races. Persistent across team changes —
    # captures intrinsic driver skill independent of the car they're in. With
    # shift(1) so the row predicting race N uses skill computed through N-1.
    race = race.assign(_finish_aux=finish.values)
    grp = race.groupby(["constructor_id", "year", "round"])["_finish_aux"]
    group_sum = grp.transform("sum")
    group_count = grp.transform("count")
    denom = (group_count - 1).replace(0, float("nan"))
    teammate_finish = (group_sum - race["_finish_aux"]) / denom
    skill_delta = teammate_finish - race["_finish_aux"]
    race["driver_skill_residual"] = (
        race.assign(_sd=skill_delta).groupby("driver_code")["_sd"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    race = race.drop(columns=["_finish_aux"])

    # Cold-start carryover: use previous season aggregates as priors.
    yearly = (
        race.assign(_pts=pts, _finish=finish, _ov=ov, _dnf=dnf)
        .groupby(["driver_code", "year"], as_index=False)
        .agg(
            driver_prev_season_avg_pts=("_pts", "mean"),
            driver_prev_season_avg_finish=("_finish", "mean"),
            driver_prev_season_dnf_rate=("_dnf", "mean"),
            driver_prev_season_overtake_rate=("_ov", "mean"),
            driver_prev_season_races=("_pts", "count"),
        )
        .sort_values(["driver_code", "year"])
    )
    carry_cols = [
        "driver_prev_season_avg_pts",
        "driver_prev_season_avg_finish",
        "driver_prev_season_dnf_rate",
        "driver_prev_season_overtake_rate",
        "driver_prev_season_races",
    ]
    for c in carry_cols:
        yearly[c] = yearly.groupby("driver_code")[c].shift(1)

    # Merge without losing the original index: use .values assignment
    carry_merged = race[["driver_code", "year"]].merge(
        yearly[["driver_code", "year"] + carry_cols],
        on=["driver_code", "year"],
        how="left",
    )
    for c in carry_cols:
        race[c] = carry_merged[c].values
    race["driver_is_cold_start"] = race["driver_prev_season_races"].isna().astype(int)
    race["driver_early_round_flag"] = (pd.to_numeric(race["round"], errors="coerce").fillna(99) <= 3).astype(int)
    race["driver_cold_start_pressure"] = race["driver_is_cold_start"] * race["driver_early_round_flag"]

    out["driver_rolling_pts_3"] = float("nan")
    out["driver_rolling_pts_5"] = float("nan")
    out["driver_avg_finish_at_circuit"] = float("nan")
    out["driver_overtake_rate"] = float("nan")
    out["driver_dnf_rate"] = float("nan")
    out["driver_skill_residual"] = float("nan")
    for c in cold_cols:
        out[c] = float("nan")
    out.loc[race.index, "driver_rolling_pts_3"] = race["driver_rolling_pts_3"]
    out.loc[race.index, "driver_rolling_pts_5"] = race["driver_rolling_pts_5"]
    out.loc[race.index, "driver_avg_finish_at_circuit"] = race["driver_avg_finish_at_circuit"]
    out.loc[race.index, "driver_overtake_rate"] = race["driver_overtake_rate"]
    out.loc[race.index, "driver_dnf_rate"] = race["driver_dnf_rate"]
    out.loc[race.index, "driver_skill_residual"] = race["driver_skill_residual"]
    for c in cold_cols:
        out.loc[race.index, c] = race[c]
    return out
