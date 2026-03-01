"""
Team (constructor) features: car-track compatibility, pitstop performance,
dev trajectory / development_score / regulation_adaptation from config,
and cold-start carryover from prior season.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def add_team_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add constructor-level features from config and from data.

    Adds:
        team_development_score, team_regulation_adaptation, team_development_trajectory,
        team_fastest_pitstop_avg, team_avg_pitstop_avg,
        team_prev_season_avg_pts, team_prev_season_avg_finish,
        team_prev_season_dnf_rate, team_prev_season_races.
    """
    if df.empty or "constructor_id" not in df.columns:
        return df
    out = df.copy()
    teams = config.get("teams", {})

    out["team_development_score"] = out["constructor_id"].map(
        lambda cid: teams.get(cid, {}).get("development_score") if cid else None
    )
    out["team_regulation_adaptation"] = out["constructor_id"].map(
        lambda cid: teams.get(cid, {}).get("regulation_adaptation", "") if cid else ""
    )
    out["team_development_trajectory"] = out["constructor_id"].map(
        lambda cid: teams.get(cid, {}).get("development_trajectory", "") if cid else ""
    )

    cold_cols = [
        "team_prev_season_avg_pts",
        "team_prev_season_avg_finish",
        "team_prev_season_dnf_rate",
        "team_prev_season_races",
    ]
    for c in cold_cols:
        out[c] = float("nan")

    race = out[out["session_type"] == "race"].copy() if "session_type" in out.columns else out.copy()
    out["team_fastest_pitstop_avg"] = float("nan")
    out["team_avg_pitstop_avg"] = float("nan")

    if not race.empty and "fastest_pitstop_ms" in race.columns:
        race = race.sort_values(["constructor_id", "year", "round"])
        fp = pd.to_numeric(race["fastest_pitstop_ms"], errors="coerce")
        if fp.notna().any():
            race["team_fastest_pitstop_avg"] = race.assign(_fp=fp).groupby("constructor_id")["_fp"].transform(
                lambda s: s.shift(1).expanding().mean()
            )
            out.loc[race.index, "team_fastest_pitstop_avg"] = race["team_fastest_pitstop_avg"]
        if "avg_pitstop_ms" in race.columns:
            ap = pd.to_numeric(race["avg_pitstop_ms"], errors="coerce")
            if ap.notna().any():
                race["team_avg_pitstop_avg"] = race.assign(_ap=ap).groupby("constructor_id")["_ap"].transform(
                    lambda s: s.shift(1).expanding().mean()
                )
                out.loc[race.index, "team_avg_pitstop_avg"] = race["team_avg_pitstop_avg"]

    # Cold-start carryover: previous season constructor aggregates
    if not race.empty and "fantasy_points_driver" in race.columns:
        pts = pd.to_numeric(race["fantasy_points_driver"], errors="coerce").fillna(0)
        finish = pd.to_numeric(race.get("finish_position"), errors="coerce").fillna(20)
        dnf_flag = race["status"].astype(str).str.upper().isin(("DNF", "DSQ")).astype(int) if "status" in race.columns else pd.Series(0, index=race.index)

        yearly = (
            race.assign(_pts=pts, _finish=finish, _dnf=dnf_flag)
            .groupby(["constructor_id", "year"], as_index=False)
            .agg(
                team_prev_season_avg_pts=("_pts", "mean"),
                team_prev_season_avg_finish=("_finish", "mean"),
                team_prev_season_dnf_rate=("_dnf", "mean"),
                team_prev_season_races=("_pts", "count"),
            )
            .sort_values(["constructor_id", "year"])
        )
        for c in cold_cols:
            yearly[c] = yearly.groupby("constructor_id")[c].shift(1)
        carry_merged = race[["constructor_id", "year"]].merge(
            yearly[["constructor_id", "year"] + cold_cols],
            on=["constructor_id", "year"],
            how="left",
        )
        for c in cold_cols:
            race[c] = carry_merged[c].values
        for c in cold_cols:
            out.loc[race.index, c] = race[c]

    return out
