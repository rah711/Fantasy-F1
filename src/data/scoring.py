"""
Fantasy points calculator for Fantasy F1 2026.

Reads scoring rules from config.yaml and provides separate functions for
qualifying, sprint, and race scoring (driver and constructor). Handles
DNF, DSQ, pitstop tiers, DOTD, fastest lap. Used by the pipeline and
by backtesting.

Usage:
    from src.config import load_config
    from src.data.scoring import compute_fantasy_points
    cfg = load_config()
    df = compute_fantasy_points(cfg, df_unified)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.schema import STATUS_CLASSIFICATION
from src.utils.logging import get_logger

log = get_logger(__name__)


def _classify_status(status: str) -> str:
    """Map raw status string to Finished / DNF / DSQ."""
    if pd.isna(status):
        return "DNF"
    key = str(status).strip().upper()
    return STATUS_CLASSIFICATION.get(key, "DNF")


# ---------------------------------------------------------------------------
# Driver scoring
# ---------------------------------------------------------------------------


def score_driver_qualifying(
    cfg: dict[str, Any],
    finish_position: int | float,
    status: str,
    q2_time_ms: float | None,
    q3_time_ms: float | None,
) -> int:
    """Driver qualifying points. P1–P10 from config; DNF/DSQ/no time = penalty."""
    sc = cfg["scoring"]["qualifying"]
    pos_map = sc["positions"]
    penalty = sc["dnf_dsq_penalty"]

    classified = _classify_status(status)
    if classified == "DSQ" or classified == "DNF":
        return penalty
    # No time set (did not set a lap in Q1) -> penalty
    if q2_time_ms is None or (isinstance(q2_time_ms, float) and pd.isna(q2_time_ms)):
        return penalty

    pos = int(finish_position) if finish_position and finish_position >= 1 else None
    if pos is None or pos not in pos_map:
        return 0
    return pos_map[pos]


def score_driver_sprint(
    cfg: dict[str, Any],
    finish_position: int | float,
    grid_position: int | float,
    status: str,
    positions_gained: int | float,
    overtakes: int | float,
    is_fastest_lap: bool,
) -> int:
    """Driver sprint points: position, positions gained/lost, overtakes, fastest lap."""
    sc = cfg["scoring"]["sprint"]
    pos_map = sc["positions"]
    penalty = sc["dnf_dsq_penalty"]
    pt_gain = sc["position_gained"]
    pt_lost = sc["position_lost"]
    overtake_pt = sc["overtake_bonus"]
    fl_pt = sc["fastest_lap"]

    classified = _classify_status(status)
    if classified in ("DSQ", "DNF"):
        return penalty

    points = 0
    pos = int(finish_position) if finish_position and finish_position >= 1 else None
    if pos is not None and pos in pos_map:
        points += pos_map[pos]

    # Positions gained/lost from sprint grid
    if positions_gained is not None and not pd.isna(positions_gained):
        gain = int(positions_gained)
        if gain > 0:
            points += gain * pt_gain
        elif gain < 0:
            points += (-gain) * pt_lost

    if overtakes is not None and not pd.isna(overtakes):
        points += int(overtakes) * overtake_pt
    if is_fastest_lap:
        points += fl_pt

    return points


def score_driver_race(
    cfg: dict[str, Any],
    finish_position: int | float,
    status: str,
    positions_gained: int | float,
    overtakes: int | float,
    is_fastest_lap: bool,
    is_dotd: bool,
) -> int:
    """Driver race points: position, positions gained/lost, overtakes, fastest lap, DOTD."""
    sc = cfg["scoring"]["race"]
    pos_map = sc["positions"]
    penalty = sc["dnf_dsq_penalty"]
    pt_gain = sc["position_gained"]
    pt_lost = sc["position_lost"]
    overtake_pt = sc["overtake_bonus"]
    fl_pt = sc["fastest_lap"]
    dotd_pt = sc["dotd"]

    classified = _classify_status(status)
    if classified in ("DSQ", "DNF"):
        return penalty

    points = 0
    pos = int(finish_position) if finish_position and finish_position >= 1 else None
    if pos is not None and pos in pos_map:
        points += pos_map[pos]

    if positions_gained is not None and not pd.isna(positions_gained):
        gain = int(positions_gained)
        if gain > 0:
            points += gain * pt_gain
        elif gain < 0:
            points += (-gain) * pt_lost

    if overtakes is not None and not pd.isna(overtakes):
        points += int(overtakes) * overtake_pt
    if is_fastest_lap:
        points += fl_pt
    if is_dotd:
        points += dotd_pt

    return points


# ---------------------------------------------------------------------------
# Constructor scoring
# ---------------------------------------------------------------------------

# Qualifying: neither_q2, one_q2, both_q2, one_q3, both_q3, dsq_per_driver (stack: both_q2 + both_q3 = 13)


def score_constructor_qualifying(
    cfg: dict[str, Any],
    driver_rows: pd.DataFrame,
) -> int:
    """Constructor qualifying points from both drivers: Q2/Q3 bonuses + DSQ penalties."""
    sc = cfg["scoring"]["qualifying"]["constructor"]
    neither_q2 = sc["neither_q2"]
    one_q2 = sc["one_q2"]
    both_q2 = sc["both_q2"]
    one_q3 = sc["one_q3"]
    both_q3 = sc["both_q3"]
    dsq_penalty = sc["dsq_per_driver"]

    points = 0
    q2_count = 0
    q3_count = 0
    dsq_count = 0

    for _, row in driver_rows.iterrows():
        status = _classify_status(row.get("status", ""))
        if status == "DSQ":
            dsq_count += 1
        has_q2 = row.get("q2_time_ms") is not None and not pd.isna(row.get("q2_time_ms"))
        has_q3 = row.get("q3_time_ms") is not None and not pd.isna(row.get("q3_time_ms"))
        if has_q2:
            q2_count += 1
        if has_q3:
            q3_count += 1

    points += dsq_count * dsq_penalty

    if q2_count == 0:
        points += neither_q2
    elif q2_count == 1:
        points += one_q2
    else:
        points += both_q2

    if q3_count == 1:
        points += one_q3
    elif q3_count == 2:
        points += both_q3

    return points


def score_constructor_sprint(
    cfg: dict[str, Any],
    driver_rows: pd.DataFrame,
) -> int:
    """Constructor sprint = sum of both drivers' sprint scores; DSQ penalty per driver."""
    sc = cfg["scoring"]["sprint"]["constructor"]
    dsq_penalty = sc["dsq_per_driver"]
    driver_sc = cfg["scoring"]["sprint"]
    pos_map = driver_sc["positions"]
    pt_gain = driver_sc["position_gained"]
    pt_lost = driver_sc["position_lost"]
    overtake_pt = driver_sc["overtake_bonus"]
    fl_pt = driver_sc["fastest_lap"]
    dnf_penalty = driver_sc["dnf_dsq_penalty"]

    total = 0
    for _, row in driver_rows.iterrows():
        status = _classify_status(row.get("status", ""))
        if status in ("DSQ", "DNF"):
            total += dnf_penalty
            continue
        pos = row.get("finish_position")
        if pos is not None and not pd.isna(pos) and int(pos) in pos_map:
            total += pos_map[int(pos)]
        pg = row.get("positions_gained")
        if pg is not None and not pd.isna(pg):
            g = int(pg)
            if g > 0:
                total += g * pt_gain
            elif g < 0:
                total += (-g) * pt_lost
        ov = row.get("overtakes")
        if ov is not None and not pd.isna(ov):
            total += int(ov) * overtake_pt
        if row.get("is_fastest_lap"):
            total += fl_pt
    return total


def _pitstop_tier_points(cfg: dict[str, Any], fastest_pitstop_ms: float | None) -> int:
    """Map fastest pitstop (ms) to tier points from config. None/NaN -> 0."""
    if fastest_pitstop_ms is None or pd.isna(fastest_pitstop_ms):
        return 0
    ms = float(fastest_pitstop_ms)
    tiers = cfg["scoring"]["race"]["constructor"]["pitstop_tiers"]
    if ms < 2000:
        return tiers["under_2_0s"]
    if ms < 2200:
        return tiers["2_00_to_2_19s"]
    if ms < 2500:
        return tiers["2_20_to_2_49s"]
    if ms < 3000:
        return tiers["2_50_to_2_99s"]
    return tiers["over_3_0s"]


def score_constructor_race(
    cfg: dict[str, Any],
    driver_rows: pd.DataFrame,
    constructor_fastest_pitstop_ms: float | None,
    is_fastest_pitstop_constructor: bool,
    fastest_pitstop_in_race_ms: float | None,
) -> int:
    """Constructor race = sum of both drivers' race scores + pitstop tier + bonuses."""
    sc = cfg["scoring"]["race"]
    dnf_penalty = sc["dnf_dsq_penalty"]
    pos_map = sc["positions"]
    pt_gain = sc["position_gained"]
    pt_lost = sc["position_lost"]
    overtake_pt = sc["overtake_bonus"]
    fl_pt = sc["fastest_lap"]
    dotd_pt = sc["dotd"]
    con = sc["constructor"]
    tier_bonus = con["fastest_pitstop_bonus"]
    world_record_bonus = con["world_record_bonus"]

    total = 0
    for _, row in driver_rows.iterrows():
        status = _classify_status(row.get("status", ""))
        if status in ("DSQ", "DNF"):
            total += dnf_penalty
            continue
        pos = row.get("finish_position")
        if pos is not None and not pd.isna(pos) and int(pos) in pos_map:
            total += pos_map[int(pos)]
        pg = row.get("positions_gained")
        if pg is not None and not pd.isna(pg):
            g = int(pg)
            if g > 0:
                total += g * pt_gain
            elif g < 0:
                total += (-g) * pt_lost
        ov = row.get("overtakes")
        if ov is not None and not pd.isna(ov):
            total += int(ov) * overtake_pt
        if row.get("is_fastest_lap"):
            total += fl_pt
        if row.get("is_dotd"):
            total += dotd_pt

    # Pitstop tier from constructor's fastest stop
    total += _pitstop_tier_points(cfg, constructor_fastest_pitstop_ms)
    if is_fastest_pitstop_constructor:
        total += tier_bonus
    if constructor_fastest_pitstop_ms is not None and not pd.isna(constructor_fastest_pitstop_ms) and constructor_fastest_pitstop_ms < 1800:
        total += world_record_bonus

    return total


# ---------------------------------------------------------------------------
# Apply to DataFrame (unified columns)
# ---------------------------------------------------------------------------

def compute_fantasy_points(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Add fantasy_points_driver and constructor_bonus to each row.

    Expects unified columns: session_type, finish_position, grid_position,
    status, positions_gained, overtakes, is_fastest_lap, is_dotd,
    q1_time_ms, q2_time_ms, q3_time_ms, fastest_pitstop_ms, driver_code,
    constructor_id, year, round.

    Adds:
        fantasy_points_driver: points for this driver in this session.
        constructor_bonus: constructor bonus for this session (same for both
            drivers in the team); total constructor score = sum of both
            drivers' fantasy_points_driver + constructor_bonus (once).
    """
    if df.empty:
        df = df.copy()
        df["fantasy_points_driver"] = pd.Series(dtype=int)
        df["constructor_bonus"] = pd.Series(dtype=int)
        return df

    out = df.copy()
    driver_pts = []

    for idx, row in out.iterrows():
        st = row.get("session_type", "")
        if st == "qualifying":
            pt = score_driver_qualifying(
                cfg,
                row.get("finish_position"),
                row.get("status", ""),
                row.get("q2_time_ms"),
                row.get("q3_time_ms"),
            )
        elif st == "sprint":
            pt = score_driver_sprint(
                cfg,
                row.get("finish_position"),
                row.get("grid_position"),
                row.get("status", ""),
                row.get("positions_gained"),
                row.get("overtakes"),
                row.get("is_fastest_lap", False),
            )
        elif st == "race":
            pt = score_driver_race(
                cfg,
                row.get("finish_position"),
                row.get("status", ""),
                row.get("positions_gained"),
                row.get("overtakes"),
                row.get("is_fastest_lap", False),
                row.get("is_dotd", False),
            )
        else:
            pt = 0
        driver_pts.append(pt)

    out["fantasy_points_driver"] = driver_pts

    # Constructor bonuses per (year, round, session_type, constructor_id)
    out["constructor_bonus"] = 0
    keys = ["year", "round", "session_type", "constructor_id"]
    if not all(k in out.columns for k in keys):
        return out

    for grp_id, grp in out.groupby(keys, dropna=False):
        year, round_num, session_type, constructor_id = grp_id
        if session_type == "qualifying":
            bonus = score_constructor_qualifying(cfg, grp)
            # Constructor qualifying is the full bonus (we don't double-count driver pts)
            # So constructor_bonus = bonus - (sum of driver qualifying pts for this team)?
            # No: in the game, constructor gets neither_q2/one_q2/both_q2/one_q3/both_q3 + dsq.
            # Driver gets position points. So constructor_bonus is just the constructor part.
            out.loc[grp.index, "constructor_bonus"] = bonus
        elif session_type == "sprint":
            # Constructor sprint = sum of drivers; no separate "bonus", so we store 0 and
            # total constructor sprint = sum(driver pts) for the two drivers.
            out.loc[grp.index, "constructor_bonus"] = 0
        elif session_type == "race":
            fastest_ms = grp["fastest_pitstop_ms"].min() if "fastest_pitstop_ms" in grp.columns else None
            if pd.isna(fastest_ms):
                fastest_ms = None
            # Is this constructor the one with the single fastest pitstop in the race?
            race_df = out[(out["year"] == year) & (out["round"] == round_num) & (out["session_type"] == "race")]
            all_fastest = race_df.groupby("constructor_id")["fastest_pitstop_ms"].min()
            min_overall = all_fastest.min() if len(all_fastest) else None
            is_fastest = (min_overall is not None and not pd.isna(min_overall) and
                          fastest_ms is not None and fastest_ms == min_overall)
            bonus = score_constructor_race(
                cfg,
                grp,
                constructor_fastest_pitstop_ms=fastest_ms,
                is_fastest_pitstop_constructor=is_fastest,
                fastest_pitstop_in_race_ms=min_overall,
            )
            # Constructor race score = sum of both drivers' driver points + pitstop tier + bonuses.
            # We store in constructor_bonus only the pitstop tier + fastest bonus + world record,
            # not the driver sum (driver sum is already in fantasy_points_driver).
            driver_sum = grp["fantasy_points_driver"].sum()
            pitstop_part = bonus - driver_sum
            out.loc[grp.index, "constructor_bonus"] = pitstop_part

    return out
