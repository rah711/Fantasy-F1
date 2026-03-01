#!/usr/bin/env python3
"""
Compute data-derived circuit characteristics and write to circuit_stats.json.

Replaces hand-estimated values in config.yaml with empirical data:
  - overtake_difficulty: 1 - normalised mean |positions_gained| per race
  - safety_car_probability: fraction of races with >= 1 safety car deployment
  - drs_zones: validated against existing config (no data source to compute from)

Usage:
    PYTHONPATH=. python3 scripts/compute_circuit_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.openf1_loader import fetch_sessions, _rate_limited_get
from src.utils.logging import get_logger

import requests

log = get_logger(__name__)

OUTPUT_PATH = Path("data/processed/circuit_stats.json")
CACHE_DIR = Path("data/cache/openf1_enrichment")
YEARS = [2023, 2024, 2025]


def _compute_overtake_difficulty(races: pd.DataFrame) -> dict[str, float]:
    """Compute overtake difficulty from mean |positions_gained| per circuit.

    Higher value = harder to overtake (Monaco-like).
    Normalised to [0, 1] across all circuits.

    Filters out sessions with corrupted grid data (where >80% of drivers
    have positions_gained == 0, which is physically implausible).
    """
    has_pg = races[races["positions_gained"].notna()].copy()
    has_pg["abs_pg"] = has_pg["positions_gained"].abs()

    # Filter out sessions with corrupted grid data: if >80% of drivers
    # in a session have pg==0, the grid positions are likely wrong.
    session_keys = ["year", "round"]
    session_zero_frac = has_pg.groupby(session_keys)["abs_pg"].apply(
        lambda x: (x == 0).mean()
    )
    bad_sessions = session_zero_frac[session_zero_frac > 0.8].index
    if len(bad_sessions) > 0:
        log.info("Filtering out %d sessions with likely corrupted grid data", len(bad_sessions))
        row_keys = list(zip(has_pg["year"], has_pg["round"]))
        bad_set = set(bad_sessions.to_list())
        keep_mask = [k not in bad_set for k in row_keys]
        has_pg = has_pg[keep_mask]

    if has_pg.empty:
        return {}

    circuit_mean = has_pg.groupby("circuit_id")["abs_pg"].mean()

    if circuit_mean.empty:
        return {}

    raw_min = circuit_mean.min()
    raw_max = circuit_mean.max()
    rng = raw_max - raw_min if raw_max > raw_min else 1.0

    result = {}
    for cid, mean_val in circuit_mean.items():
        result[str(cid)] = round(1.0 - (mean_val - raw_min) / rng, 3)

    return result


def _fetch_safety_car_data(
    races: pd.DataFrame,
    sess: requests.Session,
) -> dict[str, dict]:
    """For each (year, round) race session, check if a safety car was deployed.

    Returns dict keyed by circuit_id with {total_races, sc_races, vsc_races}.
    """
    base_url = "https://api.openf1.org/v1"
    circuit_sc: dict[str, dict] = {}

    for year in YEARS:
        year_sessions = fetch_sessions(year, session=sess)
        if year_sessions.empty:
            log.warning("No OpenF1 sessions for %d", year)
            continue

        year_races = races[(races["year"] == year) & (races["session_type"] == "race")]
        rounds = sorted(year_races["round"].dropna().astype(int).unique())

        for rnd in rounds:
            circuit_rows = year_races[year_races["round"] == rnd]
            if circuit_rows.empty:
                continue
            circuit_id = circuit_rows["circuit_id"].iloc[0]
            if pd.isna(circuit_id):
                continue
            circuit_id = str(circuit_id)

            sk_rows = year_sessions[
                (year_sessions["round"] == rnd) & (year_sessions["session_name"] == "Race")
            ]
            if sk_rows.empty:
                continue
            sk = int(sk_rows.iloc[0]["session_key"])

            cache_path = CACHE_DIR / str(year) / f"race_control_{sk}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                except Exception:
                    data = []
            else:
                try:
                    r = _rate_limited_get(sess, f"{base_url}/race_control?session_key={sk}")
                    if r.status_code == 404:
                        data = []
                    else:
                        r.raise_for_status()
                        data = r.json()
                    cache_path.write_text(json.dumps(data))
                except Exception as e:
                    log.debug("Failed to fetch race_control for sk=%d: %s", sk, e)
                    data = []

            has_sc = any("SAFETY CAR" in str(m.get("message", "")).upper()
                         and "VIRTUAL" not in str(m.get("message", "")).upper()
                         for m in data)
            has_vsc = any("VIRTUAL SAFETY CAR" in str(m.get("message", "")).upper()
                          for m in data)

            if circuit_id not in circuit_sc:
                circuit_sc[circuit_id] = {"total": 0, "sc": 0, "vsc": 0}
            circuit_sc[circuit_id]["total"] += 1
            if has_sc:
                circuit_sc[circuit_id]["sc"] += 1
            if has_vsc:
                circuit_sc[circuit_id]["vsc"] += 1

    result = {}
    for cid, counts in circuit_sc.items():
        if counts["total"] > 0:
            result[cid] = {
                "total_races": counts["total"],
                "sc_races": counts["sc"],
                "vsc_races": counts["vsc"],
                "safety_car_prob": round(counts["sc"] / counts["total"], 3),
                "any_sc_or_vsc_prob": round(
                    (counts["sc"] + counts["vsc"] - min(counts["sc"], counts["vsc"]))
                    / counts["total"],
                    3,
                ),
            }
    return result


def main() -> None:
    df = pd.read_parquet("data/processed/sessions.parquet")
    races = df[df["session_type"] == "race"].copy()
    recent = races[races["year"].isin(YEARS)]

    log.info("Computing circuit stats from %d race rows across %d circuits",
             len(recent), recent["circuit_id"].nunique())

    # 1) Overtake difficulty
    overtake = _compute_overtake_difficulty(recent)
    log.info("Computed overtake_difficulty for %d circuits", len(overtake))

    # 2) Safety car probability
    sess = requests.Session()
    safety = _fetch_safety_car_data(races, sess)
    log.info("Computed safety_car stats for %d circuits", len(safety))

    # 3) Merge into a unified stats dict
    all_circuits = sorted(set(list(overtake.keys()) + list(safety.keys())))
    stats: dict[str, dict] = {}
    for cid in all_circuits:
        entry: dict = {}
        if cid in overtake:
            entry["overtake_difficulty"] = overtake[cid]
        if cid in safety:
            entry["safety_car_probability"] = safety[cid]["safety_car_prob"]
            entry["safety_car_detail"] = safety[cid]
        stats[cid] = entry

    # 4) Print comparison vs current config
    from src.config import load_config
    config = load_config()
    circuits_cfg = config.get("circuits", {})

    print("\n=== Data-Computed vs Config Circuit Stats ===")
    print(f"{'Circuit':<20} {'OD (data)':>10} {'OD (cfg)':>10} {'SC prob (data)':>14} {'SC prob (cfg)':>14} {'DRS (cfg)':>10}")
    print("-" * 82)
    for cid in sorted(stats.keys()):
        od_data = stats[cid].get("overtake_difficulty", "N/A")
        od_cfg = circuits_cfg.get(cid, {}).get("overtake_difficulty", "N/A")
        sc_data = stats[cid].get("safety_car_probability", "N/A")
        sc_cfg = circuits_cfg.get(cid, {}).get("safety_car_probability", "N/A")
        drs_cfg = circuits_cfg.get(cid, {}).get("drs_zones", "N/A")
        print(f"{cid:<20} {str(od_data):>10} {str(od_cfg):>10} {str(sc_data):>14} {str(sc_cfg):>14} {str(drs_cfg):>10}")

    # 5) Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(stats, indent=2))
    log.info("Saved circuit stats to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
