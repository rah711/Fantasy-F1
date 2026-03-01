"""
Circuit features: data-computed values preferred, config.yaml as fallback.

Reads data/processed/circuit_stats.json (built by scripts/compute_circuit_stats.py)
for overtake_difficulty and safety_car_probability. Falls back to config.yaml values
when the stats file is missing or a circuit has no data-computed value.

Adds: circuit_type, circuit_overtake_difficulty, circuit_drs_zones,
      circuit_downforce, circuit_safety_car_prob, is_sprint_round.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

CIRCUIT_STATS_PATH = Path("data/processed/circuit_stats.json")


def _load_circuit_stats() -> dict[str, dict]:
    """Load data-computed circuit stats if available."""
    if not CIRCUIT_STATS_PATH.exists():
        log.info("No circuit_stats.json found; using config.yaml values only")
        return {}
    try:
        return json.loads(CIRCUIT_STATS_PATH.read_text())
    except Exception as e:
        log.warning("Failed to load circuit_stats.json: %s", e)
        return {}


def add_track_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add circuit-level and round-level track features.

    Prefers data-computed values from circuit_stats.json for
    overtake_difficulty and safety_car_probability, falling back
    to config.yaml when data is unavailable.
    """
    if df.empty:
        return df
    out = df.copy()
    circuits_cfg = config.get("circuits", {})
    sprint_rounds = config.get("season", {}).get("sprint_rounds", [])
    stats = _load_circuit_stats()

    type_list = []
    od_list = []
    drs_list = []
    downforce_list = []
    sc_list = []
    sprint_list = []

    for _, row in out.iterrows():
        cid = row.get("circuit_id")
        round_num = row.get("round")
        circ_cfg = circuits_cfg.get(cid, {}) if cid else {}
        circ_stats = stats.get(cid, {}) if cid else {}

        type_list.append(circ_cfg.get("type", ""))

        od_data = circ_stats.get("overtake_difficulty")
        od_cfg = circ_cfg.get("overtake_difficulty")
        od_list.append(od_data if od_data is not None else od_cfg)

        drs_list.append(circ_cfg.get("drs_zones"))
        downforce_list.append(circ_cfg.get("downforce", ""))

        sc_data = circ_stats.get("safety_car_probability")
        sc_cfg = circ_cfg.get("safety_car_probability")
        sc_list.append(sc_data if sc_data is not None else sc_cfg)

        sprint_list.append(round_num in sprint_rounds if round_num is not None else False)

    out["circuit_type"] = type_list
    out["circuit_overtake_difficulty"] = od_list
    out["circuit_drs_zones"] = drs_list
    out["circuit_downforce"] = downforce_list
    out["circuit_safety_car_prob"] = sc_list
    out["is_sprint_round"] = sprint_list
    return out
