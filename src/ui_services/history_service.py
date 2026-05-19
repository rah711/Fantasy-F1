"""History log: one row per race round documenting the locked-in team.

This is the source of truth for the visitor "follow the season" view.
Stored at data/fantasy/history.csv. When deployed on Streamlit Cloud,
the wizard appends a row + opens a PR to commit it back to the repo.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd


HISTORY_COLUMNS = [
    "round",
    "locked_at",
    "drivers",
    "constructors",
    "drs_boost",
    "chips_used",
    "chip_details",
    "budget_after",
    "free_transfers_after",
    "banked_transfers_after",
    "actual_points",
    "notes",
]


def history_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "fantasy" / "history.csv"


def load_history(project_root: str | Path) -> pd.DataFrame:
    p = history_path(project_root)
    if not p.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(p)
    # Empty cells parse as NaN; replace with "" for text columns so they
    # don't render as the string "nan" downstream.
    text_cols = ["drivers", "constructors", "drs_boost", "chips_used", "chip_details", "notes"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def append_lockin(
    project_root: str | Path,
    round_number: int,
    drivers: list[str],
    constructors: list[str],
    drs_boost: str | None,
    chips_used: list[str],
    budget_after: float,
    free_transfers_after: int,
    banked_transfers_after: int,
    notes: str = "",
    chip_details: str = "",
) -> Path:
    """Append (or replace) the row for `round_number`. Returns the file path."""
    p = history_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)

    df = load_history(project_root)
    row = {
        "round": int(round_number),
        "locked_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "drivers": ",".join(str(d).upper() for d in drivers),
        "constructors": ",".join(str(c).lower() for c in constructors),
        "drs_boost": str(drs_boost).upper() if drs_boost else "",
        "chips_used": ",".join(chips_used) if chips_used else "",
        "chip_details": str(chip_details or ""),
        "budget_after": round(float(budget_after), 2),
        "free_transfers_after": int(free_transfers_after),
        "banked_transfers_after": int(banked_transfers_after),
        "actual_points": "",  # Filled in later once race results are uploaded
        "notes": notes,
    }

    df = df[df["round"] != int(round_number)] if "round" in df.columns else df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("round").reset_index(drop=True)
    df.to_csv(p, index=False)
    return p


def update_actual_points(
    project_root: str | Path,
    round_number: int,
    actual_points: float,
) -> Path | None:
    p = history_path(project_root)
    if not p.exists():
        return None
    df = load_history(project_root)
    if df.empty or int(round_number) not in df["round"].astype(int).tolist():
        return None
    df.loc[df["round"].astype(int) == int(round_number), "actual_points"] = round(float(actual_points), 2)
    df.to_csv(p, index=False)
    return p
