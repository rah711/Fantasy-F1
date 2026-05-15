from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import inject_theme
from frontend.state import get_working_config, require_auth
from src.ui_services import (
    cumulative_points_by_team,
    current_leaderboard,
    load_history,
)


require_auth()
inject_theme()

st.title("Performance")
st.caption("How the three teams are doing, race by race.")


# ---------------------------------------------------------------------------
# Leaderboard (table form)
# ---------------------------------------------------------------------------
st.header("Leaderboard")
leaderboard = current_leaderboard(PROJECT_ROOT)
if leaderboard.empty:
    st.info("No rounds scored yet.")
else:
    display = leaderboard.copy()
    display["Team"] = display["team_name"]
    display["Points"] = display["cumulative_points"].round(1)
    display = display[["rank", "Team", "Points"]].rename(columns={"rank": "Rank"})
    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Cumulative + per-round chart
# ---------------------------------------------------------------------------
st.header("Points over time")
cum = cumulative_points_by_team(PROJECT_ROOT)
if cum.empty:
    st.info("Charts populate after the first round is scored.")
else:
    tab_cum, tab_round = st.tabs(["Cumulative", "Per round"])
    with tab_cum:
        pivot_cum = cum.pivot_table(index="round", columns="team_name", values="cumulative_points", aggfunc="last")
        st.line_chart(pivot_cum, height=380)
    with tab_round:
        pivot_round = cum.pivot_table(index="round", columns="team_name", values="round_points", aggfunc="last")
        st.bar_chart(pivot_round, height=380)


# ---------------------------------------------------------------------------
# Model team — per-round points table
# ---------------------------------------------------------------------------
st.header("Model team — per-round breakdown")
hist = load_history(PROJECT_ROOT)
if hist.empty:
    st.info("No rounds locked in yet.")
else:
    show = hist[["round", "drivers", "constructors", "drs_boost", "actual_points", "notes"]].copy()
    show["actual_points"] = pd.to_numeric(show["actual_points"], errors="coerce")
    show.columns = ["Round", "Drivers", "Constructors", "DRS Boost", "Points", "Notes"]
    st.dataframe(show.sort_values("Round"), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Current model team detail
# ---------------------------------------------------------------------------
st.header("Current model team")
cfg = get_working_config()
team = cfg.get("current_team", {})
prices_drivers = cfg.get("prices", {}).get("drivers", {})
prices_ctors = cfg.get("prices", {}).get("constructors", {})

drivers = team.get("drivers", []) or []
constructors = team.get("constructors", []) or []
drs = team.get("drs_boost", "") or ""

if drivers:
    rows = []
    for d in drivers:
        meta = prices_drivers.get(d, {})
        rows.append({
            "Code": d,
            "Driver": meta.get("name", ""),
            "Team": meta.get("team", ""),
            "Price (£M)": meta.get("price", ""),
            "DRS Boost": "★" if d == drs else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if constructors:
    crows = []
    for c in constructors:
        meta = prices_ctors.get(c, {})
        crows.append({
            "Code": c,
            "Constructor": meta.get("name", ""),
            "Price (£M)": meta.get("price", ""),
        })
    st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# How to add competitor scores
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("How the leaderboard gets its data"):
    st.markdown(
        """
        - **Model team points** come from the lock-in history (`data/fantasy/history.csv`),
          updated automatically when the wizard locks in a round and you've uploaded that
          round's race results.
        - **Human + pure-AI team points** come from a manually maintained CSV at
          `data/fantasy/competitors.csv` with columns: `round, team_key, team_name, points`.
          Valid `team_key` values: `human`, `claude_chat` (and optionally `model` if you want
          to override the auto-computed model score).
        """
    )
