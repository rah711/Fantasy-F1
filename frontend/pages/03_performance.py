from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import cumulative_chart, delta_vs_human_chart, inject_theme, per_round_chart
from frontend.state import get_working_config, require_auth
from src.ui_services import (
    calendar_rounds,
    cumulative_points_by_team,
    current_leaderboard,
    format_round_label,
    is_cancelled,
    load_history,
)


require_auth()
inject_theme()

cfg = get_working_config()
calendar = cfg.get("season", {}).get("calendar", {})
cancelled_rounds = {r for r in calendar_rounds(calendar) if is_cancelled(calendar, r)}


def _drop_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "round" not in df.columns or not cancelled_rounds:
        return df
    return df[~df["round"].astype(int).isin(cancelled_rounds)].copy()


st.title("Performance")
st.caption("How the three teams are doing, race by race.")
if cancelled_rounds:
    st.caption(f"Cancelled rounds excluded from charts: {', '.join(f'R{r}' for r in sorted(cancelled_rounds))}")


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
# Per-round / gap-to-human / cumulative charts
# ---------------------------------------------------------------------------
st.header("Points over time")
cum = _drop_cancelled(cumulative_points_by_team(PROJECT_ROOT))
sprint_rounds = set(cfg.get("season", {}).get("sprint_rounds", []) or [])
if cum.empty:
    st.info("Charts populate after the first round is scored.")
else:
    tab_round, tab_delta, tab_cum = st.tabs(["Per round", "Gap to human", "Cumulative"])
    with tab_round:
        st.altair_chart(per_round_chart(cum, calendar, sprint_rounds=sprint_rounds), use_container_width=True)
        st.caption("Side-by-side bars per race. ★ = sprint weekend (more scoring sessions → higher totals).")
    with tab_delta:
        delta = delta_vs_human_chart(cum, calendar)
        if delta is None:
            st.info("Need at least one scored round for the human team to compute the gap.")
        else:
            st.altair_chart(delta, use_container_width=True)
            st.caption(
                "Each line shows the cumulative gap to the human team. Negative = behind; positive = ahead. "
                "The dashed line is the human baseline. ✗ on the axis = cancelled round."
            )
    with tab_cum:
        st.altair_chart(cumulative_chart(cum, calendar), use_container_width=True)
        st.caption("Cumulative season totals. Always going up — use the other tabs for round-by-round insight.")


# ---------------------------------------------------------------------------
# Model team — per-round points table
# ---------------------------------------------------------------------------
st.header("Model team — per-round breakdown")
hist = _drop_cancelled(load_history(PROJECT_ROOT))
if hist.empty:
    st.info("No rounds locked in yet.")
else:
    show = hist[["round", "drivers", "constructors", "drs_boost", "actual_points", "notes"]].copy()
    show = show.sort_values("round")
    show["Race"] = show["round"].astype(int).apply(lambda r: format_round_label(calendar, r, short=True))
    show["actual_points"] = pd.to_numeric(show["actual_points"], errors="coerce")
    show = show[["Race", "drivers", "constructors", "drs_boost", "actual_points", "notes"]]
    show.columns = ["Race", "Drivers", "Constructors", "DRS Boost", "Points", "Notes"]
    st.dataframe(show, use_container_width=True, hide_index=True)


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
