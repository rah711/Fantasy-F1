from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import (
    cumulative_chart,
    delta_vs_human_chart,
    inject_theme,
    per_round_chart,
    prediction_accuracy_chart,
    prediction_vs_actual_chart,
)
from frontend.state import get_working_config, require_auth
from src.ui_services import (
    calendar_rounds,
    cumulative_points_by_team,
    current_leaderboard,
    format_round_label,
    is_cancelled,
    load_chip_usage,
    load_history,
    prediction_accuracy_over_time,
    prediction_vs_actual,
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
# Chip usage context
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("Chip usage timeline")
chips = _drop_cancelled(load_chip_usage(PROJECT_ROOT))
if chips.empty:
    st.caption("No chips recorded yet.")
else:
    chips = chips.copy().sort_values(["round", "team_key"])
    chips["Race"] = chips["round"].astype(int).apply(lambda r: format_round_label(calendar, r, short=True))
    display = chips[["Race", "team_name", "chip", "details"]].rename(
        columns={"team_name": "Team", "chip": "Chip", "details": "Notes"}
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Context: chips can create large one-off score jumps, so use this timeline when comparing round-to-round results.")


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
# Prediction vs actual
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("Prediction vs actual")
st.caption(
    "How close the model's predictions came to actual fantasy points. "
    "Note: predictions for past rounds (R1-R6) were generated post-hoc with the "
    "current model, so they're slightly biased by data the model has since seen. "
    "From R7 Canada onward, predictions are the ones the wizard made at race-week time."
)

acc = prediction_accuracy_over_time(PROJECT_ROOT)
if acc.empty:
    st.info("No prediction data yet — predictions populate after each race week's wizard run.")
else:
    all_rounds = sorted(int(r) for r in acc["round"].unique())
    selected_rounds = st.multiselect(
        "Rounds to inspect",
        options=all_rounds,
        default=all_rounds,
        format_func=lambda r: format_round_label(calendar, int(r), short=True),
    )
    if not selected_rounds:
        st.caption("Pick at least one round to see charts.")
    else:
        st.subheader("Accuracy over time")
        plot_df = acc[acc["round"].astype(int).isin(selected_rounds)]
        st.altair_chart(prediction_accuracy_chart(plot_df, calendar), use_container_width=True)
        st.caption(
            "Lower MAE (Mean Average Error) = predictions closer to reality. "
            "Watch this trend downward as the model learns from 2026 results.\n\n"
            "Rounds marked X were cancelled."
        )

        st.subheader("Per-round detail")
        for rnd in sorted(selected_rounds):
            with st.expander(f"{format_round_label(calendar, rnd, short=True)} — predicted vs actual"):
                rvs = prediction_vs_actual(PROJECT_ROOT, rnd)
                if rvs.empty:
                    st.caption("No data for this round.")
                    continue
                st.altair_chart(prediction_vs_actual_chart(rvs, rnd), use_container_width=True)
                show = rvs[["driver_code", "constructor_id", "predicted", "actual", "error"]].copy()
                show.columns = ["Driver", "Team", "Predicted", "Actual", "Error"]
                show["Predicted"] = show["Predicted"].round(1)
                show["Actual"] = show["Actual"].round(1)
                show["Error"] = show["Error"].round(1)
                st.dataframe(show.sort_values("Actual", ascending=False), use_container_width=True, hide_index=True)


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
