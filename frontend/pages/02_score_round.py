from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import inject_theme
from frontend.state import (
    get_working_config,
    github_settings_from_secrets,
    is_owner,
    require_auth,
)
from src.ui_services import (
    THREE_TEAM_LABELS,
    append_breakdown,
    append_competitor_score,
    calendar_rounds,
    format_round_label,
    history_path,
    is_cancelled,
    load_competitor_history,
    load_history,
    parse_score_breakdown,
    previous_active_round,
    propose_files_pr,
    update_actual_points,
)
from src.ui_services.season_service import breakdowns_path
from src.ui_services.season_service import competitors_path


require_auth()

if not is_owner():
    inject_theme()
    st.title("Score Round")
    st.warning("This page is for the team principal only. Switch to **Performance** in the sidebar to follow the season.")
    st.stop()

inject_theme()

st.title("Score round")
st.caption("After each race: enter what each of the three teams actually scored. Powers the leaderboard + charts.")


# ---------------------------------------------------------------------------
# Round selector
# ---------------------------------------------------------------------------
cfg = get_working_config()
season = cfg.get("season", {})
calendar = season.get("calendar", {})
weather = cfg.get("weather_override", {})
default_round = previous_active_round(calendar, int(weather.get("next_race_round", 1)))

st.header("1. Pick the round")
all_rounds = calendar_rounds(calendar, include_cancelled=True) or list(range(1, int(season.get("total_rounds", 24)) + 1))
try:
    default_index = all_rounds.index(default_round)
except ValueError:
    default_index = 0
round_number = st.selectbox(
    "Round to score",
    options=all_rounds,
    index=default_index,
    format_func=lambda r: format_round_label(calendar, r),
    help="Pick the round you're entering scores for. Cancelled rounds stay listed but flagged.",
)


# ---------------------------------------------------------------------------
# Existing entries for this round
# ---------------------------------------------------------------------------
hist = load_history(PROJECT_ROOT)
hist_row = hist[hist["round"].astype(int) == int(round_number)] if not hist.empty else pd.DataFrame()
existing_model_pts = None
if not hist_row.empty:
    val = hist_row.iloc[0].get("actual_points", "")
    try:
        existing_model_pts = float(val) if str(val).strip() not in {"", "nan"} else None
    except (TypeError, ValueError):
        existing_model_pts = None

comp_df = load_competitor_history(PROJECT_ROOT)
comp_round = comp_df[comp_df["round"].astype(int) == int(round_number)] if not comp_df.empty else pd.DataFrame()
existing_human = comp_round[comp_round["team_key"] == "human"]["points"].iloc[0] if not comp_round.empty and (comp_round["team_key"] == "human").any() else None
existing_claude = comp_round[comp_round["team_key"] == "claude_chat"]["points"].iloc[0] if not comp_round.empty and (comp_round["team_key"] == "claude_chat").any() else None


# ---------------------------------------------------------------------------
# Score entry form
# ---------------------------------------------------------------------------
st.header("2. Enter scores")
st.caption(
    "Enter the **official total** the F1 Fantasy site shows for each team. "
    "Optionally, paste the per-driver breakdown to capture a richer record for the visitor view."
)


def _team_score_block(
    team_key: str,
    label: str,
    placeholder_caption: str,
    existing_total: float | None,
    breakdown_placeholder: str,
) -> tuple[float, list, list[str]]:
    """Render one team's score input + optional breakdown paste. Returns (total, rows, warnings)."""
    st.markdown(f"**{label}**")
    st.caption(placeholder_caption)
    breakdown_text = st.text_area(
        "Paste per-driver breakdown (optional)",
        key=f"breakdown_{team_key}",
        height=170,
        placeholder=breakdown_placeholder,
    )
    rows, parsed_total, parse_warnings = parse_score_breakdown(breakdown_text, cfg)
    if rows:
        st.caption(f"Parsed sum from breakdown: **{parsed_total:.1f}** ({len(rows)} entries)")
    default_total = parsed_total if rows else (
        existing_total if existing_total is not None else 0.0
    )
    total = st.number_input(
        f"{label} — total points",
        min_value=-200.0, max_value=2000.0, value=float(default_total),
        step=0.5, key=f"score_{team_key}",
        help="Defaults to the parsed sum if you pasted a breakdown; otherwise enter manually.",
    )
    return total, rows, parse_warnings


_HUMAN_PLACEHOLDER = (
    "Russell: 54\n"
    "Gasly: 14\n"
    "Lawson: 10\n"
    "Sainz: 4\n"
    "Bearman: -14\n"
    "Ferrari: 75\n"
    "Racing Bulls: 18"
)
_CLAUDE_PLACEHOLDER = _HUMAN_PLACEHOLDER  # same format
_MODEL_PLACEHOLDER = _HUMAN_PLACEHOLDER

c1, c2, c3 = st.columns(3)
with c1:
    human_pts, human_breakdown, human_warns = _team_score_block(
        "human", THREE_TEAM_LABELS["human"],
        "Your scores from the official Fantasy F1 site.",
        existing_human, _HUMAN_PLACEHOLDER,
    )
with c2:
    claude_pts, claude_breakdown, claude_warns = _team_score_block(
        "claude_chat", THREE_TEAM_LABELS["claude_chat"],
        "Pure-AI Claude chat team's scores.",
        existing_claude, _CLAUDE_PLACEHOLDER,
    )
with c3:
    model_pts, model_breakdown, model_warns = _team_score_block(
        "model", THREE_TEAM_LABELS["model"],
        "From the official site — what the model's lineup actually scored.",
        existing_model_pts, _MODEL_PLACEHOLDER,
    )

for w in human_warns + claude_warns + model_warns:
    st.warning(w)

st.write("")

if st.button("Save scores for this round", type="primary", key="save_scores"):
    paths_changed: list[str] = []

    # 1. Update model team's actual_points in history.csv if a row exists
    if not hist_row.empty:
        p = update_actual_points(PROJECT_ROOT, int(round_number), float(model_pts))
        if p:
            paths_changed.append(str(p))

    # 2. Append/replace competitor rows
    p_human = append_competitor_score(PROJECT_ROOT, int(round_number), "human", float(human_pts))
    p_claude = append_competitor_score(PROJECT_ROOT, int(round_number), "claude_chat", float(claude_pts))
    if hist_row.empty:
        # No lock-in for this round — also store the model team's score in competitors.csv
        # so it shows up in the leaderboard / charts despite the missing history row.
        append_competitor_score(PROJECT_ROOT, int(round_number), "model", float(model_pts))

    # 3. Save breakdowns (if any team's breakdown was pasted)
    for team_key, rows in (
        ("human", human_breakdown),
        ("claude_chat", claude_breakdown),
        ("model", model_breakdown),
    ):
        if rows:
            bp = append_breakdown(
                PROJECT_ROOT, int(round_number), team_key,
                [{"asset": r.asset, "name": r.name, "kind": r.kind, "points": r.points} for r in rows],
            )
            paths_changed.append(str(bp))
    paths_changed += [str(p_human), str(p_claude)]

    st.success(f"Saved scores for R{int(round_number)}: human {human_pts:.1f} · claude {claude_pts:.1f} · model {model_pts:.1f}")
    for pth in paths_changed:
        st.caption(f"Updated {pth}")

    # 3. Auto-PR if creds configured (commits both history.csv + competitors.csv)
    gh = github_settings_from_secrets()
    if gh.get("token") and gh["token"] != "ghp_xxx":
        files_to_pr: dict[str, str] = {}
        h_path = history_path(PROJECT_ROOT)
        if h_path.exists():
            files_to_pr["data/fantasy/history.csv"] = h_path.read_text()
        c_path = competitors_path(PROJECT_ROOT)
        if c_path.exists():
            files_to_pr["data/fantasy/competitors.csv"] = c_path.read_text()
        b_path = breakdowns_path(PROJECT_ROOT)
        if b_path.exists():
            files_to_pr["data/fantasy/breakdowns.csv"] = b_path.read_text()
        with st.spinner("Opening PR with score updates…"):
            pr = propose_files_pr(
                files=files_to_pr,
                title=f"Score R{int(round_number)} (human {human_pts:.0f} · claude {claude_pts:.0f} · model {model_pts:.0f})",
                body="Auto-PR from Score Round page.",
                branch_prefix="score-round",
                settings=gh,
            )
        if pr.ok:
            st.success(f"PR opened: {pr.pr_url}")
        else:
            st.warning(f"PR write-back failed: {pr.message}")
    else:
        st.caption("GitHub creds not configured — files saved locally. Add `GITHUB_TOKEN` to enable auto-PR.")


# ---------------------------------------------------------------------------
# All scored rounds — quick reference table
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("Already-scored rounds")
if comp_df.empty and (hist.empty or "actual_points" not in hist.columns):
    st.caption("No scores recorded yet.")
else:
    summary = pd.DataFrame()
    if not hist.empty and "actual_points" in hist.columns:
        m = hist[["round", "actual_points"]].copy()
        m = m[pd.to_numeric(m["actual_points"], errors="coerce").notna()]
        m["team"] = THREE_TEAM_LABELS["model"]
        m["points"] = pd.to_numeric(m["actual_points"], errors="coerce")
        summary = pd.concat([summary, m[["round", "team", "points"]]], ignore_index=True)
    if not comp_df.empty:
        c = comp_df[["round", "team_name", "points"]].rename(columns={"team_name": "team"}).copy()
        summary = pd.concat([summary, c], ignore_index=True)

    cancelled_rounds = {r for r in calendar_rounds(calendar) if is_cancelled(calendar, r)}
    if cancelled_rounds and not summary.empty:
        summary = summary[~summary["round"].astype(int).isin(cancelled_rounds)]

    if summary.empty:
        st.caption("No scores recorded yet.")
    else:
        summary["Race"] = summary["round"].astype(int).apply(lambda r: format_round_label(calendar, r, short=True))
        pivot = summary.pivot_table(index="Race", columns="team", values="points", aggfunc="last")
        st.dataframe(pivot, use_container_width=True)
