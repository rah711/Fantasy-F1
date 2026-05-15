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
    append_competitor_score,
    compute_team_points_for_round,
    history_path,
    load_competitor_history,
    load_history,
    propose_files_pr,
    update_actual_points,
)
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
total_rounds = int(season.get("total_rounds", 24))
weather = cfg.get("weather_override", {})
default_round = max(1, int(weather.get("next_race_round", 1)) - 1)

st.header("1. Pick the round")
round_number = st.number_input(
    "Round to score",
    min_value=1, max_value=total_rounds, value=default_round, step=1,
)
event = calendar.get(int(round_number), {}) or calendar.get(round_number, {})
if event:
    st.caption(f"**R{int(round_number)} — {event.get('name', '')}** · {event.get('country', '')} · {event.get('dates', '')}")


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
# Auto-compute model team's points if race results exist
# ---------------------------------------------------------------------------
auto_model_pts: float | None = None
if not hist_row.empty:
    drivers = str(hist_row.iloc[0].get("drivers", "")).split(",")
    drivers = [d.strip() for d in drivers if d.strip()]
    constructors = str(hist_row.iloc[0].get("constructors", "")).split(",")
    constructors = [c.strip() for c in constructors if c.strip()]
    drs = str(hist_row.iloc[0].get("drs_boost", "") or "")
    scored = compute_team_points_for_round(
        project_root=PROJECT_ROOT,
        round_number=int(round_number),
        drivers=drivers,
        constructors=constructors,
        drs_boost=drs,
    )
    if scored:
        auto_model_pts = float(scored.get("total_driver_points", 0.0))


# ---------------------------------------------------------------------------
# Score entry form
# ---------------------------------------------------------------------------
st.header("2. Enter scores")

if hist_row.empty:
    st.warning(
        f"No locked-in lineup found for R{int(round_number)} in `data/fantasy/history.csv`. "
        "Lock in the round via **This Week** first, then come back here."
    )

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**{THREE_TEAM_LABELS['human']}**")
    st.caption("Your scores from the official Fantasy F1 site.")
    human_pts = st.number_input(
        "Human team points", min_value=-200.0, max_value=1000.0,
        value=float(existing_human) if existing_human is not None else 0.0,
        step=0.5, key="score_human",
    )
with c2:
    st.markdown(f"**{THREE_TEAM_LABELS['claude_chat']}**")
    st.caption("Pure-AI Claude chat team's scores.")
    claude_pts = st.number_input(
        "Claude chat team points", min_value=-200.0, max_value=1000.0,
        value=float(existing_claude) if existing_claude is not None else 0.0,
        step=0.5, key="score_claude",
    )
with c3:
    st.markdown(f"**{THREE_TEAM_LABELS['model']}**")
    if auto_model_pts is not None:
        st.caption(f"Auto-computed from race results: **{auto_model_pts:.1f}** (you can override)")
    elif not hist_row.empty:
        st.caption("Upload race results in **This Week** to auto-compute, or enter manually.")
    default_model = existing_model_pts if existing_model_pts is not None else (auto_model_pts or 0.0)
    model_pts = st.number_input(
        "Model team points", min_value=-200.0, max_value=1000.0,
        value=float(default_model), step=0.5, key="score_model",
    )

st.write("")

if st.button("Save scores for this round", type="primary", key="save_scores"):
    paths_changed: list[str] = []

    # 1. Update model team's actual_points in history.csv (only if a row exists)
    if not hist_row.empty:
        p = update_actual_points(PROJECT_ROOT, int(round_number), float(model_pts))
        if p:
            paths_changed.append(str(p))

    # 2. Append/replace competitor rows
    p_human = append_competitor_score(PROJECT_ROOT, int(round_number), "human", float(human_pts))
    p_claude = append_competitor_score(PROJECT_ROOT, int(round_number), "claude_chat", float(claude_pts))
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

    if summary.empty:
        st.caption("No scores recorded yet.")
    else:
        pivot = summary.pivot_table(index="round", columns="team", values="points", aggfunc="last")
        st.dataframe(pivot, use_container_width=True)
