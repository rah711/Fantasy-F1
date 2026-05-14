from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import render_health_table, role_badge
from frontend.state import auth_role, get_working_config, init_session_state, logout_button, require_auth
from src.ui_services import collect_health_checks

st.set_page_config(page_title="Fantasy F1 2026 Console", layout="wide")

init_session_state()
require_auth()

role_badge(auth_role() or "")
logout_button()

cfg = get_working_config()
season = cfg.get("season", {})
weather = cfg.get("weather_override", {})
team = cfg.get("current_team", {})

st.title("Fantasy F1 2026 - Weekly Control Tower")

c1, c2, c3 = st.columns(3)
with c1:
    round_num = int(weather.get("next_race_round", 1))
    event = season.get("calendar", {}).get(round_num, {})
    st.metric("Next Round", f"R{round_num}")
    st.caption(f"{event.get('name', 'Unknown')} ({event.get('country', '')})")
with c2:
    st.metric("Current Budget", f"{float(team.get('budget', 100.0)):.1f}M")
    st.caption(f"Free transfers: {int(team.get('free_transfers', 2))} | Banked: {int(team.get('banked_transfers', 0))}")
with c3:
    st.metric("Rain Probability", f"{100*float(weather.get('rain_probability', 0.0)):.0f}%")
    st.caption(str(weather.get("notes", "")))

st.subheader("Current Team")
st.write(
    {
        "drivers": team.get("drivers", []),
        "constructors": team.get("constructors", []),
        "drs_boost": team.get("drs_boost"),
    }
)

st.subheader("Artifact Health")
health = collect_health_checks(PROJECT_ROOT)
render_health_table(health)

st.subheader("Runbook")
runbook = PROJECT_ROOT / "docs/weekly_workflow_2026.md"
if runbook.exists():
    st.markdown(runbook.read_text())
else:
    st.info("Weekly runbook not found.")

st.info("Use the left sidebar to open Race Workspace, Prices & Team Editor, Analytics, and Settings.")
