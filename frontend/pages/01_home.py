from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import render_health_table, role_badge
from frontend.state import auth_role, get_working_config, logout_button, require_auth
from src.ui_services import collect_health_checks

st.set_page_config(page_title="Home - Fantasy F1", layout="wide")
require_auth()

role_badge(auth_role() or "")
logout_button()

cfg = get_working_config()
st.title("Home / Weekly Control Tower")

weather = cfg.get("weather_override", {})
team = cfg.get("current_team", {})
st.write(
    {
        "next_round": weather.get("next_race_round", 1),
        "rain_probability": weather.get("rain_probability", 0.0),
        "team_budget": team.get("budget", 100.0),
        "free_transfers": team.get("free_transfers", 2),
        "banked_transfers": team.get("banked_transfers", 0),
    }
)

st.subheader("Health Checks")
render_health_table(collect_health_checks(PROJECT_ROOT))
