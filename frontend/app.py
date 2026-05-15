from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import role_pill
from frontend.state import auth_role, init_session_state, is_owner, logout_button, require_auth


st.set_page_config(
    page_title="Fantasy F1 2026",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
require_auth()

# Persistent sidebar identity + logout (appears under nav on every page).
with st.sidebar:
    role_pill(auth_role() or "")
    logout_button()
    st.divider()


# Pages available to everyone (visitors + owner)
public_pages = [
    st.Page("views/landing.py", title="Home", default=True),
    st.Page("pages/03_performance.py", title="Performance"),
    st.Page("pages/04_history.py", title="History"),
]

# Owner-only pages (the race-week workflow)
owner_pages = [
    st.Page("pages/01_this_week.py", title="This Week"),
    st.Page("pages/02_score_round.py", title="Score Round"),
]

if is_owner():
    nav = st.navigation({"Public": public_pages, "Owner workflow": owner_pages})
else:
    nav = st.navigation({"Public": public_pages})

nav.run()
