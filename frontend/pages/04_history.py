from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import inject_theme
from frontend.state import require_auth
from src.ui_services import driver_tenure, load_history, transfer_log


require_auth()
inject_theme()

st.title("History")
st.caption("Every transfer, every reasoning note, every chip burned.")


# ---------------------------------------------------------------------------
# Transfer log
# ---------------------------------------------------------------------------
st.header("Transfer log")
log = transfer_log(PROJECT_ROOT)
if log.empty:
    st.info("No rounds locked in yet — transfer log is empty.")
else:
    show = log.copy()
    show.columns = ["Round", "In (drivers)", "Out (drivers)", "In (constructors)", "Out (constructors)", "DRS Boost", "Chips used", "Points scored", "Notes"]
    st.dataframe(show.sort_values("Round", ascending=False), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Driver tenure
# ---------------------------------------------------------------------------
st.header("Driver tenure")
st.caption("Which drivers have spent the most time on the model team.")
tenure = driver_tenure(PROJECT_ROOT)
if tenure.empty:
    st.info("No tenure data yet.")
else:
    show = tenure.copy()
    show.columns = ["Driver", "Rounds owned", "First round", "Last round"]
    st.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Per-round notes
# ---------------------------------------------------------------------------
st.header("Per-round reasoning")
hist = load_history(PROJECT_ROOT)
if hist.empty:
    st.info("Lock in some rounds to start building a season story.")
else:
    for _, r in hist.sort_values("round", ascending=False).iterrows():
        notes = str(r.get("notes", "") or "").strip()
        with st.container():
            cols = st.columns([1, 6])
            with cols[0]:
                st.markdown(
                    f"<div style='background:#1F1F2C;border-radius:6px;padding:0.6rem;text-align:center;'>"
                    f"<div style='font-size:0.65rem;opacity:0.6;letter-spacing:0.1em;'>ROUND</div>"
                    f"<div style='font-size:1.6rem;font-weight:800;'>{int(r['round'])}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(f"**Drivers:** {r['drivers']}  ·  **Constructors:** {r['constructors']}  ·  **DRS:** {r['drs_boost']}")
                if notes:
                    st.markdown(f"_{notes}_")
                else:
                    st.caption("(no notes recorded for this round)")
        st.markdown("&nbsp;")
