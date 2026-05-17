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
    calendar_rounds,
    driver_tenure,
    format_round_label,
    is_cancelled,
    load_history,
    transfer_log,
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


st.title("History")
st.caption("Every transfer, every reasoning note, every chip burned.")
if cancelled_rounds:
    st.caption(f"Cancelled rounds excluded: {', '.join(f'R{r}' for r in sorted(cancelled_rounds))}")


# ---------------------------------------------------------------------------
# Transfer log
# ---------------------------------------------------------------------------
st.header("Transfer log")
log = _drop_cancelled(transfer_log(PROJECT_ROOT))
if log.empty:
    st.info("No rounds locked in yet — transfer log is empty.")
else:
    show = log.copy().sort_values("round", ascending=False)
    show["Race"] = show["round"].astype(int).apply(lambda r: format_round_label(calendar, r, short=True))
    cols_avail = [c for c in ["Race", "drivers_in", "drivers_out", "constructors_in", "constructors_out", "drs_boost", "chips_used", "chip_details", "actual_points", "notes"] if c in show.columns]
    show = show[cols_avail]
    rename_map = {
        "drivers_in": "In (drivers)", "drivers_out": "Out (drivers)",
        "constructors_in": "In (constructors)", "constructors_out": "Out (constructors)",
        "drs_boost": "DRS Boost", "chips_used": "Chips used",
        "chip_details": "Chip detail", "actual_points": "Points scored", "notes": "Notes",
    }
    show = show.rename(columns=rename_map)
    st.dataframe(show, use_container_width=True, hide_index=True)


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
hist = _drop_cancelled(load_history(PROJECT_ROOT))
if hist.empty:
    st.info("Lock in some rounds to start building a season story.")
else:
    for _, r in hist.sort_values("round", ascending=False).iterrows():
        notes = str(r.get("notes", "") or "").strip()
        race_label = format_round_label(calendar, int(r["round"]), short=True)
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
                st.markdown(f"**{race_label}**")
                st.markdown(f"**Drivers:** {r['drivers']}  ·  **Constructors:** {r['constructors']}  ·  **DRS:** {r['drs_boost']}")
                chip_used = str(r.get("chips_used", "") or "").strip()
                chip_detail = str(r.get("chip_details", "") or "").strip()
                if chip_used:
                    line = f"**Chip:** `{chip_used}`"
                    if chip_detail:
                        line += f" — {chip_detail}"
                    st.markdown(line)
                if notes:
                    st.markdown(f"_{notes}_")
                elif not chip_used:
                    st.caption("(no notes recorded for this round)")
        st.markdown("&nbsp;")
