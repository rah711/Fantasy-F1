from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import readonly_notice, role_badge
from frontend.state import (
    auth_role,
    get_working_config,
    is_owner,
    set_working_config,
    logout_button,
    require_auth,
)
from src.ui_services import recommend_round, recommend_transfers, update_weather_override

st.set_page_config(page_title="Race Workspace", layout="wide")
require_auth()

role_badge(auth_role() or "")
logout_button()
if not is_owner():
    readonly_notice()

cfg = get_working_config()
season = cfg.get("season", {})
weather = cfg.get("weather_override", {})

st.title("Race Workspace")

round_default = int(weather.get("next_race_round", 1))
round_number = st.number_input(
    "Target Round",
    min_value=1,
    max_value=int(season.get("total_rounds", 24)),
    value=round_default,
    step=1,
)

rain_probability = st.slider("Rain probability", min_value=0.0, max_value=1.0, value=float(weather.get("rain_probability", 0.0)), step=0.01)
notes = st.text_input("Weather notes", value=str(weather.get("notes", "")))

c1, c2 = st.columns(2)
with c1:
    if st.button("Apply weather to draft config", disabled=not is_owner()):
        new_cfg = update_weather_override(cfg, int(round_number), float(rain_probability), notes)
        set_working_config(new_cfg)
        st.success("Draft weather updated.")

with c2:
    if st.button("Generate predictions + lineup"):
        run_cfg = get_working_config()
        out_path = None
        if not is_owner():
            out_path = Path(f"/tmp/fantasy_f1_visitor_round{int(round_number)}.parquet")
        with st.spinner("Running inference and optimizer..."):
            try:
                result = recommend_round(
                    cfg=run_cfg,
                    round_number=int(round_number),
                    output_path=out_path,
                )
                st.session_state["last_recommendation"] = result
                st.success("Recommendation generated.")
            except Exception as exc:
                st.error(f"Failed: {exc}")

rec = st.session_state.get("last_recommendation")
if rec:
    st.subheader("Lineup Recommendation")
    st.json(rec.get("recommendation", {}))
    st.caption(f"Predictions file: {rec.get('predictions_path', '')}")
    top = rec.get("top_projected_drivers", [])
    if top:
        st.dataframe(pd.DataFrame(top), use_container_width=True)

st.divider()
st.subheader("Transfer Recommendation")
default_pred_path = ""
if rec:
    default_pred_path = str(rec.get("predictions_path", ""))
predictions_path = st.text_input("Predictions parquet path", value=default_pred_path)

if st.button("Run transfer recommendation"):
    if not predictions_path.strip():
        st.error("Provide predictions parquet path first.")
    else:
        with st.spinner("Running transfer optimizer..."):
            try:
                out = recommend_transfers(
                    cfg=get_working_config(),
                    predictions_path=predictions_path,
                    season_year=int(season.get("year", 2026)),
                    round_number=int(round_number),
                )
                st.session_state["last_transfer_recommendation"] = out
                st.success("Transfer recommendation generated.")
            except Exception as exc:
                st.error(f"Failed: {exc}")

if st.session_state.get("last_transfer_recommendation"):
    out = st.session_state["last_transfer_recommendation"]
    st.json(out)
