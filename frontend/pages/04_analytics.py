from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import file_download_button, render_health_table, role_badge
from frontend.state import auth_role, logout_button, require_auth
from src.ui_services import load_analytics_bundle

st.set_page_config(page_title="Analytics", layout="wide")
require_auth()
role_badge(auth_role() or "")
logout_button()

st.title("Analytics")
bundle = load_analytics_bundle(PROJECT_ROOT)

st.subheader("Artifact Health")
render_health_table(bundle.get("health", {}))

st.subheader("KPI Summary")
kpi = bundle.get("kpi_summary", pd.DataFrame())
if kpi.empty:
    st.info("No KPI summary CSV found.")
else:
    st.dataframe(kpi, use_container_width=True)
    if {"year", "kpi_percentile"}.issubset(kpi.columns):
        chart_df = kpi[["year", "kpi_percentile"]].set_index("year")
        st.line_chart(chart_df)

st.subheader("Strict Train/Test Tuning")
strict_train = bundle.get("strict_tuning_train", pd.DataFrame())
strict_summary = bundle.get("strict_tuning_summary", {})
if strict_train.empty:
    st.info("No strict tuning train results found.")
else:
    st.dataframe(strict_train, use_container_width=True)
if strict_summary:
    st.json(strict_summary)

st.subheader("Price Weight Sweep")
sweep = bundle.get("price_weight_sweep", pd.DataFrame())
if sweep.empty:
    st.info("No price weight sweep CSV found.")
else:
    st.dataframe(sweep, use_container_width=True)
    if {"price_appreciation_weight", "total_points", "ending_budget"}.issubset(sweep.columns):
        st.line_chart(sweep.set_index("price_appreciation_weight")[["total_points", "ending_budget"]])

png_paths = [
    PROJECT_ROOT / "data/processed/optimizer_reports/kpi_dashboard.png",
    PROJECT_ROOT / "data/processed/optimizer_reports/price_weight_sweep_2025.png",
]
for p in png_paths:
    if p.exists():
        st.image(str(p), caption=p.name)

st.subheader("Downloads")
file_download_button(PROJECT_ROOT / "data/processed/optimizer_reports/kpi_summary.csv", "Download kpi_summary.csv")
file_download_button(PROJECT_ROOT / "data/processed/optimizer_tuning_strict_weight/optimizer_tuning_best_summary.json", "Download strict_tuning_summary.json")
file_download_button(PROJECT_ROOT / "data/processed/optimizer_reports/price_weight_sweep_2025.csv", "Download price_weight_sweep_2025.csv")
