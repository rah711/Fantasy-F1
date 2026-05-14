from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import readonly_notice, role_badge, show_diff
from frontend.state import (
    auth_role,
    get_working_config,
    github_settings_from_secrets,
    is_owner,
    logout_button,
    require_auth,
    set_working_config,
)
from src.ui_services import (
    dump_config_yaml,
    generate_config_diff,
    load_config_file,
    propose_config_change_via_pr,
    update_current_team,
    validate_and_apply_price_csv,
)

st.set_page_config(page_title="Prices & Team Editor", layout="wide")
require_auth()

role_badge(auth_role() or "")
logout_button()
if not is_owner():
    readonly_notice()

st.title("Prices & Team Editor")

base_cfg = load_config_file()
cfg = get_working_config()

# -----------------------
# Price table editors
# -----------------------
st.subheader("Driver Prices")
drivers = cfg.get("prices", {}).get("drivers", {})
driver_df = pd.DataFrame(
    [
        {"driver_code": code, "name": meta.get("name", ""), "team": meta.get("team", ""), "price": float(meta.get("price", 0.0))}
        for code, meta in drivers.items()
    ]
).sort_values("driver_code")

edited_driver_df = st.data_editor(
    driver_df,
    column_config={"driver_code": st.column_config.TextColumn(disabled=True), "name": st.column_config.TextColumn(disabled=True), "team": st.column_config.TextColumn(disabled=True)},
    hide_index=True,
    use_container_width=True,
)
if st.button("Apply driver table edits", disabled=not is_owner()):
    out = copy.deepcopy(cfg)
    for row in edited_driver_df.to_dict(orient="records"):
        code = str(row["driver_code"]).upper()
        out["prices"]["drivers"][code]["price"] = float(row["price"])
    set_working_config(out)
    st.success("Driver price edits applied to draft config.")

st.subheader("Constructor Prices")
ctors = cfg.get("prices", {}).get("constructors", {})
ctor_df = pd.DataFrame(
    [{"constructor_id": cid, "name": meta.get("name", ""), "price": float(meta.get("price", 0.0))} for cid, meta in ctors.items()]
).sort_values("constructor_id")

edited_ctor_df = st.data_editor(
    ctor_df,
    column_config={"constructor_id": st.column_config.TextColumn(disabled=True), "name": st.column_config.TextColumn(disabled=True)},
    hide_index=True,
    use_container_width=True,
)
if st.button("Apply constructor table edits", disabled=not is_owner()):
    out = copy.deepcopy(cfg)
    for row in edited_ctor_df.to_dict(orient="records"):
        cid = str(row["constructor_id"]).lower()
        out["prices"]["constructors"][cid]["price"] = float(row["price"])
    set_working_config(out)
    st.success("Constructor price edits applied to draft config.")

st.divider()
st.subheader("Bulk CSV Price Update")
col1, col2 = st.columns(2)
with col1:
    driver_csv = st.text_area("Driver CSV (code + price)", height=140, placeholder="driver_code,price\nVER,28.1\nNOR,29.7")
    if st.button("Apply driver CSV", disabled=not is_owner()):
        res = validate_and_apply_price_csv(get_working_config(), driver_csv, "driver")
        if res.ok and res.updated_config is not None:
            set_working_config(res.updated_config)
            st.success(f"Applied {res.updated_rows} driver price updates.")
            for w in res.warnings:
                st.warning(w)
        else:
            for e in res.errors:
                st.error(e)

with col2:
    ctor_csv = st.text_area("Constructor CSV (id + price)", height=140, placeholder="constructor_id,price\nmclaren,29.1\nferrari,23.9")
    if st.button("Apply constructor CSV", disabled=not is_owner()):
        res = validate_and_apply_price_csv(get_working_config(), ctor_csv, "constructor")
        if res.ok and res.updated_config is not None:
            set_working_config(res.updated_config)
            st.success(f"Applied {res.updated_rows} constructor price updates.")
            for w in res.warnings:
                st.warning(w)
        else:
            for e in res.errors:
                st.error(e)

st.divider()
st.subheader("Current Team Editor")
team = cfg.get("current_team", {})
all_drivers = sorted(cfg.get("prices", {}).get("drivers", {}).keys())
all_ctors = sorted(cfg.get("prices", {}).get("constructors", {}).keys())

team_drivers = st.multiselect("Drivers (5)", options=all_drivers, default=list(team.get("drivers", [])))
team_ctors = st.multiselect("Constructors (2)", options=all_ctors, default=list(team.get("constructors", [])))
drs = st.selectbox("DRS driver", options=[""] + team_drivers, index=0 if not team.get("drs_boost") else ([""] + team_drivers).index(team.get("drs_boost")) if team.get("drs_boost") in ([""] + team_drivers) else 0)
budget = st.number_input("Budget", min_value=0.0, value=float(team.get("budget", 100.0)), step=0.1)
free_tf = st.number_input("Free transfers", min_value=0, value=int(team.get("free_transfers", 2)), step=1)
banked_tf = st.number_input("Banked transfers", min_value=0, value=int(team.get("banked_transfers", 0)), step=1)

if st.button("Apply team edits", disabled=not is_owner()):
    if len(team_drivers) != 5 or len(team_ctors) != 2:
        st.error("Team must have exactly 5 drivers and 2 constructors.")
    else:
        out = update_current_team(
            cfg=get_working_config(),
            drivers=team_drivers,
            constructors=team_ctors,
            drs_boost=drs or None,
            budget=float(budget),
            free_transfers=int(free_tf),
            banked_transfers=int(banked_tf),
        )
        set_working_config(out)
        st.success("Current team updated in draft config.")

st.divider()
st.subheader("Config Diff Preview")
current_cfg = get_working_config()
diff = generate_config_diff(base_cfg, current_cfg)
show_diff(diff)

updated_yaml = dump_config_yaml(current_cfg)
st.download_button(
    label="Download updated config.yaml",
    data=updated_yaml,
    file_name="config.yaml",
    mime="text/yaml",
)

st.subheader("Propose Config Update (GitHub PR)")
pr_title = st.text_input("PR title", value="Update fantasy config from frontend")
pr_body = st.text_area("PR body", value="Automated config update from Streamlit frontend.")

if st.button("Create branch + PR", disabled=(not is_owner())):
    settings = github_settings_from_secrets()
    res = propose_config_change_via_pr(
        updated_cfg_yaml=updated_yaml,
        title=pr_title.strip() or "Update fantasy config from frontend",
        body=pr_body.strip() or "Automated update",
        settings=settings,
    )
    if res.ok:
        st.success(f"PR created. Branch: {res.branch_name}")
        if res.pr_url:
            st.markdown(f"[Open PR]({res.pr_url})")
    else:
        st.error(res.message)
