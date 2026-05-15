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
    set_working_config,
)
from src.ui_services import (
    append_lockin,
    c_to_f,
    compute_team_points_for_round,
    dump_config_yaml,
    generate_config_diff,
    history_path,
    ingest_constructor_prices,
    ingest_driver_prices,
    ingest_qualifying_results,
    ingest_race_results,
    load_config_file,
    parse_weather_description,
    propose_files_pr,
    recommend_round,
    recommend_transfers,
    update_current_team,
    update_weather_override,
)


require_auth()

if not is_owner():
    inject_theme()
    st.title("This Week")
    st.warning("This page is for the team principal only. Switch to **Performance** in the sidebar to follow the season.")
    st.stop()

inject_theme()

st.title("This Week")
st.caption("Upload latest data → see recommendation → lock in.")


# ---------------------------------------------------------------------------
# 1. Race context
# ---------------------------------------------------------------------------
cfg = get_working_config()
season = cfg.get("season", {})
weather = cfg.get("weather_override", {})
team = cfg.get("current_team", {})
calendar = season.get("calendar", {})

st.header("1. Race context")

ctx_a, ctx_b = st.columns([1, 3])
with ctx_a:
    round_default = int(weather.get("next_race_round", 1))
    round_number = st.number_input(
        "Next round",
        min_value=1,
        max_value=int(season.get("total_rounds", 24)),
        value=round_default,
        step=1,
    )
with ctx_b:
    event = calendar.get(int(round_number), {}) or calendar.get(round_number, {})
    if event:
        st.markdown("&nbsp;")
        st.markdown(
            f"### R{int(round_number)} — {event.get('name', '')}  "
            f"<span style='color:#999;font-size:0.7em;'>{event.get('country', '')} · {event.get('dates', '')}</span>",
            unsafe_allow_html=True,
        )

st.markdown("##### Weather")
forecast_text = st.text_area(
    "Paste a forecast description (optional)",
    value="",
    placeholder="e.g. 22°C, partly cloudy with a 60% chance of light showers around 3pm",
    height=68,
    key="forecast_text",
)
if st.button("Parse forecast", key="parse_forecast"):
    parsed = parse_weather_description(forecast_text)
    if parsed.rain_probability is not None:
        st.session_state["wx_rain"] = parsed.rain_probability
    if parsed.temperature_c is not None:
        st.session_state["wx_temp_c"] = parsed.temperature_c
    bits = []
    if parsed.matched_phrase:
        bits.append(f"rain → **{int((parsed.rain_probability or 0)*100)}%** (matched: _{parsed.matched_phrase}_)")
    if parsed.matched_temperature_phrase:
        bits.append(f"temperature → **{parsed.temperature_c:.1f}°C** (matched: _{parsed.matched_temperature_phrase}_)")
    if bits:
        st.success("Parsed: " + " · ".join(bits))
    else:
        st.warning("Couldn't extract rain or temperature. Set them manually below.")

wa, wb, wc = st.columns([1, 1, 2])
with wa:
    rain = st.slider(
        "Rain probability",
        min_value=0.0, max_value=1.0,
        value=float(st.session_state.get("wx_rain", weather.get("rain_probability", 0.0))),
        step=0.05,
        key="wx_rain",
    )
with wb:
    temp_c = st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=55.0,
        value=float(st.session_state.get("wx_temp_c", weather.get("temperature_c", 22.0))),
        step=0.5,
        key="wx_temp_c",
    )
    st.caption(f"= **{c_to_f(temp_c):.0f}°F** · {'cool tyres' if temp_c < 18 else 'warm tyres' if temp_c < 30 else 'hot tyres → faster deg'}")
with wc:
    notes = st.text_input("Notes (kept in history)", value=str(weather.get("notes", "")))


# ---------------------------------------------------------------------------
# 2. Latest data (CSV uploads)
# ---------------------------------------------------------------------------
st.header("2. Latest data")
st.caption("Drop in this week's CSVs. Each tab is independent — upload only what's new.")

tabs = st.tabs(["Driver prices", "Constructor prices", "Last race results", "Last qualifying"])


def _csv_input(label: str, key: str) -> str:
    """Return the CSV text from either a file upload or pasted area."""
    upload = st.file_uploader(f"Upload {label} CSV", type=["csv"], key=f"{key}_file")
    pasted = st.text_area(f"…or paste {label} CSV", key=f"{key}_paste", height=140, placeholder="code,price\nVER,28.1\nNOR,26.8\n…")
    if upload is not None:
        return upload.getvalue().decode("utf-8")
    return pasted


with tabs[0]:
    text = _csv_input("driver prices", "drv_prices")
    if st.button("Apply driver prices", key="apply_drv"):
        res = ingest_driver_prices(get_working_config(), text)
        if res.ok:
            set_working_config(res.updated_config)
            st.success(f"Updated prices for {res.rows} drivers.")
            for w in res.warnings:
                st.warning(w)
        else:
            for e in res.errors:
                st.error(e)

with tabs[1]:
    text = _csv_input("constructor prices", "ctor_prices")
    if st.button("Apply constructor prices", key="apply_ctor"):
        res = ingest_constructor_prices(get_working_config(), text)
        if res.ok:
            set_working_config(res.updated_config)
            st.success(f"Updated prices for {res.rows} constructors.")
            for w in res.warnings:
                st.warning(w)
        else:
            for e in res.errors:
                st.error(e)

with tabs[2]:
    text = _csv_input("race results (round just finished)", "race_res")
    last_round = max(1, int(round_number) - 1)
    st.caption(f"These are saved as the results for **R{last_round}** (the round before the upcoming one).")
    if st.button("Save race results", key="save_race"):
        res = ingest_race_results(text, PROJECT_ROOT, last_round)
        if res.ok:
            st.success(f"Saved {res.rows} drivers' race results to {res.saved_path}")
            for w in res.warnings:
                st.warning(w)
            if res.parsed is not None and not res.parsed.empty:
                st.dataframe(res.parsed.head(10), use_container_width=True)
        else:
            for e in res.errors:
                st.error(e)

with tabs[3]:
    text = _csv_input("qualifying results", "quali_res")
    last_round = max(1, int(round_number) - 1)
    st.caption(f"Saved as qualifying for **R{last_round}**.")
    if st.button("Save qualifying results", key="save_quali"):
        res = ingest_qualifying_results(text, PROJECT_ROOT, last_round)
        if res.ok:
            st.success(f"Saved {res.rows} drivers' qualifying results to {res.saved_path}")
            for w in res.warnings:
                st.warning(w)
            if res.parsed is not None and not res.parsed.empty:
                st.dataframe(res.parsed.head(10), use_container_width=True)
        else:
            for e in res.errors:
                st.error(e)


# ---------------------------------------------------------------------------
# 3. Generate recommendation
# ---------------------------------------------------------------------------
st.header("3. Recommendation")

col_gen, col_info = st.columns([1, 3])
with col_gen:
    if st.button("Generate", type="primary", use_container_width=True, key="gen_rec"):
        cfg_now = update_weather_override(
            get_working_config(),
            int(round_number),
            float(rain),
            str(notes),
            temperature_c=float(temp_c),
        )
        set_working_config(cfg_now)
        try:
            with st.spinner("Running model + optimizer…"):
                round_out = recommend_round(cfg=cfg_now, round_number=int(round_number))
                if int(round_number) == 1:
                    st.session_state["wizard_mode"] = "initial"
                    st.session_state["wizard_recommendation"] = round_out
                else:
                    transfer_out = recommend_transfers(
                        cfg=cfg_now,
                        predictions_path=round_out["predictions_path"],
                        season_year=int(season.get("year", 2026)),
                        round_number=int(round_number),
                    )
                    st.session_state["wizard_mode"] = "transfers"
                    st.session_state["wizard_recommendation"] = {
                        "round_out": round_out,
                        "transfer_out": transfer_out,
                    }
        except FileNotFoundError as e:
            st.error(f"Missing artifact: {e}. Run the data pipeline + train the model first (see README).")
            st.stop()
        except Exception as e:
            st.error(f"Recommendation failed: {e}")
            st.stop()
with col_info:
    st.caption(
        "Round 1 → suggests a fresh 5+2 lineup. "
        "Every other round → suggests transfers from your current team within your free-transfer allowance."
    )


def _render_initial(round_out: dict) -> None:
    rec = round_out["recommendation"]
    st.subheader(f"Suggested initial team — R{rec['round_number']}")
    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Total cost", f"£{rec['total_cost']:.1f}M")
    with cB:
        st.metric("Expected points", f"{rec['expected_points_next_race']:.1f}")
    with cC:
        st.metric("DRS Boost", rec["drs_boost"])

    drivers = rec["drivers"]
    constructors = rec["constructors"]
    drivers_cfg = cfg.get("prices", {}).get("drivers", {})
    constructors_cfg = cfg.get("prices", {}).get("constructors", {})

    rows = []
    for d in drivers:
        meta = drivers_cfg.get(d, {})
        rows.append({"Driver": d, "Name": meta.get("name", ""), "Team": meta.get("team", ""), "Price (£M)": meta.get("price", "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    crows = []
    for c in constructors:
        meta = constructors_cfg.get(c, {})
        crows.append({"Constructor": c, "Name": meta.get("name", ""), "Price (£M)": meta.get("price", "")})
    st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)


def _render_transfers(payload: dict) -> None:
    transfer_out = payload["transfer_out"]
    rec = transfer_out["recommendation"]
    drivers_in = transfer_out["drivers_in"]
    drivers_out = transfer_out["drivers_out"]
    ctors_in = transfer_out["constructors_in"]
    ctors_out = transfer_out["constructors_out"]
    free_alw = transfer_out["free_transfer_allowance"]
    n_trans = transfer_out["num_transfers"]
    penalty = transfer_out["transfer_penalty_points"]

    st.subheader(f"Suggested transfers — R{rec['round_number']}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Transfers", f"{n_trans} / {free_alw} free")
    with m2:
        st.metric("Penalty", f"-{penalty:.0f} pts" if penalty else "0 pts")
    with m3:
        st.metric("Expected pts", f"{rec['expected_points_next_race']:.1f}")
    with m4:
        st.metric("Team cost", f"£{rec['total_cost']:.1f}M")

    if not drivers_in and not drivers_out and not ctors_in and not ctors_out:
        st.info("Model says: **hold** — no transfers worth making this week.")
    else:
        if drivers_in or drivers_out:
            st.markdown("##### Drivers")
            io = max(len(drivers_in), len(drivers_out))
            for i in range(io):
                left = drivers_out[i] if i < len(drivers_out) else "—"
                right = drivers_in[i] if i < len(drivers_in) else "—"
                left_meta = cfg.get("prices", {}).get("drivers", {}).get(left, {})
                right_meta = cfg.get("prices", {}).get("drivers", {}).get(right, {})
                st.markdown(
                    f"- **OUT** `{left}` ({left_meta.get('name', '')}, £{left_meta.get('price', '?')}M)  "
                    f"→  **IN** `{right}` ({right_meta.get('name', '')}, £{right_meta.get('price', '?')}M)"
                )
        if ctors_in or ctors_out:
            st.markdown("##### Constructors")
            io = max(len(ctors_in), len(ctors_out))
            for i in range(io):
                left = ctors_out[i] if i < len(ctors_out) else "—"
                right = ctors_in[i] if i < len(ctors_in) else "—"
                st.markdown(f"- **OUT** `{left}`  →  **IN** `{right}`")

    st.markdown("##### Resulting team")
    drivers = rec["drivers"]
    constructors = rec["constructors"]
    drivers_cfg = cfg.get("prices", {}).get("drivers", {})
    constructors_cfg = cfg.get("prices", {}).get("constructors", {})
    rows = []
    for d in drivers:
        meta = drivers_cfg.get(d, {})
        rows.append({"Driver": d, "Name": meta.get("name", ""), "Team": meta.get("team", ""), "Price (£M)": meta.get("price", "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    crows = []
    for c in constructors:
        meta = constructors_cfg.get(c, {})
        crows.append({"Constructor": c, "Name": meta.get("name", ""), "Price (£M)": meta.get("price", "")})
    st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)

    st.markdown("##### DRS Boost")
    drs = rec["drs_boost"]
    drs_meta = drivers_cfg.get(drs, {})
    st.markdown(
        f"Recommended: **{drs}** ({drs_meta.get('name', '')}). "
        f"DRS Boost doubles the chosen driver's full weekend score (qualifying + race + sprint if applicable)."
    )

    st.markdown("##### Chip considerations")
    chips_left = team.get("chips_remaining", []) or []
    if not chips_left:
        st.caption("No chips remaining.")
    else:
        chip_notes = _chip_notes(chips_left, rain_prob=float(rain), is_sprint=int(round_number) in set(season.get("sprint_rounds", [])))
        for chip, note in chip_notes.items():
            st.markdown(f"- **{chip}** — {note}")


def _chip_notes(chips_left: list[str], rain_prob: float, is_sprint: bool) -> dict[str, str]:
    notes: dict[str, str] = {}
    for chip in chips_left:
        cl = chip.lower()
        if cl == "no_negative":
            if rain_prob >= 0.5:
                notes[chip] = "_Worth considering_ — high rain probability raises DNF risk."
            else:
                notes[chip] = "Save for a wet/chaotic round."
        elif cl == "extra_drs_boost":
            if is_sprint:
                notes[chip] = "_Worth considering_ — sprint weekend means more scoring sessions to double."
            else:
                notes[chip] = "Save for a sprint weekend or a high-confidence pick."
        elif cl == "wildcard":
            notes[chip] = "Use only if the optimal team would gain 30+ points over your current team and free transfers can't cover it."
        elif cl == "limitless":
            notes[chip] = "Use when one expensive driver dominates a track (e.g. Verstappen at high-downforce circuits in the wet)."
        elif cl == "autopilot":
            notes[chip] = "Use if you're locked into a poor performer you can't transfer out."
        elif cl == "final_fix":
            notes[chip] = "Hold for late-season qualifying surprises — lets you swap one driver after Saturday."
        else:
            notes[chip] = "Save for later."
    return notes


payload = st.session_state.get("wizard_recommendation")
mode = st.session_state.get("wizard_mode")
if payload and mode == "initial":
    _render_initial(payload)
elif payload and mode == "transfers":
    _render_transfers(payload)
else:
    st.caption("Click _Generate_ above to see this week's recommendation.")


# ---------------------------------------------------------------------------
# 4. Lock in
# ---------------------------------------------------------------------------
st.header("4. Lock in")

if not payload:
    st.caption("Generate a recommendation first.")
else:
    if mode == "initial":
        rec = payload["recommendation"]
    else:
        rec = payload["transfer_out"]["recommendation"]

    locked_drivers = list(rec["drivers"])
    locked_constructors = list(rec["constructors"])
    locked_drs = rec["drs_boost"]

    lk1, lk2 = st.columns(2)
    with lk1:
        budget_after = st.number_input(
            "Budget remaining after this lock-in (£M)",
            min_value=0.0, max_value=200.0, value=float(team.get("budget", 0.0)), step=0.1,
        )
        free_after = st.number_input(
            "Free transfers next round",
            min_value=0, max_value=10, value=int(team.get("free_transfers", 2)), step=1,
        )
    with lk2:
        banked_after = st.number_input(
            "Banked transfers",
            min_value=0, max_value=10, value=int(team.get("banked_transfers", 0)), step=1,
        )
        chips_used = st.multiselect(
            "Chips used this week (if any)",
            options=team.get("chips_remaining", []) or [],
            default=[],
        )
    lockin_notes = st.text_input("Notes (optional, shown in visitor view)", value="")

    if st.button("Lock in this team", type="primary", key="lockin"):
        new_cfg = update_current_team(
            get_working_config(),
            drivers=locked_drivers,
            constructors=locked_constructors,
            drs_boost=locked_drs,
            budget=float(budget_after),
            free_transfers=int(free_after),
            banked_transfers=int(banked_after),
        )
        # Move chips from remaining to used
        ct = new_cfg.setdefault("current_team", {})
        used_now = list(ct.get("chips_used", []) or [])
        rem_now = [c for c in (ct.get("chips_remaining", []) or []) if c not in chips_used]
        used_now.extend(chips_used)
        ct["chips_used"] = used_now
        ct["chips_remaining"] = rem_now
        set_working_config(new_cfg)

        hist_csv_path = append_lockin(
            project_root=PROJECT_ROOT,
            round_number=int(round_number),
            drivers=locked_drivers,
            constructors=locked_constructors,
            drs_boost=locked_drs,
            chips_used=chips_used,
            budget_after=float(budget_after),
            free_transfers_after=int(free_after),
            banked_transfers_after=int(banked_after),
            notes=lockin_notes,
        )
        st.success(f"Locked in. History updated at {hist_csv_path}.")

        # Auto-PR (config.yaml + history.csv) if GitHub creds are configured
        gh = github_settings_from_secrets()
        if gh.get("token") and gh["token"] != "ghp_xxx":
            base_cfg = load_config_file()
            updated_yaml = dump_config_yaml(new_cfg)
            diff = generate_config_diff(base_cfg, new_cfg)
            files_to_pr: dict[str, str] = {"config.yaml": updated_yaml}
            if hist_csv_path.exists():
                files_to_pr["data/fantasy/history.csv"] = hist_csv_path.read_text()
            with st.spinner("Opening PR with config + history update…"):
                pr = propose_files_pr(
                    files=files_to_pr,
                    title=f"Lock in R{int(round_number)} team",
                    body=f"Auto-PR from This Week wizard.\n\n```diff\n{diff[:3000]}\n```",
                    branch_prefix="lockin",
                    settings=gh,
                )
            if pr.ok:
                st.success(f"PR opened: {pr.pr_url}")
            else:
                st.warning(f"PR write-back failed: {pr.message}")
        else:
            st.caption("GitHub credentials not configured — history saved locally only. (Add `GITHUB_TOKEN` to `.streamlit/secrets.toml` to auto-PR.)")

        # If we have race results for the previous round, surface what your team scored
        scored = compute_team_points_for_round(
            project_root=PROJECT_ROOT,
            round_number=max(1, int(round_number) - 1),
            drivers=team.get("drivers", []),
            constructors=team.get("constructors", []),
            drs_boost=team.get("drs_boost"),
        )
        if scored:
            st.markdown("##### Last round's actual points")
            st.metric("Total driver points scored", f"{scored['total_driver_points']:.1f}")
            st.dataframe(pd.DataFrame(scored["drivers"]), use_container_width=True, hide_index=True)
