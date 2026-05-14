from __future__ import annotations

import streamlit as st


F1_RED = "#E10600"
F1_BLACK = "#15151E"
F1_DARK_GREY = "#1F1F2C"
F1_LIGHT_GREY = "#38383F"
F1_OFF_WHITE = "#F0F0F0"


TEAM_COLORS: dict[str, str] = {
    "red_bull": "#1E5BC6",
    "mercedes": "#00D2BE",
    "ferrari": "#DC0000",
    "mclaren": "#FF8000",
    "aston_martin": "#229971",
    "alpine": "#0093CC",
    "williams": "#1868DB",
    "racing_bulls": "#6692FF",
    "audi": "#52E252",
    "haas": "#B6BABD",
    "kick_sauber": "#52E252",
    "alpha_tauri": "#6692FF",
    "alfa_romeo": "#900000",
}


def team_color(team_id: str | None, fallback: str = F1_LIGHT_GREY) -> str:
    if not team_id:
        return fallback
    return TEAM_COLORS.get(str(team_id).lower(), fallback)


def driver_palette(driver_team_pairs: list[tuple[str, str]]) -> dict[str, str]:
    return {driver: team_color(team) for driver, team in driver_team_pairs}


_CUSTOM_CSS = """
<style>
/* F1 vibes — typography + accent polish */
html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, sans-serif;
    letter-spacing: 0.01em;
}

/* Page title — bigger, bolder, with red underline */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    border-bottom: 3px solid #E10600;
    padding-bottom: 0.4rem;
    display: inline-block;
}

/* Section headers — racing-stripe accent */
h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}

/* Metric cards — feel like a pit-wall HUD */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(225,6,0,0.08), rgba(255,255,255,0.02));
    border: 1px solid rgba(225,6,0,0.25);
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
}
[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem !important;
    opacity: 0.7;
}

/* Buttons — racing red */
.stButton button[kind="primary"] {
    background-color: #E10600 !important;
    border: none !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stButton button:hover {
    border-color: #E10600 !important;
}

/* Tabs — racing-stripe underline on active */
[data-baseweb="tab"][aria-selected="true"] {
    color: #E10600 !important;
    border-bottom-color: #E10600 !important;
}

/* Data tables — quieter borders */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
}

/* Sidebar — slightly darker for hierarchy */
[data-testid="stSidebar"] {
    background-color: #0E0E16;
    border-right: 1px solid rgba(225,6,0,0.15);
}

</style>
"""


def inject_theme() -> None:
    """Inject F1-themed CSS. Call once near the top of every page."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def role_pill(role: str) -> None:
    """Render a small role indicator pill in F1 colors."""
    color = F1_RED if role == "owner" else F1_LIGHT_GREY
    label = "TEAM PRINCIPAL" if role == "owner" else "PADDOCK PASS"
    st.markdown(
        f'<div style="display:inline-block;background:{color};color:#fff;'
        f'padding:0.25rem 0.7rem;border-radius:999px;font-size:0.7rem;'
        f'font-weight:700;letter-spacing:0.1em;">{label}</div>',
        unsafe_allow_html=True,
    )
