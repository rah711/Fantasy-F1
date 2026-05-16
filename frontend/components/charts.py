"""Altair charts for the visitor pages — team brand colors, race-name tick labels,
cancelled-round indicators, and rich hover tooltips.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

from frontend.components.theme import F1_RED
from src.ui_services import calendar_rounds, format_round_label, is_cancelled


# Team brand palette for the 3 experiment teams. Matched to the landing-page cards
# so colors stay consistent across the app.
TEAM_VIZ_COLORS: dict[str, str] = {
    "Pure human judgement": "#00D2BE",
    "Pure-AI Claude chat": "#FF8000",
    "Vibe-coded data science model": F1_RED,
}


def _add_race_label(df: pd.DataFrame, calendar: dict[Any, Any]) -> pd.DataFrame:
    df = df.copy()
    df["race_label"] = df["round"].astype(int).apply(
        lambda r: format_round_label(calendar, r, short=True)
    )
    return df


def _color_scale() -> alt.Scale:
    return alt.Scale(
        domain=list(TEAM_VIZ_COLORS.keys()),
        range=list(TEAM_VIZ_COLORS.values()),
    )


def _x_axis_labels_js(calendar: dict[Any, Any], sprint_rounds: set[int] | None = None) -> str:
    """Build a Vega `labelExpr` mapping round numbers to short race labels.

    Cancelled rounds are flagged with ✗. Sprint rounds get a ★.
    """
    sprint_rounds = sprint_rounds or set()
    pairs: list[str] = []
    for r in calendar_rounds(calendar, include_cancelled=True):
        event = calendar.get(r) or calendar.get(int(r)) or {}
        country = event.get("country", "")
        cancelled = is_cancelled(calendar, r)
        marker = ""
        if cancelled:
            marker = " ✗"
        elif int(r) in sprint_rounds:
            marker = " ★"
        label = f"R{r}"
        if country:
            label += f" {country}"
        label += marker
        # Escape any embedded quotes (none expected here, but defensive)
        label = label.replace('"', '\\"')
        pairs.append(f'{int(r)}: "{label}"')
    obj = "{" + ", ".join(pairs) + "}"
    return f"{obj}[datum.value] || ('R' + datum.value)"


def _cancelled_overlays(
    calendar: dict[Any, Any], y_min: float | None = None, y_max: float | None = None
) -> tuple[alt.Chart, alt.Chart] | None:
    """Vertical dashed rules + 'cancelled' labels at cancelled rounds. Returns None if none."""
    cancelled = [r for r in calendar_rounds(calendar) if is_cancelled(calendar, r)]
    if not cancelled:
        return None
    df_rules = pd.DataFrame({"round": cancelled})
    rule = (
        alt.Chart(df_rules)
        .mark_rule(color="#888", strokeDash=[3, 3], opacity=0.7)
        .encode(x="round:Q")
    )
    label_df = pd.DataFrame({
        "round": cancelled,
        "label": ["cancelled"] * len(cancelled),
        "y": [y_max if y_max is not None else 0] * len(cancelled),
    })
    text = (
        alt.Chart(label_df)
        .mark_text(color="#888", fontSize=10, angle=270, dy=-32, align="left")
        .encode(x="round:Q", y="y:Q", text="label:N")
    )
    return rule, text


def cumulative_chart(cum_df: pd.DataFrame, calendar: dict[Any, Any]) -> alt.Chart:
    df = _add_race_label(cum_df, calendar)
    label_expr = _x_axis_labels_js(calendar)

    lines = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=3)
        .encode(
            x=alt.X(
                "round:Q", title="Round",
                axis=alt.Axis(
                    labelExpr=label_expr, labelAngle=-30, tickMinStep=1,
                    values=calendar_rounds(calendar, include_cancelled=True),
                ),
            ),
            y=alt.Y("cumulative_points:Q", title="Cumulative points"),
            color=alt.Color(
                "team_name:N", scale=_color_scale(),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("race_label:N", title="Race"),
                alt.Tooltip("team_name:N", title="Team"),
                alt.Tooltip("cumulative_points:Q", title="Cumulative", format=".0f"),
                alt.Tooltip("round_points:Q", title="This round", format=".0f"),
            ],
        )
    )
    overlays = _cancelled_overlays(calendar, y_max=float(df["cumulative_points"].max()))
    chart = lines
    if overlays:
        chart = chart + overlays[0] + overlays[1]
    return chart.properties(height=380)


def per_round_chart(
    cum_df: pd.DataFrame,
    calendar: dict[Any, Any],
    sprint_rounds: set[int] | None = None,
) -> alt.Chart:
    df = _add_race_label(cum_df, calendar)
    label_expr = _x_axis_labels_js(calendar, sprint_rounds=sprint_rounds or set())

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "round:O", title="Round",
                sort=alt.EncodingSortField("round"),
                axis=alt.Axis(labelExpr=label_expr, labelAngle=-30),
            ),
            xOffset=alt.XOffset("team_name:N"),
            y=alt.Y("round_points:Q", title="Points scored"),
            color=alt.Color(
                "team_name:N", scale=_color_scale(),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("race_label:N", title="Race"),
                alt.Tooltip("team_name:N", title="Team"),
                alt.Tooltip("round_points:Q", title="Points", format=".0f"),
            ],
        )
        .properties(height=380)
    )


def delta_vs_human_chart(cum_df: pd.DataFrame, calendar: dict[Any, Any]) -> alt.Chart | None:
    """Cumulative gap to the human team, per round. Returns None if human has no data."""
    human = cum_df[cum_df["team_key"] == "human"].set_index("round")["cumulative_points"]
    if human.empty:
        return None

    df = cum_df[cum_df["team_key"] != "human"].copy()
    if df.empty:
        return None

    df["delta_vs_human"] = df.apply(
        lambda r: float(r["cumulative_points"]) - float(human.get(int(r["round"]), 0)),
        axis=1,
    )
    df = _add_race_label(df, calendar)
    label_expr = _x_axis_labels_js(calendar)

    rule_zero = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#888", strokeDash=[5, 5])
        .encode(y="zero:Q")
    )
    baseline_label = (
        alt.Chart(pd.DataFrame({"x": [float(cum_df["round"].max())], "y": [0], "label": ["human baseline"]}))
        .mark_text(align="right", baseline="bottom", color="#888", fontSize=11, dx=-4, dy=-4)
        .encode(x="x:Q", y="y:Q", text="label:N")
    )
    lines = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=3)
        .encode(
            x=alt.X(
                "round:Q", title="Round",
                axis=alt.Axis(
                    labelExpr=label_expr, labelAngle=-30, tickMinStep=1,
                    values=calendar_rounds(calendar, include_cancelled=True),
                ),
            ),
            y=alt.Y("delta_vs_human:Q", title="Points behind / ahead of human"),
            color=alt.Color(
                "team_name:N", scale=_color_scale(),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("race_label:N", title="Race"),
                alt.Tooltip("team_name:N", title="Team"),
                alt.Tooltip("delta_vs_human:Q", title="Vs human", format="+.0f"),
                alt.Tooltip("cumulative_points:Q", title="Cumulative", format=".0f"),
            ],
        )
    )
    overlays = _cancelled_overlays(calendar, y_max=float(df["delta_vs_human"].max()))
    chart = rule_zero + baseline_label + lines
    if overlays:
        chart = chart + overlays[0] + overlays[1]
    return chart.properties(height=380)
