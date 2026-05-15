"""Calendar helpers: render labels, list active rounds, handle cancellations.

A round in `season.calendar` can be marked cancelled by setting `cancelled: true`
in config.yaml — e.g.:

    4: {name: "Bahrain", country: "BHR", dates: "10-12 Apr", cancelled: true}

Cancelled rounds remain in the calendar (so round numbers don't shift) but are
skipped when picking sensible defaults.
"""

from __future__ import annotations

from typing import Any


def _event_for(calendar: dict[Any, Any], round_num: int) -> dict[str, Any]:
    return calendar.get(round_num) or calendar.get(int(round_num)) or {}


def is_cancelled(calendar: dict[Any, Any], round_num: int) -> bool:
    return bool(_event_for(calendar, round_num).get("cancelled", False))


def format_round_label(calendar: dict[Any, Any], round_num: int, short: bool = False) -> str:
    event = _event_for(calendar, round_num)
    name = event.get("name", "?")
    country = event.get("country", "")
    dates = event.get("dates", "")
    cancelled = bool(event.get("cancelled", False))

    if short:
        label = f"R{round_num} — {name}"
        if cancelled:
            label += " (cancelled)"
        return label

    label = f"R{round_num} — {name}"
    extras = [x for x in (country, dates) if x]
    if extras:
        label += " · " + " · ".join(extras)
    if cancelled:
        label += "  · CANCELLED"
    return label


def calendar_rounds(calendar: dict[Any, Any], include_cancelled: bool = True) -> list[int]:
    """All round numbers in the calendar, sorted."""
    rounds: list[int] = []
    for k, v in (calendar or {}).items():
        try:
            rn = int(k)
        except (TypeError, ValueError):
            continue
        if not include_cancelled and bool((v or {}).get("cancelled", False)):
            continue
        rounds.append(rn)
    return sorted(rounds)


def next_active_round(calendar: dict[Any, Any], target: int) -> int:
    """First non-cancelled round >= target. Falls back to the last round if none qualify."""
    for r in calendar_rounds(calendar, include_cancelled=False):
        if r >= int(target):
            return r
    actives = calendar_rounds(calendar, include_cancelled=False)
    if actives:
        return actives[-1]
    rounds = calendar_rounds(calendar, include_cancelled=True)
    return rounds[-1] if rounds else int(target)


def previous_active_round(calendar: dict[Any, Any], target: int) -> int:
    """Most recent non-cancelled round < target. Falls back to round 1."""
    earlier = [r for r in calendar_rounds(calendar, include_cancelled=False) if r < int(target)]
    return earlier[-1] if earlier else max(1, int(target) - 1)
