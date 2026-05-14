"""Parse free-text weather forecasts into structured fields.

Used by the wizard so the user can paste "22C, light showers around 3pm,
chance of rain 60%" and get rain_probability + temperature_c filled in.

Pure Python — no API calls. Falls back gracefully when fields are missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class WeatherParse:
    rain_probability: float | None
    temperature_c: float | None
    matched_phrase: str | None = None
    matched_temperature_phrase: str | None = None


# Keyword → rain probability. First match wins (most-specific patterns first).
_RAIN_KEYWORDS: list[tuple[str, float]] = [
    (r"\bthunderstorm(?:s)?\b|\bheavy rain\b|\bdownpour\b|\btorrential\b|\bstorm(?:s)?\b", 0.9),
    (r"\bwet\b|\bmonsoon\b|\bpersistent rain\b", 0.8),
    (r"\bshowers?\b|\brain expected\b|\brainy\b", 0.65),
    (r"\bdrizzle\b|\blight rain\b|\bscattered showers?\b|\boccasional rain\b|\bchance of rain\b", 0.5),
    (r"\bovercast\b|\bcloudy\b|\bgrey\b|\bgray\b", 0.25),
    (r"\bpartly cloudy\b|\bsome cloud\b|\bmostly cloudy\b", 0.2),
    (r"\bclear\b|\bsunny\b|\bdry\b|\bfair\b|\bblue sk(?:y|ies)\b", 0.05),
]


_PCT_RE = re.compile(r"(\d{1,3})\s*%")
_TEMP_C_RE = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*)?c\b", re.IGNORECASE)
_TEMP_F_RE = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s*(?:°\s*)?f\b", re.IGNORECASE)


def parse_weather_description(text: str) -> WeatherParse:
    if not text or not text.strip():
        return WeatherParse(rain_probability=None, temperature_c=None)

    t = text.lower()

    rain: float | None = None
    matched_phrase: str | None = None

    pct = _PCT_RE.search(t)
    if pct:
        rain = max(0.0, min(1.0, int(pct.group(1)) / 100))
        matched_phrase = pct.group(0)
    else:
        for pattern, score in _RAIN_KEYWORDS:
            m = re.search(pattern, t)
            if m:
                rain = score
                matched_phrase = m.group(0)
                break

    temp_c: float | None = None
    matched_temp: str | None = None
    c = _TEMP_C_RE.search(t)
    if c:
        temp_c = float(c.group(1))
        matched_temp = c.group(0)
    else:
        f = _TEMP_F_RE.search(t)
        if f:
            temp_c = (float(f.group(1)) - 32.0) * 5.0 / 9.0
            matched_temp = f.group(0)

    return WeatherParse(
        rain_probability=rain,
        temperature_c=temp_c,
        matched_phrase=matched_phrase,
        matched_temperature_phrase=matched_temp,
    )


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0
