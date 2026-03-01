"""
Contextual features: weather condition, season phase, era_weight.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import get_season_phase


def add_contextual_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add weather, season phase, and regulation era_weight.

    Expects df to have round, and optionally air_temp_c, track_temp_c, rainfall.
    Adds: season_phase (early/mid/late), era_weight (from config), rainfall_flag.
    """
    if df.empty:
        return df
    out = df.copy()
    reg = config.get("regulation", {})
    era = reg.get("era_weight", 0.05)

    out["season_phase"] = out["round"].map(lambda r: get_season_phase(int(r)) if pd.notna(r) else "")
    out["era_weight"] = era
    out["rainfall_flag"] = out.get("rainfall", pd.Series(dtype=bool)).fillna(False)
    return out
