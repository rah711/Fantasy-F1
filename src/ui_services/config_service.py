from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import load_config


@dataclass
class ValidationResult:
    ok: bool
    updated_config: dict[str, Any] | None = None
    updated_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_config_file(config_path: str | Path | None = None) -> dict[str, Any]:
    return load_config(str(config_path) if config_path else None)


def dump_config_yaml(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False)


def generate_config_diff(old_cfg: dict[str, Any], new_cfg: dict[str, Any], file_name: str = "config.yaml") -> str:
    old_text = dump_config_yaml(old_cfg).splitlines(keepends=True)
    new_text = dump_config_yaml(new_cfg).splitlines(keepends=True)
    diff = difflib.unified_diff(old_text, new_text, fromfile=f"a/{file_name}", tofile=f"b/{file_name}")
    return "".join(diff)


def update_weather_override(
    cfg: dict[str, Any],
    round_number: int,
    rain_probability: float,
    notes: str,
    temperature_c: float | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("weather_override", {})
    out["weather_override"]["next_race_round"] = int(round_number)
    out["weather_override"]["rain_probability"] = float(max(0.0, min(1.0, rain_probability)))
    out["weather_override"]["notes"] = str(notes)
    if temperature_c is not None:
        out["weather_override"]["temperature_c"] = round(float(temperature_c), 1)
    return out


def _pick_code_column(df: pd.DataFrame, entity_type: str) -> str | None:
    candidates = [
        "asset",
        "code",
        "id",
        "driver_code" if entity_type == "driver" else "constructor_id",
        "driver" if entity_type == "driver" else "constructor",
        "DR" if entity_type == "driver" else "CR",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _pick_price_column(df: pd.DataFrame) -> str | None:
    candidates = ["price", "Price", "PRICE", "value", "Value"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def validate_and_apply_price_csv(cfg: dict[str, Any], csv_text: str, entity_type: str) -> ValidationResult:
    et = entity_type.strip().lower()
    if et not in {"driver", "constructor"}:
        return ValidationResult(ok=False, errors=["entity_type must be 'driver' or 'constructor'"])

    if not csv_text.strip():
        return ValidationResult(ok=False, errors=["CSV input is empty"])

    try:
        df = pd.read_csv(StringIO(csv_text.strip()))
    except Exception as exc:
        return ValidationResult(ok=False, errors=[f"Failed to parse CSV: {exc}"])

    if df.empty:
        return ValidationResult(ok=False, errors=["CSV has no rows"])

    code_col = _pick_code_column(df, et)
    price_col = _pick_price_column(df)

    errors: list[str] = []
    warnings: list[str] = []

    if code_col is None:
        errors.append("Could not find code column (expected one of: code/asset/driver_code/constructor_id/DR/CR)")
    if price_col is None:
        errors.append("Could not find price column (expected 'price')")
    if errors:
        return ValidationResult(ok=False, errors=errors)

    target = cfg.get("prices", {}).get("drivers" if et == "driver" else "constructors", {})
    if not isinstance(target, dict):
        return ValidationResult(ok=False, errors=["Invalid config shape for prices section"])

    out = copy.deepcopy(cfg)
    out_target = out["prices"]["drivers" if et == "driver" else "constructors"]

    updated = 0
    floor_price = 3.0

    for i, row in df.iterrows():
        raw_code = str(row.get(code_col, "")).strip()
        if not raw_code or raw_code.lower() == "nan":
            warnings.append(f"Row {i+1}: empty code; skipped")
            continue

        code = raw_code.upper() if et == "driver" else raw_code.lower()
        if code not in out_target:
            errors.append(f"Row {i+1}: unknown {et} code '{raw_code}'")
            continue

        try:
            price = float(row.get(price_col))
        except Exception:
            errors.append(f"Row {i+1}: invalid price for {raw_code}")
            continue

        if price < floor_price:
            errors.append(f"Row {i+1}: price {price} below floor {floor_price} for {raw_code}")
            continue

        out_target[code]["price"] = round(price, 3)
        updated += 1

    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    return ValidationResult(ok=True, updated_config=out, updated_rows=updated, warnings=warnings)


def update_current_team(
    cfg: dict[str, Any],
    drivers: list[str],
    constructors: list[str],
    drs_boost: str | None,
    budget: float,
    free_transfers: int,
    banked_transfers: int,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("current_team", {})
    out["current_team"]["drivers"] = [str(x).upper() for x in drivers]
    out["current_team"]["constructors"] = [str(x).lower() for x in constructors]
    out["current_team"]["drs_boost"] = str(drs_boost).upper() if drs_boost else None
    out["current_team"]["budget"] = float(budget)
    out["current_team"]["free_transfers"] = int(free_transfers)
    out["current_team"]["banked_transfers"] = int(banked_transfers)
    return out
