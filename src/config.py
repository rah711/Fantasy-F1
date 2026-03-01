"""
Configuration loader for Fantasy F1 2026.

This module reads config.yaml and makes all parameters available
to the rest of the codebase. Every manual parameter — scoring rules,
prices, team overrides, model settings — lives in config.yaml so
you never need to edit Python code for weekly updates.

Usage:
    from src.config import load_config
    cfg = load_config()
    budget = cfg['fantasy']['budget']  # 100.0
    sprint_rounds = cfg['season']['sprint_rounds']  # [2, 6, 7, ...]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Project root = the directory containing config.yaml
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and validate config.yaml.

    Args:
        config_path: Optional path to config file. Defaults to
                     config.yaml in the project root.

    Returns:
        Dictionary with all configuration parameters.

    Raises:
        FileNotFoundError: If config.yaml doesn't exist.
        ValueError: If required sections are missing.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Expected it in the project root directory."
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Check that all required top-level sections exist."""
    required_sections = [
        "season",
        "scoring",
        "fantasy",
        "prices",
        "circuits",
        "teams",
        "model",
        "regulation",
        "optimizer",
        "data",
    ]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(
            f"Config is missing required sections: {missing}. "
            f"Check config.yaml for typos."
        )

    # Validate driver count matches expectations
    driver_count = len(cfg["prices"]["drivers"])
    if driver_count < 20:
        print(
            f"Warning: Only {driver_count} drivers in config. "
            f"Expected at least 20 for a full grid."
        )

    # Validate constructor count
    constructor_count = len(cfg["prices"]["constructors"])
    if constructor_count < 10:
        print(
            f"Warning: Only {constructor_count} constructors in config. "
            f"Expected at least 10."
        )


def get_driver_price(cfg: dict, driver_code: str) -> float:
    """Get a driver's current price from config."""
    return cfg["prices"]["drivers"][driver_code]["price"]


def get_constructor_price(cfg: dict, constructor_id: str) -> float:
    """Get a constructor's current price from config."""
    return cfg["prices"]["constructors"][constructor_id]["price"]


def get_team_for_driver(cfg: dict, driver_code: str) -> str:
    """Get the constructor ID for a driver."""
    return cfg["prices"]["drivers"][driver_code]["team"]


def is_sprint_round(cfg: dict, round_number: int) -> bool:
    """Check if a round is a sprint weekend."""
    return round_number in cfg["season"]["sprint_rounds"]


def get_season_phase(round_number: int) -> str:
    """Determine season phase from round number.

    Early season (R1-R6): High uncertainty, teams still adapting.
    Mid season (R7-R16): Performance stabilising, upgrades arriving.
    Late season (R17-R24): Established hierarchy, some teams shifting
                           focus to next year's car.
    """
    if round_number <= 6:
        return "early"
    elif round_number <= 16:
        return "mid"
    else:
        return "late"
