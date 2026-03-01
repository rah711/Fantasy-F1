"""Tests for src.data.scoring."""

import pytest
import pandas as pd

from src.config import load_config
from src.data.scoring import (
    score_driver_qualifying,
    score_driver_sprint,
    score_driver_race,
    score_constructor_qualifying,
    score_constructor_sprint,
    score_constructor_race,
    compute_fantasy_points,
)
from src.data.schema import STATUS_CLASSIFICATION


@pytest.fixture
def cfg():
    return load_config()


def test_status_classification_exists():
    assert "Finished" in STATUS_CLASSIFICATION.values()
    assert "DNF" in STATUS_CLASSIFICATION.values()
    assert "DSQ" in STATUS_CLASSIFICATION.values()
    assert STATUS_CLASSIFICATION.get("1") == "Finished"
    assert STATUS_CLASSIFICATION.get("FINISHED") == "Finished"
    assert STATUS_CLASSIFICATION.get("DNF") == "DNF" or "DNF" in str(STATUS_CLASSIFICATION.values())


def test_driver_qualifying_p1(cfg):
    assert score_driver_qualifying(cfg, 1, "Finished", 99e3, 99e3) == 10


def test_driver_qualifying_p10(cfg):
    assert score_driver_qualifying(cfg, 10, "Finished", 99e3, None) == 1


def test_driver_qualifying_dnf_penalty(cfg):
    assert score_driver_qualifying(cfg, 5, "DNF", None, None) == -5


def test_driver_qualifying_dsq_penalty(cfg):
    assert score_driver_qualifying(cfg, 3, "DSQ", 99e3, 99e3) == -5


def test_driver_race_p1(cfg):
    assert score_driver_race(cfg, 1, "Finished", 0, 0, False, False) == 25


def test_driver_race_fastest_lap(cfg):
    assert score_driver_race(cfg, 5, "Finished", 0, 0, True, False) == 10 + 10  # P5 + FL


def test_driver_race_dotd(cfg):
    assert score_driver_race(cfg, 3, "Finished", 0, 0, False, True) == 15 + 10  # P3 + DOTD


def test_driver_race_dnf_penalty(cfg):
    assert score_driver_race(cfg, 0, "DNF", 0, 0, False, False) == -20


def test_compute_fantasy_points_empty(cfg):
    df = pd.DataFrame()
    out = compute_fantasy_points(cfg, df)
    assert out.empty
    assert "fantasy_points_driver" in out.columns or out.columns.empty
    assert "constructor_bonus" in out.columns or out.columns.empty


def test_compute_fantasy_points_qualifying_row(cfg):
    df = pd.DataFrame([{
        "year": 2024,
        "round": 1,
        "circuit_id": "bahrain",
        "session_type": "qualifying",
        "driver_code": "VER",
        "constructor_id": "red_bull",
        "grid_position": 1,
        "finish_position": 1,
        "status": "Finished",
        "q1_time_ms": 99e3,
        "q2_time_ms": 99e3,
        "q3_time_ms": 99e3,
        "positions_gained": None,
        "overtakes": None,
        "is_fastest_lap": False,
        "is_dotd": False,
    }])
    out = compute_fantasy_points(cfg, df)
    assert len(out) == 1
    assert out["fantasy_points_driver"].iloc[0] == 10
