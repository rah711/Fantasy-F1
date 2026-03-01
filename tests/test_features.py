"""Tests for feature engineering."""

import pytest
import pandas as pd

from src.config import load_config
from src.features.track_features import add_track_features
from src.features.contextual_features import add_contextual_features
from src.features.builder import build_features, FEATURE_COLUMNS


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def minimal_sessions():
    return pd.DataFrame([
        {"year": 2024, "round": 1, "circuit_id": "bahrain", "session_type": "race", "driver_code": "VER", "constructor_id": "red_bull"},
        {"year": 2024, "round": 1, "circuit_id": "bahrain", "session_type": "race", "driver_code": "PER", "constructor_id": "red_bull"},
    ])


def test_add_track_features(cfg, minimal_sessions):
    out = add_track_features(minimal_sessions, cfg)
    assert "circuit_type" in out.columns
    assert "circuit_overtake_difficulty" in out.columns
    assert "circuit_safety_car_prob" in out.columns
    assert "is_sprint_round" in out.columns
    assert out["circuit_id"].iloc[0] == "bahrain"


def test_add_contextual_features(cfg, minimal_sessions):
    out = add_contextual_features(minimal_sessions, cfg)
    assert "season_phase" in out.columns
    assert "era_weight" in out.columns
    assert out["season_phase"].iloc[0] == "early"


def test_build_features_empty(cfg):
    out = build_features(pd.DataFrame(), config=cfg)
    assert out.empty


def test_build_features_minimal(cfg, minimal_sessions):
    minimal_sessions["fantasy_points_driver"] = 10
    minimal_sessions["finish_position"] = 1
    minimal_sessions["status"] = "Finished"
    minimal_sessions["overtakes"] = 0
    minimal_sessions["fastest_pitstop_ms"] = pd.NA
    minimal_sessions["avg_pitstop_ms"] = pd.NA
    out = build_features(minimal_sessions, config=cfg)
    assert len(out) == 2
    for col in ["circuit_type", "season_phase", "era_weight",
                "driver_is_cold_start", "team_prev_season_avg_pts",
                "team_pts_at_circuit_type_hist"]:
        assert col in out.columns
