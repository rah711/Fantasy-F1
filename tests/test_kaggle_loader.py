"""Tests for src.data.kaggle_loader."""

import pytest
import pandas as pd
from pathlib import Path

from src.data.kaggle_loader import (
    _qualifying_time_to_ms,
    _race_status_from_time,
    load_kaggle_sessions,
)


def test_qualifying_time_to_ms():
    assert _qualifying_time_to_ms("1:39.103") == 99_103.0
    assert _qualifying_time_to_ms("1:40.000") == 100_000.0
    assert _qualifying_time_to_ms(None) is None
    assert _qualifying_time_to_ms("") is None


def test_race_status_from_time():
    assert _race_status_from_time("DNF") == "DNF"
    assert _race_status_from_time("1:35:17.520") == "Finished"
    assert _race_status_from_time("+1 lap") == "Finished"
    assert _race_status_from_time(None) == "DNF"


def test_load_kaggle_sessions_empty_dir(tmp_path):
    df = load_kaggle_sessions(tmp_path, year=2099)
    assert df.empty or "year" in df.columns


def test_load_kaggle_sessions_with_data():
    # Use project data/raw/kaggle if present
    kaggle_dir = Path("data/raw/kaggle")
    if not kaggle_dir.exists():
        pytest.skip("data/raw/kaggle not found")
    df = load_kaggle_sessions(kaggle_dir, year=2024)
    if df.empty:
        pytest.skip("No 2024 Kaggle data")
    assert "driver_code" in df.columns
    assert "constructor_id" in df.columns
    assert "circuit_id" in df.columns
    assert "session_type" in df.columns
    assert "year" in df.columns
    assert "round" in df.columns
