"""Tests for src.data.schema."""

import pytest

from src.data.schema import (
    normalise_driver,
    normalise_circuit,
    normalise_constructor,
    STATUS_CLASSIFICATION,
    UNIFIED_COLUMNS,
)


def test_normalise_driver():
    assert normalise_driver("Max Verstappen") == "VER"
    assert normalise_driver("verstappen") == "VER"
    assert normalise_driver("Lewis Hamilton") == "HAM"
    assert normalise_driver("HAM") == "HAM"


def test_normalise_driver_unknown_raises():
    with pytest.raises(KeyError):
        normalise_driver("Unknown Driver XYZ")


def test_normalise_circuit():
    assert normalise_circuit("Albert Park") == "albert_park"
    assert normalise_circuit("Melbourne") == "albert_park"
    assert normalise_circuit("Bahrain") == "bahrain"
    assert normalise_circuit("Monaco") == "monaco"


def test_normalise_constructor():
    assert normalise_constructor("Red Bull Racing") == "red_bull"
    assert normalise_constructor("Mercedes") == "mercedes"
    assert normalise_constructor("McLaren") == "mclaren"


def test_status_classification_has_required():
    assert "Finished" in STATUS_CLASSIFICATION.values()
    assert "DNF" in STATUS_CLASSIFICATION.values()
    assert "DSQ" in STATUS_CLASSIFICATION.values()


def test_unified_columns_has_core():
    assert "year" in UNIFIED_COLUMNS
    assert "round" in UNIFIED_COLUMNS
    assert "circuit_id" in UNIFIED_COLUMNS
    assert "session_type" in UNIFIED_COLUMNS
    assert "driver_code" in UNIFIED_COLUMNS
    assert "constructor_id" in UNIFIED_COLUMNS
    assert "finish_position" in UNIFIED_COLUMNS
    assert "status" in UNIFIED_COLUMNS
