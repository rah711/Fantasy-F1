from __future__ import annotations

from src.config import load_config
from src.ui_services.config_service import generate_config_diff, validate_and_apply_price_csv


def test_generate_config_diff_shows_change() -> None:
    cfg = load_config()
    new_cfg = load_config()
    new_cfg["weather_override"]["rain_probability"] = 0.33
    diff = generate_config_diff(cfg, new_cfg)
    assert "rain_probability" in diff
    assert "+  rain_probability: 0.33" in diff or "+  rain_probability: 0.330" in diff


def test_validate_and_apply_driver_price_csv_success() -> None:
    cfg = load_config()
    csv_text = "driver_code,price\nVER,27.9\nNOR,27.5\n"
    res = validate_and_apply_price_csv(cfg, csv_text, "driver")
    assert res.ok
    assert res.updated_config is not None
    assert res.updated_rows == 2
    assert abs(res.updated_config["prices"]["drivers"]["VER"]["price"] - 27.9) < 1e-9


def test_validate_and_apply_constructor_price_csv_unknown_fails() -> None:
    cfg = load_config()
    csv_text = "constructor_id,price\nunknown_team,9.9\n"
    res = validate_and_apply_price_csv(cfg, csv_text, "constructor")
    assert not res.ok
    assert any("unknown constructor" in e for e in res.errors)
