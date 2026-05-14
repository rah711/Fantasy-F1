from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.optimizer import TeamOptimizer
from src.price import PriceModel


def test_price_model_round1_driver_rule() -> None:
    cfg = load_config()
    model = PriceModel(cfg)

    upd = model.apply_update(
        entity_type="driver",
        asset_id="VER",
        current_price=27.7,
        predicted_points=35.0,
        round_number=1,
    )

    assert upd.band == "great"
    assert abs(upd.delta - 0.3) < 1e-9
    assert abs(upd.new_price - 28.0) < 1e-9


def test_price_model_floor() -> None:
    cfg = load_config()
    model = PriceModel(cfg)

    upd = model.apply_update(
        entity_type="driver",
        asset_id="PER",
        current_price=3.2,
        predicted_points=-20.0,
        round_number=2,
    )

    assert upd.new_price == 3.0
    assert abs(upd.delta + 0.2) < 1e-9


def test_optimizer_initial_team_respects_constraints() -> None:
    cfg = load_config()

    # Tiny synthetic season with the 2026 asset universe from config.
    rows = []
    for rnd in [1, 2, 3]:
        for i, (drv, dmeta) in enumerate(cfg["prices"]["drivers"].items()):
            rows.append(
                {
                    "year": 2026,
                    "round": rnd,
                    "driver_code": drv,
                    "constructor_id": dmeta["team"],
                    "y_pred": float(30 - i),
                    "y_pred_risk_adj": float(30 - i),
                    "y_true": float(30 - i),
                }
            )
    preds = pd.DataFrame(rows)

    opt = TeamOptimizer(cfg)
    rec = opt.recommend_initial_team(predictions=preds, season_year=2026, round_number=1)

    assert len(rec.drivers) == cfg["fantasy"]["num_drivers"]
    assert len(rec.constructors) == cfg["fantasy"]["num_constructors"]
    assert rec.total_cost <= cfg["fantasy"]["budget"] + 1e-9
    assert rec.drs_boost in rec.drivers


def test_historical_prices_are_reinferred_from_season_context() -> None:
    cfg = load_config()
    cfg = dict(cfg)
    cfg.pop("historical_prices", None)
    model = PriceModel(cfg)

    preds = pd.DataFrame(
        [
            {"year": 2025, "round": 1, "driver_code": "SAI", "constructor_id": "ferrari", "y_pred": 30.0},
            {"year": 2025, "round": 1, "driver_code": "HAM", "constructor_id": "mclaren", "y_pred": 20.0},
            {"year": 2025, "round": 1, "driver_code": "VER", "constructor_id": "williams", "y_pred": 8.0},
        ]
    )

    prices = model.infer_opening_prices(predictions=preds, entity_type="driver", season_year=2025)

    assert set(prices.keys()) == {"SAI", "HAM", "VER"}
    assert prices["SAI"] > prices["HAM"] > prices["VER"]
    assert prices["SAI"] != float(cfg["prices"]["drivers"]["SAI"]["price"])


def test_config_season_keeps_official_prices() -> None:
    cfg = load_config()
    model = PriceModel(cfg)

    preds = pd.DataFrame(
        [
            {"year": 2026, "round": 1, "driver_code": "VER", "constructor_id": "williams", "y_pred": 1.0},
            {"year": 2026, "round": 1, "driver_code": "BOT", "constructor_id": "red_bull", "y_pred": 99.0},
        ]
    )

    prices = model.infer_opening_prices(predictions=preds, entity_type="driver", season_year=2026)

    assert len(prices) == len(cfg["prices"]["drivers"])
    assert prices["VER"] == float(cfg["prices"]["drivers"]["VER"]["price"])
    assert prices["BOT"] == float(cfg["prices"]["drivers"]["BOT"]["price"])


def test_historical_overrides_take_priority() -> None:
    cfg = load_config()
    cfg = dict(cfg)
    cfg["historical_prices"] = {
        "2025": {
            "drivers": {
                "VER": 31.1,
                "SAI": 12.5,
            },
            "constructors": {
                "mclaren": 33.3,
            },
        }
    }
    model = PriceModel(cfg)

    preds = pd.DataFrame(
        [
            {"year": 2025, "round": 1, "driver_code": "VER", "constructor_id": "red_bull", "y_pred": 1.0},
            {"year": 2025, "round": 1, "driver_code": "SAI", "constructor_id": "williams", "y_pred": 99.0},
            {"year": 2025, "round": 1, "driver_code": "HAM", "constructor_id": "ferrari", "y_pred": 50.0},
        ]
    )

    d_prices = model.infer_opening_prices(predictions=preds, entity_type="driver", season_year=2025)
    c_prices = model.infer_opening_prices(predictions=preds, entity_type="constructor", season_year=2025)

    assert d_prices["VER"] == 31.1
    assert d_prices["SAI"] == 12.5
    assert "HAM" in d_prices
    assert c_prices["mclaren"] == 33.3


def test_seed_missing_prices_for_midseason_entry() -> None:
    cfg = load_config()
    cfg = dict(cfg)
    cfg["historical_prices"] = {
        "2025": {
            "drivers": {"VER": 28.4, "NOR": 29.0},
            "constructors": {"red_bull": 25.2, "mclaren": 30.0},
        }
    }
    model = PriceModel(cfg)
    existing = model.infer_opening_prices(
        predictions=pd.DataFrame(
            [
                {"year": 2025, "round": 1, "driver_code": "VER", "constructor_id": "red_bull", "y_pred": 20.0},
                {"year": 2025, "round": 1, "driver_code": "NOR", "constructor_id": "mclaren", "y_pred": 21.0},
            ]
        ),
        entity_type="driver",
        season_year=2025,
    )
    assert "COL" not in existing

    preds = pd.DataFrame(
        [
            {"year": 2025, "round": 7, "driver_code": "VER", "constructor_id": "red_bull", "y_pred": 18.0},
            {"year": 2025, "round": 7, "driver_code": "NOR", "constructor_id": "mclaren", "y_pred": 19.0},
            {"year": 2025, "round": 7, "driver_code": "COL", "constructor_id": "williams", "y_pred": 12.0},
        ]
    )
    updated, seeded = model.seed_missing_prices_for_round(
        predictions=preds,
        entity_type="driver",
        season_year=2025,
        round_number=7,
        existing_prices=existing,
        score_col="y_pred",
    )
    assert "COL" in seeded
    assert updated["COL"] >= 3.0


def test_initial_team_kpi_uses_budget_constraint() -> None:
    cfg = load_config()
    cfg = dict(cfg)
    cfg.pop("historical_prices", None)

    rows = []
    # 8 drivers across 4 constructors, 3 rounds.
    drivers = [
        ("D1", "t1", 20.0),
        ("D2", "t1", 18.0),
        ("D3", "t2", 16.0),
        ("D4", "t2", 14.0),
        ("D5", "t3", 12.0),
        ("D6", "t3", 10.0),
        ("D7", "t4", 8.0),
        ("D8", "t4", 6.0),
    ]
    for rnd in [1, 2, 3]:
        for code, ctor, base in drivers:
            rows.append(
                {
                    "year": 2030,
                    "round": rnd,
                    "driver_code": code,
                    "constructor_id": ctor,
                    "y_pred": base + rnd,
                    "y_pred_risk_adj": base + rnd,
                    "y_true": base + rnd,
                    "is_dnf": 0,
                    "dnf_prob": 0.0,
                }
            )
    preds = pd.DataFrame(rows)

    opt = TeamOptimizer(cfg)
    kpi = opt.evaluate_initial_team_kpi(predictions=preds, season_year=2030, num_alternatives=20)

    assert kpi["starting_budget_constraint"] == float(cfg["fantasy"]["budget"])
    assert kpi["chosen_team"]["cost"] <= float(cfg["fantasy"]["budget"]) + 1e-9
    assert 0.0 <= kpi["percentile_vs_feasible_alternatives"] <= 1.0
