"""Tests for walk-forward backtesting."""

from pathlib import Path

import pandas as pd

from src.backtest.walk_forward import walk_forward_backtest


def _make_row(year, round_num, driver, team, pts):
    return {
        "year": year,
        "round": round_num,
        "session_type": "race",
        "driver_code": driver,
        "constructor_id": team,
        "fantasy_points_driver": float(pts),
        "status": "Finished",
        "circuit_overtake_difficulty": 0.4,
        "circuit_drs_zones": 2,
        "circuit_safety_car_prob": 0.4,
        "is_sprint_round": False,
        "era_weight": 0.2,
        "rainfall_flag": False,
        "driver_rolling_pts_3": 10.0,
        "driver_rolling_pts_5": 20.0,
        "driver_avg_finish_at_circuit": 7.0,
        "driver_overtake_rate": 2.0,
        "driver_dnf_rate": 0.1,
        "driver_prev_season_avg_pts": 12.0,
        "driver_prev_season_avg_finish": 8.0,
        "driver_prev_season_dnf_rate": 0.05,
        "driver_prev_season_overtake_rate": 1.5,
        "driver_prev_season_races": 20.0,
        "driver_is_cold_start": 0,
        "driver_early_round_flag": 1 if round_num <= 3 else 0,
        "driver_cold_start_pressure": 0,
        "team_development_score": 3.0,
        "team_fastest_pitstop_avg": 2500.0,
        "team_avg_pitstop_avg": 2800.0,
        "team_prev_season_avg_pts": 11.0,
        "team_prev_season_avg_finish": 9.0,
        "team_prev_season_dnf_rate": 0.08,
        "team_prev_season_races": 40.0,
        "team_pts_at_circuit_type_hist": 10.0,
        "team_pts_at_circuit_type_season": 10.0,
        "team_pts_at_downforce_hist": 10.0,
        "team_pts_at_downforce_season": 10.0,
        "team_circuit_type_delta": 0.0,
        "team_season_rounds_so_far": float(round_num - 1),
        "driver_pts_at_circuit_type_hist": 12.0,
        "driver_pts_at_circuit_type_season": 12.0,
        "driver_pts_at_downforce_hist": 12.0,
        "driver_pts_at_downforce_season": 12.0,
        "circuit_type": "balanced",
        "circuit_downforce": "medium",
        "season_phase": "early",
    }


def test_walk_forward_backtest(tmp_path):
    rows = []
    # train year
    for rnd in [1, 2, 3]:
        rows.append(_make_row(2020, rnd, "VER", "red_bull", 20 + rnd))
        rows.append(_make_row(2020, rnd, "NOR", "mclaren", 15 + rnd))
    # test year with two rounds (walk-forward should predict both)
    for rnd in [1, 2]:
        rows.append(_make_row(2021, rnd, "VER", "red_bull", 21 + rnd))
        rows.append(_make_row(2021, rnd, "NOR", "mclaren", 16 + rnd))

    df = pd.DataFrame(rows)
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path, index=False)

    cfg = {
        "model": {
            "engine": "lightgbm",
            "hyperparameters": {
                "lightgbm": {"n_estimators": 25, "learning_rate": 0.1, "max_depth": 4}
            },
        }
    }

    result = walk_forward_backtest(
        features_path=features_path,
        config=cfg,
        test_year=2021,
        train_start_year=2020,
        output_dir=tmp_path,
    )

    assert result["test_year"] == 2021
    assert result["overall"]["n"] > 0
    assert result["overall"]["mae"] >= 0
    assert result["overall"]["rmse"] >= 0
    assert len(result["per_round"]) >= 1
    assert Path(result["predictions_path"]).exists()
    assert "overall_risk_adjusted" in result
    assert "dnf_risk" in result
    assert "uncertainty" in result
