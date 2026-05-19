from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.model.train import CAT_COLS, FEATURE_COLS
from src.optimizer import TeamOptimizer


def _prepare_xy_for_inference(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS + CAT_COLS if c in df.columns]
    X = df[cols].copy()
    for c in [c for c in CAT_COLS if c in X.columns]:
        dummies = pd.get_dummies(X[c], prefix=c, drop_first=True, dtype=float)
        X = pd.concat([X.drop(columns=[c]), dummies], axis=1)
    X = X.astype(float).fillna(0.0)

    for c in feature_names:
        if c not in X.columns:
            X[c] = 0.0
    return X[feature_names]


def _default_round_to_circuit() -> dict[int, str]:
    return {
        1: "albert_park",
        2: "shanghai",
        3: "suzuka",
        4: "bahrain",
        5: "jeddah",
        6: "miami",
        7: "montreal",
        8: "monaco",
        9: "barcelona",
        10: "red_bull_ring",
        11: "silverstone",
        12: "spa",
        13: "hungaroring",
        14: "zandvoort",
        15: "monza",
        16: "valencia",
        17: "baku",
        18: "marina_bay",
        19: "cota",
        20: "mexico_city",
        21: "interlagos",
        22: "las_vegas",
        23: "losail",
        24: "yas_marina",
    }


def _season_phase(round_number: int) -> str:
    if round_number <= 6:
        return "early"
    if round_number <= 16:
        return "mid"
    return "late"


def _build_round_prediction_rows(cfg: dict[str, Any], features: pd.DataFrame, round_number: int) -> pd.DataFrame:
    race = features[features["session_type"] == "race"].copy().sort_values(["year", "round"])
    if race.empty:
        raise ValueError("features.parquet has no race rows")

    latest_driver = race.drop_duplicates(subset=["driver_code"], keep="last").set_index("driver_code")
    latest_ctor = race.drop_duplicates(subset=["constructor_id"], keep="last").set_index("constructor_id")
    global_means = race[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").mean().to_dict()

    circuit_id = _default_round_to_circuit().get(round_number, "albert_park")
    cinfo = cfg.get("circuits", {}).get(circuit_id, {})

    rain_prob = float(cfg.get("weather_override", {}).get("rain_probability", 0.0) or 0.0)
    era_weight = float(cfg.get("regulation", {}).get("era_weight", 0.05))
    sprint_rounds = set(cfg.get("season", {}).get("sprint_rounds", []))

    rows: list[dict[str, Any]] = []
    drivers_cfg = cfg.get("prices", {}).get("drivers", {})
    teams_cfg = cfg.get("teams", {})

    for dcode, dmeta in drivers_cfg.items():
        ctor = str(dmeta.get("team"))
        drow = latest_driver.loc[dcode] if dcode in latest_driver.index else None
        crow = latest_ctor.loc[ctor] if ctor in latest_ctor.index else None

        row: dict[str, Any] = {
            "year": int(cfg.get("season", {}).get("year", 2026)),
            "round": int(round_number),
            "driver_code": str(dcode),
            "constructor_id": ctor,
            "session_type": "race",
            "circuit_id": circuit_id,
        }

        row["circuit_overtake_difficulty"] = float(cinfo.get("overtake_difficulty", global_means.get("circuit_overtake_difficulty", 0.5)))
        row["circuit_drs_zones"] = float(cinfo.get("drs_zones", global_means.get("circuit_drs_zones", 2.0)))
        row["circuit_safety_car_prob"] = float(cinfo.get("safety_car_probability", global_means.get("circuit_safety_car_prob", 0.4)))
        row["is_sprint_round"] = float(1 if round_number in sprint_rounds else 0)
        row["era_weight"] = era_weight
        row["rainfall_flag"] = float(1 if rain_prob >= 0.5 else 0)

        row["circuit_type"] = str(cinfo.get("type", "balanced"))
        row["circuit_downforce"] = str(cinfo.get("downforce", "medium"))
        row["season_phase"] = _season_phase(round_number)

        for f in FEATURE_COLS:
            if f in row:
                continue
            if f.startswith("driver_"):
                row[f] = drow[f] if (drow is not None and f in drow.index) else global_means.get(f, 0.0)
            elif f.startswith("team_"):
                if f == "team_development_score":
                    row[f] = float(teams_cfg.get(ctor, {}).get("development_score", global_means.get(f, 3.0)))
                else:
                    row[f] = crow[f] if (crow is not None and f in crow.index) else global_means.get(f, 0.0)
            else:
                row[f] = drow[f] if (drow is not None and f in drow.index) else global_means.get(f, 0.0)

        rows.append(row)

    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def recommend_round(
    cfg: dict[str, Any],
    round_number: int,
    features_path: str | Path = "data/processed/features.parquet",
    model_path: str | Path = "data/processed/models/fantasy_model.joblib",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    features_path = Path(features_path)
    model_path = Path(model_path)
    if output_path is None:
        output_path = Path(f"data/processed/predictions/2026_round{int(round_number)}_predictions.parquet")
    else:
        output_path = Path(output_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model file: {model_path}")

    features = pd.read_parquet(features_path)
    model_blob = joblib.load(model_path)
    model = model_blob["model"]
    feature_names = model_blob["feature_names"]

    pred_rows = _build_round_prediction_rows(cfg, features, round_number=int(round_number))
    X = _prepare_xy_for_inference(pred_rows, feature_names)
    y_pred = model.predict(X)

    spread = 6.0
    wf_2025 = Path("data/processed/models/backtest/walk_forward_2025_predictions.parquet")
    if wf_2025.exists():
        wf = pd.read_parquet(wf_2025)
        if {"y_pred_q10", "y_pred_q90"}.issubset(wf.columns):
            widths = (wf["y_pred_q90"] - wf["y_pred_q10"]).dropna()
            if not widths.empty:
                spread = float(widths.mean() / 2.0)

    uncertainty_mult = float(cfg.get("regulation", {}).get("uncertainty_multiplier", 1.0))
    spread *= max(0.1, uncertainty_mult)

    out = pred_rows[["year", "round", "driver_code", "constructor_id"]].copy()
    out["y_pred"] = y_pred.astype(float)
    out["y_pred_risk_adj"] = out["y_pred"]
    out["y_pred_q10"] = out["y_pred"] - spread
    out["y_pred_q90"] = out["y_pred"] + spread

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    # Also archive a slim, git-friendly CSV for the visitor "prediction vs actual"
    # view. data/fantasy/predictions/ is committed to the repo.
    slim_dir = Path("data") / "fantasy" / "predictions"
    slim_dir.mkdir(parents=True, exist_ok=True)
    slim_path = slim_dir / f"round_{int(round_number):02d}_predictions.csv"
    slim = out[["year", "round", "driver_code", "constructor_id", "y_pred"]].copy()
    slim = slim.sort_values("y_pred", ascending=False).reset_index(drop=True)
    slim.to_csv(slim_path, index=False)

    opt = TeamOptimizer(cfg)
    # Initial-team computation uses the full fantasy budget cap (£100M by default).
    # Note: cfg.current_team.budget is the *bank remaining*, not the cap — using it
    # here would forbid any team since no single asset fits within bank.
    budget = float(cfg.get("fantasy", {}).get("budget", 100.0))
    season_year = int(cfg.get("season", {}).get("year", 2026))
    rec = opt.recommend_initial_team(
        predictions=out,
        season_year=season_year,
        round_number=int(round_number),
        budget=budget,
    )

    top_drivers = (
        out[["driver_code", "constructor_id", "y_pred"]]
        .sort_values("y_pred", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    return {
        "predictions_path": str(output_path),
        "recommendation": rec.__dict__,
        "top_projected_drivers": top_drivers,
    }


def recommend_transfers(
    cfg: dict[str, Any],
    predictions_path: str | Path,
    season_year: int,
    round_number: int,
) -> dict[str, Any]:
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {pred_path}")

    predictions = pd.read_parquet(pred_path)
    opt = TeamOptimizer(cfg)
    d_prices, c_prices = opt._initial_prices(predictions, season_year)

    out = opt.recommend_transfers(
        predictions=predictions,
        season_year=int(season_year),
        round_number=int(round_number),
        current_team=cfg.get("current_team", {}),
        driver_prices=d_prices,
        constructor_prices=c_prices,
    )

    rec = out.get("recommendation")
    if rec is not None:
        out["recommendation"] = rec.__dict__
    return out
