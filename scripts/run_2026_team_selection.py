#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
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


def build_round_prediction_rows(cfg: dict[str, Any], features: pd.DataFrame, round_number: int) -> pd.DataFrame:
    race = features[features["session_type"] == "race"].copy()
    race = race.sort_values(["year", "round"])

    if race.empty:
        raise ValueError("features.parquet has no race rows")

    latest_driver = race.drop_duplicates(subset=["driver_code"], keep="last").set_index("driver_code")
    latest_ctor = race.drop_duplicates(subset=["constructor_id"], keep="last").set_index("constructor_id")

    global_means = race[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").mean().to_dict()

    r2c = _default_round_to_circuit()
    circuit_id = r2c.get(round_number, "albert_park")
    cinfo = cfg.get("circuits", {}).get(circuit_id, {})

    rain_prob = float(cfg.get("weather_override", {}).get("rain_probability", 0.0) or 0.0)
    era_weight = float(cfg.get("regulation", {}).get("era_weight", 0.05))
    sprint_rounds = set(cfg.get("season", {}).get("sprint_rounds", []))

    rows = []
    drivers_cfg = cfg.get("prices", {}).get("drivers", {})
    teams_cfg = cfg.get("teams", {})

    for dcode, dmeta in drivers_cfg.items():
        ctor = str(dmeta.get("team"))
        drow = latest_driver.loc[dcode] if dcode in latest_driver.index else None
        crow = latest_ctor.loc[ctor] if ctor in latest_ctor.index else None

        row: dict[str, Any] = {
            "year": 2026,
            "round": int(round_number),
            "driver_code": str(dcode),
            "constructor_id": ctor,
            "session_type": "race",
            "circuit_id": circuit_id,
        }

        # Track/context
        row["circuit_overtake_difficulty"] = float(cinfo.get("overtake_difficulty", global_means.get("circuit_overtake_difficulty", 0.5)))
        row["circuit_drs_zones"] = float(cinfo.get("drs_zones", global_means.get("circuit_drs_zones", 2)))
        row["circuit_safety_car_prob"] = float(cinfo.get("safety_car_probability", global_means.get("circuit_safety_car_prob", 0.4)))
        row["is_sprint_round"] = float(1 if round_number in sprint_rounds else 0)
        row["era_weight"] = era_weight
        row["rainfall_flag"] = float(1 if rain_prob >= 0.5 else 0)

        # Categorical model features
        row["circuit_type"] = str(cinfo.get("type", "balanced"))
        row["circuit_downforce"] = str(cinfo.get("downforce", "medium"))
        row["season_phase"] = _season_phase(round_number)

        # Driver/team feature carry-forward with config overrides where relevant
        for f in FEATURE_COLS:
            if f in row:
                continue
            if f.startswith("driver_"):
                if drow is not None and f in drow.index:
                    row[f] = drow[f]
                else:
                    row[f] = global_means.get(f, 0.0)
            elif f.startswith("team_"):
                if f == "team_development_score":
                    row[f] = float(teams_cfg.get(ctor, {}).get("development_score", global_means.get(f, 3.0)))
                elif crow is not None and f in crow.index:
                    row[f] = crow[f]
                else:
                    row[f] = global_means.get(f, 0.0)
            else:
                if drow is not None and f in drow.index:
                    row[f] = drow[f]
                else:
                    row[f] = global_means.get(f, 0.0)

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.replace([np.inf, -np.inf], np.nan)
    return pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2026 R1 predictions parquet and recommend starting team")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--model", type=Path, default=Path("data/processed/models/fantasy_model.joblib"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/predictions/2026_round1_predictions.parquet"),
        help="Where to save generated predictions parquet",
    )
    parser.add_argument("--round", type=int, default=1, help="Target round number (default: 1)")
    args = parser.parse_args()

    cfg = load_config(str(args.config)) if args.config else load_config()

    if not args.features.exists():
        raise FileNotFoundError(f"Missing features file: {args.features}")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing trained model file: {args.model}")

    features = pd.read_parquet(args.features)
    model_blob = joblib.load(args.model)
    model = model_blob["model"]
    feature_names = model_blob["feature_names"]

    pred_rows = build_round_prediction_rows(cfg, features, round_number=args.round)
    X = _prepare_xy_for_inference(pred_rows, feature_names)
    y_pred = model.predict(X)

    # Simple uncertainty proxy from historical walk-forward quantile widths.
    spread = 6.0
    wf_2025 = Path("data/processed/models/backtest/walk_forward_2025_predictions.parquet")
    if wf_2025.exists():
        wf = pd.read_parquet(wf_2025)
        if {"y_pred_q10", "y_pred_q90"}.issubset(wf.columns):
            w = (wf["y_pred_q90"] - wf["y_pred_q10"]).dropna()
            if not w.empty:
                spread = float(w.mean() / 2.0)

    out = pred_rows[["year", "round", "driver_code", "constructor_id"]].copy()
    out["y_pred"] = y_pred.astype(float)
    out["y_pred_risk_adj"] = out["y_pred"]
    out["y_pred_q10"] = out["y_pred"] - spread
    out["y_pred_q90"] = out["y_pred"] + spread

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    opt = TeamOptimizer(cfg)
    rec = opt.recommend_initial_team(
        predictions=out,
        season_year=2026,
        round_number=int(args.round),
        budget=float(cfg.get("fantasy", {}).get("budget", 100.0)),
    )

    print("Saved predictions parquet:")
    print(f"  {args.output}")
    print("\nStarting team recommendation:")
    print(json.dumps(rec.__dict__, indent=2))
    print("\nTop projected drivers:")
    print(
        out[["driver_code", "constructor_id", "y_pred"]]
        .sort_values("y_pred", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
