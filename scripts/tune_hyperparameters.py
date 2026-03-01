#!/usr/bin/env python3
"""
Hyperparameter tuning via Optuna with walk-forward cross-validation.

Minimises average walk-forward MAE across 2023-2025 test years.
Each trial proposes LightGBM hyperparameters, runs a lightweight
walk-forward for each validation year, and reports the mean MAE.

Usage:
    PYTHONPATH=. python3 scripts/tune_hyperparameters.py
    PYTHONPATH=. python3 scripts/tune_hyperparameters.py --n-trials 30
    PYTHONPATH=. python3 scripts/tune_hyperparameters.py --years 2024 2025
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import optuna
import yaml

from src.backtest.walk_forward import walk_forward_backtest
from src.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
CONFIG_PATH = Path("config.yaml")


def _objective(trial: optuna.Trial, config: dict, years: list[int]) -> float:
    """Optuna objective: mean walk-forward MAE across validation years."""

    hp = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
    }

    cfg = copy.deepcopy(config)
    engine = cfg.get("model", {}).get("engine", "lightgbm")
    cfg["model"]["hyperparameters"][engine] = hp

    maes = []
    for year in years:
        try:
            result = walk_forward_backtest(
                features_path=FEATURES_PATH,
                config=cfg,
                test_year=year,
                train_start_year=2018,
                output_dir=Path("data/processed/models/tuning"),
            )
            mae = result["overall"]["mae"]
            maes.append(mae)
            trial.report(mae, step=year)
            if trial.should_prune():
                raise optuna.TrialPruned()
        except Exception as e:
            log.warning("Trial %d failed on year %d: %s", trial.number, year, e)
            maes.append(float("inf"))

    mean_mae = sum(maes) / len(maes) if maes else float("inf")
    return mean_mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune model hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds")
    args = parser.parse_args()

    config = load_config()

    study = optuna.create_study(
        direction="minimize",
        study_name="fantasy_f1_hp_tuning",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    study.optimize(
        lambda trial: _objective(trial, config, args.years),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_params
    print("\n=== Best Hyperparameters ===")
    print(json.dumps(best, indent=2))
    print(f"Best mean walk-forward MAE: {study.best_value:.4f}")

    # Save best params to a JSON file for reference
    output = {
        "best_params": best,
        "best_mae": study.best_value,
        "n_trials": len(study.trials),
        "validation_years": args.years,
    }
    out_path = Path("data/processed/models/tuning/best_hyperparameters.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to {out_path}")

    # Update config.yaml with the best params
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    engine = raw.get("model", {}).get("engine", "lightgbm")
    if "model" not in raw:
        raw["model"] = {}
    if "hyperparameters" not in raw["model"]:
        raw["model"]["hyperparameters"] = {}
    if engine not in raw["model"]["hyperparameters"]:
        raw["model"]["hyperparameters"][engine] = {}

    for k, v in best.items():
        if isinstance(v, float):
            v = round(v, 6)
        raw["model"]["hyperparameters"][engine][k] = v

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Updated {CONFIG_PATH} with best {engine} hyperparameters")


if __name__ == "__main__":
    main()
