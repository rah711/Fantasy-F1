#!/usr/bin/env python3
"""
Train the fantasy points model and run optional backtests.

Usage:
    PYTHONPATH=. python scripts/train_model.py
    PYTHONPATH=. python scripts/train_model.py --test-year 2025 --walk-forward
    PYTHONPATH=. python scripts/train_model.py --no-walk-forward
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from src.backtest.walk_forward import walk_forward_backtest
from src.config import load_config
from src.model.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fantasy points model")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"), help="Path to features.parquet")
    parser.add_argument("--test-year", type=int, default=None, help="Holdout/backtest year (default: config backtest.primary_test_year)")
    parser.add_argument("--train-start", type=int, default=None, help="First year for training (default: config backtest.train_start_year)")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory to save model (default: data/processed/models)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--walk-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run proper round-by-round walk-forward backtest (default: true)",
    )
    args = parser.parse_args()

    config = load_config(str(args.config)) if args.config else load_config()

    # 1) Train and save the production model (temporal holdout summary retained)
    result = train_model(
        features_path=args.features,
        config=config,
        test_year=args.test_year,
        train_start_year=args.train_start,
        model_dir=args.model_dir,
    )
    print("Train MAE:", result["train_metrics"]["mae"], "RMSE:", result["train_metrics"]["rmse"])
    if result["test_metrics"]:
        print(
            "Holdout Test MAE:",
            result["test_metrics"]["mae"],
            "RMSE:",
            result["test_metrics"]["rmse"],
            f"(year={result['test_year']})",
        )
    else:
        print("No holdout test rows for selected year.")

    # 2) Proper walk-forward backtest (optional, default on)
    if args.walk_forward:
        backtest_cfg = config.get("backtest", {})
        test_year = args.test_year or backtest_cfg.get("primary_test_year", 2025)
        train_start = args.train_start or backtest_cfg.get("train_start_year", 2020)
        model_dir = Path(args.model_dir or backtest_cfg.get("model_dir") or "data/processed/models")
        wf_output = model_dir / "backtest"

        wf = walk_forward_backtest(
            features_path=args.features,
            config=config,
            test_year=int(test_year),
            train_start_year=int(train_start),
            output_dir=wf_output,
        )
        print(
            "Walk-forward MAE:",
            wf["overall"]["mae"],
            "RMSE:",
            wf["overall"]["rmse"],
            f"(year={wf['test_year']})",
        )
        if "benchmark_overall" in wf:
            print("Walk-forward benchmarks (MAE):")
            for name, m in wf["benchmark_overall"].items():
                print(f"  {name}: {m['mae']:.4f} (RMSE={m['rmse']:.4f})")
        print("Walk-forward predictions:", wf["predictions_path"])

        # Snapshot baseline metrics for reproducible comparisons over time.
        snapshot_dir = model_dir / "backtest"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot = {
            "created_at_utc": ts,
            "test_year": int(test_year),
            "train_start_year": int(train_start),
            "holdout": {
                "mae": result["test_metrics"].get("mae") if result["test_metrics"] else None,
                "rmse": result["test_metrics"].get("rmse") if result["test_metrics"] else None,
            },
            "walk_forward": wf.get("overall", {}),
            "walk_forward_benchmarks": wf.get("benchmark_overall", {}),
            "walk_forward_predictions_path": wf.get("predictions_path"),
        }
        latest_path = snapshot_dir / f"baseline_{int(test_year)}_latest.json"
        timestamped_path = snapshot_dir / f"baseline_{int(test_year)}_{ts}.json"
        latest_path.write_text(json.dumps(snapshot, indent=2))
        timestamped_path.write_text(json.dumps(snapshot, indent=2))
        print("Baseline snapshot saved:", latest_path)
        print("Baseline snapshot archived:", timestamped_path)


if __name__ == "__main__":
    main()
