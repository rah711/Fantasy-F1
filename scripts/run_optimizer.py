#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.optimizer import TeamOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fantasy F1 team optimizer")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to walk-forward predictions parquet")
    parser.add_argument("--season-year", type=int, required=True, help="Season year to optimize/backtest")
    parser.add_argument("--round", type=int, default=1, help="Target round")
    parser.add_argument(
        "--mode",
        choices=["initial", "transfers", "backtest"],
        default="initial",
        help="Optimizer mode",
    )
    args = parser.parse_args()

    cfg = load_config(str(args.config)) if args.config else load_config()
    preds = pd.read_parquet(args.predictions)

    opt = TeamOptimizer(cfg)

    if args.mode == "initial":
        rec = opt.recommend_initial_team(
            predictions=preds,
            season_year=args.season_year,
            round_number=args.round,
            budget=float(cfg.get("current_team", {}).get("budget", cfg["fantasy"]["budget"])),
        )
        print(json.dumps(rec.__dict__, indent=2))
        return

    if args.mode == "transfers":
        d_prices, c_prices = opt._initial_prices(preds, args.season_year)
        out = opt.recommend_transfers(
            predictions=preds,
            season_year=args.season_year,
            round_number=args.round,
            current_team=cfg.get("current_team", {}),
            driver_prices=d_prices,
            constructor_prices=c_prices,
        )
        out["recommendation"] = out["recommendation"].__dict__
        print(json.dumps(out, indent=2))
        return

    backtest = opt.backtest_season(
        predictions=preds,
        season_year=args.season_year,
        current_team=cfg.get("current_team", {}),
    )
    print(json.dumps(backtest, indent=2))


if __name__ == "__main__":
    main()
