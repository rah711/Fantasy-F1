#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.optimizer import TeamOptimizer


def _parse_csv_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _build_optimizer_cfg(base_cfg: dict, lookahead: int, weight: float, risk: str, kpi_num_alts: int, seed: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("optimizer", {})
    cfg["optimizer"]["lookahead_races"] = int(lookahead)
    cfg["optimizer"]["price_appreciation_weight"] = float(weight)
    cfg["optimizer"]["risk_tolerance"] = str(risk)
    cfg["optimizer"].setdefault("initial_team_kpi", {})
    cfg["optimizer"]["initial_team_kpi"]["enabled"] = True
    cfg["optimizer"]["initial_team_kpi"]["num_alternatives"] = int(kpi_num_alts)
    cfg["optimizer"]["initial_team_kpi"]["random_seed"] = int(seed)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict-split optimizer tuning (train 2023-2024, test 2025)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/processed/models/backtest"),
        help="Directory containing walk_forward_<year>_predictions.parquet",
    )
    parser.add_argument("--train-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--lookaheads", type=str, default="2,3,4")
    parser.add_argument("--price-weights", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--risk-levels", type=str, default="conservative,moderate,aggressive")
    parser.add_argument("--kpi-num-alternatives", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-combos", type=int, default=None, help="Limit evaluated combinations for faster runs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/optimizer_tuning"),
        help="Output directory",
    )
    args = parser.parse_args()

    cfg = load_config(str(args.config)) if args.config else load_config()

    lookaheads = _parse_csv_list(args.lookaheads, int)
    weights = _parse_csv_list(args.price_weights, float)
    risks = _parse_csv_list(args.risk_levels, str)

    combos = list(product(lookaheads, weights, risks))
    if args.max_combos is not None:
        combos = combos[: max(0, int(args.max_combos))]

    years_needed = sorted(set(args.train_years + [args.test_year]))
    preds_by_year = {}
    for y in years_needed:
        p = args.predictions_dir / f"walk_forward_{y}_predictions.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing predictions for {y}: {p}")
        preds_by_year[y] = pd.read_parquet(p)

    rows = []
    total = len(combos)
    for i, (lookahead, weight, risk) in enumerate(combos, start=1):
        print(f"[{i}/{total}] lookahead={lookahead} weight={weight} risk={risk}")
        combo_cfg = _build_optimizer_cfg(
            base_cfg=cfg,
            lookahead=lookahead,
            weight=weight,
            risk=risk,
            kpi_num_alts=args.kpi_num_alternatives,
            seed=args.random_seed,
        )
        opt = TeamOptimizer(combo_cfg)

        train_points = []
        train_budgets = []
        for y in args.train_years:
            sim = opt._simulate_strategy(
                predictions=preds_by_year[y],
                season_year=y,
                current_team={},
                optimize_transfers=True,
                lookahead=opt.lookahead,
                price_weight=opt.price_weight,
            )
            train_points.append(float(sim["total_points"]))
            train_budgets.append(float(sim["ending_budget"]))

        rows.append(
            {
                "lookahead_races": int(lookahead),
                "price_appreciation_weight": float(weight),
                "risk_tolerance": str(risk),
                "train_points_mean": float(sum(train_points) / len(train_points)),
                "train_ending_budget_mean": float(sum(train_budgets) / len(train_budgets)),
            }
        )

    if not rows:
        raise ValueError("No combinations evaluated")

    train_df = pd.DataFrame(rows).sort_values(
        by=["train_points_mean", "train_ending_budget_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best = train_df.iloc[0].to_dict()
    print("Best train combo:", best)

    best_cfg = _build_optimizer_cfg(
        base_cfg=cfg,
        lookahead=int(best["lookahead_races"]),
        weight=float(best["price_appreciation_weight"]),
        risk=str(best["risk_tolerance"]),
        kpi_num_alts=args.kpi_num_alternatives,
        seed=args.random_seed,
    )
    best_opt = TeamOptimizer(best_cfg)

    # Strict split: test year only after best train combo is selected.
    test_backtest = best_opt.backtest_season(
        predictions=preds_by_year[args.test_year],
        season_year=args.test_year,
        current_team={},
    )

    summary = {
        "split": {
            "train_years": args.train_years,
            "test_year": int(args.test_year),
        },
        "grid": {
            "lookaheads": lookaheads,
            "price_weights": weights,
            "risk_levels": risks,
            "num_combos_evaluated": int(len(train_df)),
        },
        "kpi_num_alternatives": int(args.kpi_num_alternatives),
        "random_seed": int(args.random_seed),
        "best_train_combo": {
            "lookahead_races": int(best["lookahead_races"]),
            "price_appreciation_weight": float(best["price_appreciation_weight"]),
            "risk_tolerance": str(best["risk_tolerance"]),
            "train_points_mean": float(best["train_points_mean"]),
            "train_ending_budget_mean": float(best["train_ending_budget_mean"]),
        },
        "test_results": {
            "season_year": int(args.test_year),
            "total_points": float(test_backtest.get("total_points", 0.0)),
            "ending_budget": float(test_backtest.get("ending_budget", 0.0)),
            "initial_team_kpi": test_backtest.get("initial_team_kpi", {}),
            "prediction_decision_metrics": test_backtest.get("prediction_decision_metrics", {}),
            "baselines": test_backtest.get("baselines", {}),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = args.output_dir / "optimizer_tuning_train_results.csv"
    summary_json = args.output_dir / "optimizer_tuning_best_summary.json"
    train_df.to_csv(train_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2))

    print("Saved:")
    print(f"  {train_csv}")
    print(f"  {summary_json}")


if __name__ == "__main__":
    main()
