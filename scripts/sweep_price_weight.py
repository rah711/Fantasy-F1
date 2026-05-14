#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.optimizer import TeamOptimizer


def _parse_weights(raw: str) -> list[float]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("No weights provided")
    return out


def _is_pareto(points: pd.Series, budget: pd.Series) -> list[bool]:
    flags: list[bool] = []
    for i in range(len(points)):
        p_i = float(points.iloc[i])
        b_i = float(budget.iloc[i])
        dominated = False
        for j in range(len(points)):
            if i == j:
                continue
            p_j = float(points.iloc[j])
            b_j = float(budget.iloc[j])
            if (p_j >= p_i and b_j >= b_i) and (p_j > p_i or b_j > b_i):
                dominated = True
                break
        flags.append(not dominated)
    return flags


def _normalize(col: pd.Series) -> pd.Series:
    vmin = float(col.min())
    vmax = float(col.max())
    if vmax == vmin:
        return pd.Series([1.0] * len(col), index=col.index)
    return (col - vmin) / (vmax - vmin)


def _plot(df: pd.DataFrame, output_png: Path, season_year: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Tradeoff scatter (Pareto front highlighted)
    ax = axes[0]
    colors = ["#d62728" if x else "#1f77b4" for x in df["is_pareto"]]
    ax.scatter(df["ending_budget"], df["total_points"], c=colors)
    for _, r in df.iterrows():
        ax.annotate(f"w={r['price_appreciation_weight']:.2f}", (r["ending_budget"], r["total_points"]), fontsize=8)
    ax.set_xlabel("Ending Budget (M)")
    ax.set_ylabel("Total Season Points")
    ax.set_title(f"{season_year} Tradeoff: Points vs Budget")

    # Weight curves
    ax = axes[1]
    ax.plot(df["price_appreciation_weight"], df["total_points"], marker="o", label="Total points")
    ax.plot(df["price_appreciation_weight"], df["ending_budget"], marker="o", label="Ending budget")
    ax.set_xlabel("price_appreciation_weight")
    ax.set_title("Metric Curves by Weight")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep optimizer price-appreciation weight on a target season")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/processed/models/backtest"))
    parser.add_argument("--season-year", type=int, default=2025)
    parser.add_argument("--weights", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--risk", type=str, default=None, choices=["conservative", "moderate", "aggressive"])
    parser.add_argument("--num-alternatives", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--enable-initial-kpi",
        action="store_true",
        help="Enable initial-team KPI inside each backtest (slower).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/optimizer_reports"))
    args = parser.parse_args()

    cfg = load_config(str(args.config)) if args.config else load_config()

    pred_path = args.predictions_dir / f"walk_forward_{args.season_year}_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {pred_path}")
    preds = pd.read_parquet(pred_path)

    weights = _parse_weights(args.weights)
    rows: list[dict[str, float | int | bool]] = []

    for w in weights:
        combo_cfg = copy.deepcopy(cfg)
        combo_cfg.setdefault("optimizer", {})
        combo_cfg["optimizer"]["price_appreciation_weight"] = float(w)
        if args.lookahead is not None:
            combo_cfg["optimizer"]["lookahead_races"] = int(args.lookahead)
        if args.risk is not None:
            combo_cfg["optimizer"]["risk_tolerance"] = str(args.risk)
        combo_cfg["optimizer"].setdefault("initial_team_kpi", {})
        combo_cfg["optimizer"]["initial_team_kpi"]["enabled"] = bool(args.enable_initial_kpi)
        combo_cfg["optimizer"]["initial_team_kpi"]["num_alternatives"] = int(args.num_alternatives)
        combo_cfg["optimizer"]["initial_team_kpi"]["random_seed"] = int(args.random_seed)

        opt = TeamOptimizer(combo_cfg)
        result = opt.backtest_season(predictions=preds, season_year=args.season_year, current_team={})

        baselines = result.get("baselines", {})
        deltas_pts_only = baselines.get("deltas_vs_points_only", {}) if isinstance(baselines, dict) else {}
        init_kpi = result.get("initial_team_kpi", {}) if isinstance(result, dict) else {}

        rows.append(
            {
                "season_year": int(args.season_year),
                "price_appreciation_weight": float(w),
                "lookahead_races": int(opt.lookahead),
                "risk_tolerance": str(opt.risk_tolerance),
                "total_points": float(result.get("total_points", 0.0)),
                "ending_budget": float(result.get("ending_budget", 0.0)),
                "delta_vs_points_only_points": float(deltas_pts_only.get("points", 0.0)),
                "delta_vs_points_only_budget": float(deltas_pts_only.get("ending_budget", 0.0)),
                "initial_kpi_percentile": float(init_kpi.get("percentile_vs_feasible_alternatives", 0.0)) if isinstance(init_kpi, dict) and init_kpi.get("percentile_vs_feasible_alternatives") is not None else float("nan"),
            }
        )

    df = pd.DataFrame(rows).sort_values("price_appreciation_weight").reset_index(drop=True)
    df["is_pareto"] = _is_pareto(df["total_points"], df["ending_budget"])
    df["points_norm"] = _normalize(df["total_points"])
    df["budget_norm"] = _normalize(df["ending_budget"])
    df["balance_score"] = (df["points_norm"] * df["budget_norm"]) ** 0.5

    best_points = df.sort_values(["total_points", "ending_budget"], ascending=[False, False]).iloc[0]
    best_budget = df.sort_values(["ending_budget", "total_points"], ascending=[False, False]).iloc[0]
    best_balance = df.sort_values(["balance_score", "total_points"], ascending=[False, False]).iloc[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"price_weight_sweep_{args.season_year}.csv"
    png_path = args.output_dir / f"price_weight_sweep_{args.season_year}.png"
    summary_path = args.output_dir / f"price_weight_sweep_{args.season_year}_summary.json"

    df.to_csv(csv_path, index=False)
    _plot(df, png_path, args.season_year)

    summary = {
        "season_year": int(args.season_year),
        "weights": [float(w) for w in weights],
        "best_total_points": best_points.to_dict(),
        "best_ending_budget": best_budget.to_dict(),
        "best_balance": best_balance.to_dict(),
        "pareto_weights": df[df["is_pareto"]]["price_appreciation_weight"].tolist(),
        "note": "best_balance uses geometric mean of normalized total_points and ending_budget",
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Saved:")
    print(f"  CSV: {csv_path}")
    print(f"  PNG: {png_path}")
    print(f"  JSON: {summary_path}")
    print("\nTop picks:")
    print(f"  best_points_weight={best_points['price_appreciation_weight']:.3f} total_points={best_points['total_points']:.2f} ending_budget={best_points['ending_budget']:.2f}")
    print(f"  best_budget_weight={best_budget['price_appreciation_weight']:.3f} total_points={best_budget['total_points']:.2f} ending_budget={best_budget['ending_budget']:.2f}")
    print(f"  best_balance_weight={best_balance['price_appreciation_weight']:.3f} total_points={best_balance['total_points']:.2f} ending_budget={best_balance['ending_budget']:.2f}")


if __name__ == "__main__":
    main()
