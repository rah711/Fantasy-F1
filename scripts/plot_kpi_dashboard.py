#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Allow running this script directly without requiring PYTHONPATH=.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.optimizer import TeamOptimizer


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or pd.isna(v):
        return ""
    return f"{float(v):.{digits}f}"


def _compute_summary(
    years: list[int],
    predictions_dir: Path,
    config_path: Path | None,
    num_alternatives: int | None,
    random_seed: int,
) -> pd.DataFrame:
    cfg = load_config(str(config_path)) if config_path else load_config()
    opt = TeamOptimizer(cfg)

    num_alts = num_alternatives if num_alternatives is not None else opt.initial_kpi_num_alternatives

    rows: list[dict[str, float | int | str | None]] = []
    for year in years:
        pred_path = predictions_dir / f"walk_forward_{year}_predictions.parquet"
        if not pred_path.exists():
            print(f"Skipping {year}: missing {pred_path}")
            continue

        preds = pd.read_parquet(pred_path)
        kpi = opt.evaluate_initial_team_kpi(
            predictions=preds,
            season_year=year,
            num_alternatives=num_alts,
            random_seed=random_seed,
        )
        dec = opt.evaluate_prediction_decision_metrics(predictions=preds, season_year=year)

        chosen_team = kpi.get("chosen_team", {}) if isinstance(kpi, dict) else {}
        rows.append(
            {
                "year": year,
                "kpi_percentile": kpi.get("percentile_vs_feasible_alternatives"),
                "kpi_delta_vs_avg": kpi.get("delta_vs_avg_alternative"),
                "kpi_delta_vs_best": kpi.get("delta_vs_best_alternative"),
                "kpi_num_feasible_total": kpi.get("num_feasible_total"),
                "kpi_num_alternatives_tested": kpi.get("num_feasible_alternatives_tested"),
                "chosen_team_points": kpi.get("chosen_team_points"),
                "chosen_team_cost": chosen_team.get("cost"),
                "driver_rank_spearman_mean": dec.get("driver_rank_spearman_mean") if isinstance(dec, dict) else None,
                "driver_top5_hit_rate": dec.get("driver_top5_hit_rate") if isinstance(dec, dict) else None,
                "drs_top1_hit_rate": dec.get("drs_top1_hit_rate") if isinstance(dec, dict) else None,
                "constructor_top2_hit_rate": dec.get("constructor_top2_hit_rate") if isinstance(dec, dict) else None,
                "dnf_brier_score": dec.get("dnf_brier_score") if isinstance(dec, dict) else None,
            }
        )

    if not rows:
        raise ValueError("No yearly prediction files found; nothing to plot.")
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _plot_dashboard(df: pd.DataFrame, output_png: Path) -> None:
    years = df["year"].astype(str).tolist()
    xi = list(range(len(df)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Fantasy F1 Optimizer KPI Dashboard", fontsize=14, fontweight="bold")

    # 1) Main KPI: percentile vs feasible alternatives
    ax = axes[0, 0]
    ax.bar(xi, df["kpi_percentile"], color="#1f77b4")
    ax.set_xticks(xi, years)
    ax.set_ylim(0, 1)
    ax.set_title("Initial Team KPI Percentile")
    ax.set_ylabel("Percentile (0-1)")
    for i, v in enumerate(df["kpi_percentile"]):
        ax.text(i, float(v) + 0.02, _fmt(v, 2), ha="center", fontsize=9)

    # 2) KPI deltas vs alternatives
    ax = axes[0, 1]
    w = 0.35
    ax.bar([i - w / 2 for i in xi], df["kpi_delta_vs_avg"], width=w, label="vs Avg Alternative", color="#2ca02c")
    ax.bar([i + w / 2 for i in xi], df["kpi_delta_vs_best"], width=w, label="vs Best Alternative", color="#ff7f0e")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(xi, years)
    ax.set_title("Initial Team KPI Point Deltas")
    ax.set_ylabel("Season Points Delta")
    ax.legend()

    # 3) Decision hit-rate metrics
    ax = axes[1, 0]
    ax.plot(xi, df["driver_top5_hit_rate"], marker="o", label="Driver Top-5 Hit Rate")
    ax.plot(xi, df["drs_top1_hit_rate"], marker="o", label="DRS Top-1 Hit Rate")
    ax.plot(xi, df["constructor_top2_hit_rate"], marker="o", label="Constructor Top-2 Hit Rate")
    ax.set_xticks(xi, years)
    ax.set_ylim(0, 1)
    ax.set_title("Decision Hit Rates")
    ax.set_ylabel("Rate (0-1)")
    ax.legend(loc="lower right")

    # 4) Rank quality + DNF calibration
    ax = axes[1, 1]
    ax.plot(xi, df["driver_rank_spearman_mean"], marker="o", color="#9467bd", label="Driver Rank Spearman")
    ax.plot(xi, df["dnf_brier_score"], marker="o", color="#8c564b", label="DNF Brier (lower better)")
    ax.set_xticks(xi, years)
    ax.set_title("Ranking & DNF Calibration")
    ax.legend(loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fantasy F1 optimizer KPI dashboard")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025], help="Season years to include")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/processed/models/backtest"),
        help="Directory containing walk_forward_<year>_predictions.parquet",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/optimizer_reports"),
        help="Directory to write dashboard and CSV",
    )
    parser.add_argument(
        "--num-alternatives",
        type=int,
        default=None,
        help="Override number of feasible alternatives for initial-team KPI",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for alternative-team sampling")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Use an existing kpi_summary.csv and only render the PNG",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "kpi_summary.csv"
    png_path = args.output_dir / "kpi_dashboard.png"

    if args.summary_csv:
        if not args.summary_csv.exists():
            raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")
        df = pd.read_csv(args.summary_csv).sort_values("year").reset_index(drop=True)
    else:
        df = _compute_summary(
            years=args.years,
            predictions_dir=args.predictions_dir,
            config_path=args.config,
            num_alternatives=args.num_alternatives,
            random_seed=args.random_seed,
        )
        df.to_csv(csv_path, index=False)

    _plot_dashboard(df, png_path)

    print("Saved:")
    if not args.summary_csv:
        print(f"  CSV: {csv_path}")
    print(f"  PNG: {png_path}")
    print("\nKPI Summary:")

    printable = df.copy()
    for col in [
        "kpi_percentile",
        "kpi_delta_vs_avg",
        "kpi_delta_vs_best",
        "driver_rank_spearman_mean",
        "driver_top5_hit_rate",
        "drs_top1_hit_rate",
        "constructor_top2_hit_rate",
        "dnf_brier_score",
    ]:
        if col in printable.columns:
            printable[col] = printable[col].map(lambda v: _fmt(v, 3))
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
