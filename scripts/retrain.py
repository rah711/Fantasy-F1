#!/usr/bin/env python3
"""Bundled retrain: pipeline → features → train.

Re-pulls 2026 race results from FastF1 (and any other live sources),
rebuilds features, and retrains the fantasy-points model. Use this after
a race weekend so future predictions incorporate the latest results.

Usage:
    PYTHONPATH=. python scripts/retrain.py            # all years (slow, safe)
    PYTHONPATH=. python scripts/retrain.py --skip-pipeline   # features + train only
    PYTHONPATH=. python scripts/retrain.py --no-walk-forward # skip backtest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, env={**dict(__import__('os').environ), "PYTHONPATH": str(PROJECT_ROOT)})
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"!!! {label} FAILED after {elapsed:.0f}s (exit {res.returncode})")
        sys.exit(res.returncode)
    print(f"--- {label} done in {elapsed:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the fantasy model end-to-end.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip the data pipeline step")
    parser.add_argument("--mode", choices=["full", "incremental", "2026_only"], default="full",
                        help="Pipeline mode (default: full — needed when training, since the model uses historical data)")
    parser.add_argument("--no-walk-forward", action="store_true", help="Skip the walk-forward backtest")
    args = parser.parse_args()

    if not args.skip_pipeline:
        _run("1/3 Pipeline (sessions.parquet)", [PYTHON, "scripts/run_pipeline.py", "--mode", args.mode])
    else:
        print("Skipping pipeline step (--skip-pipeline).")

    _run("2/3 Features (features.parquet)", [PYTHON, "scripts/run_features.py"])

    train_cmd = [PYTHON, "scripts/train_model.py"]
    if args.no_walk_forward:
        train_cmd.append("--no-walk-forward")
    _run("3/3 Train model (fantasy_model.joblib)", train_cmd)

    print("\nRetrain complete.")


if __name__ == "__main__":
    main()
