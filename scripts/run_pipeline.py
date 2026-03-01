#!/usr/bin/env python3
"""
CLI entry point for the Fantasy F1 data pipeline.

Usage:
    python scripts/run_pipeline.py [--mode full|incremental|2026_only] [--output PATH]
"""

import argparse
from pathlib import Path

from src.data.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fantasy F1 data pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "2026_only"],
        default="full",
        help="Pipeline mode: full (all years), incremental, 2026_only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for sessions.parquet (default: data/processed/sessions.parquet)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project root)",
    )
    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        output_path=args.output,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
