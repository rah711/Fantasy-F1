#!/usr/bin/env python3
"""
CLI entry point for feature engineering.

Usage:
    python scripts/run_features.py [--sessions PATH] [--output PATH]
"""

import argparse
from pathlib import Path

from src.config import load_config
from src.features.builder import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features from sessions.parquet")
    parser.add_argument(
        "--sessions",
        type=Path,
        default=Path("data/processed/sessions.parquet"),
        help="Path to sessions.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for features.parquet (default: data/processed/features.parquet)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    config = load_config(str(args.config)) if args.config else None
    build_features(
        sessions=args.sessions,
        config=config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
