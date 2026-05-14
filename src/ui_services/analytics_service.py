from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FileHealth:
    path: str
    exists: bool
    mtime: float | None
    size_bytes: int | None


def _file_health(path: str | Path) -> FileHealth:
    p = Path(path)
    if not p.exists():
        return FileHealth(path=str(p), exists=False, mtime=None, size_bytes=None)
    stat = p.stat()
    return FileHealth(path=str(p), exists=True, mtime=stat.st_mtime, size_bytes=stat.st_size)


def collect_health_checks(project_root: str | Path = ".") -> dict[str, FileHealth]:
    root = Path(project_root)
    targets = {
        "features": root / "data/processed/features.parquet",
        "model": root / "data/processed/models/fantasy_model.joblib",
        "predictions_dir": root / "data/processed/predictions",
        "kpi_png": root / "data/processed/optimizer_reports/kpi_dashboard.png",
        "strict_summary": root / "data/processed/optimizer_tuning_strict_weight/optimizer_tuning_best_summary.json",
        "weight_sweep_csv": root / "data/processed/optimizer_reports/price_weight_sweep_2025.csv",
    }
    return {k: _file_health(v) for k, v in targets.items()}


def load_csv_if_exists(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    import json

    return json.loads(p.read_text())


def load_analytics_bundle(project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    return {
        "kpi_summary": load_csv_if_exists(root / "data/processed/optimizer_reports/kpi_summary.csv"),
        "strict_tuning_train": load_csv_if_exists(root / "data/processed/optimizer_tuning_strict_weight/optimizer_tuning_train_results.csv"),
        "strict_tuning_summary": load_json_if_exists(root / "data/processed/optimizer_tuning_strict_weight/optimizer_tuning_best_summary.json"),
        "price_weight_sweep": load_csv_if_exists(root / "data/processed/optimizer_reports/price_weight_sweep_2025.csv"),
        "health": collect_health_checks(root),
    }
