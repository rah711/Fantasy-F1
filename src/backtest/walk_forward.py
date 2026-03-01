"""Walk-forward backtest for Fantasy F1 model.

This implements a proper temporal walk-forward evaluation for one test season:
for each round in the test year, train on all data up to (but not including)
that round, predict that round, and accumulate metrics.

Includes:
  - Main fantasy-points regressor
  - Separate DNF/penalty risk classifier (binary: did the driver DNF/DSQ?)
  - Quantile regression for prediction intervals (uncertainty estimates)
  - Naive baseline benchmarks for comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from src.model.train import CAT_COLS, FEATURE_COLS, TARGET
from src.utils.logging import get_logger

log = get_logger(__name__)

DNF_TARGET = "is_dnf"


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build X with one-hot categorical encoding and numeric fill."""
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].copy()
    for c in [c for c in cat_cols if c in X.columns]:
        dummies = pd.get_dummies(X[c], prefix=c, drop_first=True, dtype=float)
        X = pd.concat([X.drop(columns=[c]), dummies], axis=1)
    X = X.astype(float).fillna(0)
    return X, list(X.columns)


def _build_regressor(engine: str, hp: dict[str, Any]):
    """Instantiate point-prediction regressor from config."""
    if engine == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**hp, verbosity=-1, random_state=42)
    if engine == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**hp, random_state=42)
    raise ValueError(f"Unknown model engine: {engine}")


def _build_classifier(engine: str, hp: dict[str, Any]):
    """Binary classifier for DNF/DSQ risk."""
    clf_hp = {k: v for k, v in hp.items() if k not in ("objective", "metric")}
    clf_hp["n_estimators"] = min(clf_hp.get("n_estimators", 100), 150)
    if engine == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**clf_hp, verbosity=-1, random_state=42)
    if engine == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**clf_hp, random_state=42, use_label_encoder=False, eval_metric="logloss")
    raise ValueError(f"Unknown model engine: {engine}")


def _build_quantile_regressor(engine: str, hp: dict[str, Any], alpha: float):
    """Quantile regressor for prediction intervals."""
    q_hp = {k: v for k, v in hp.items() if k not in ("objective", "metric")}
    if engine == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**q_hp, objective="quantile", alpha=alpha, verbosity=-1, random_state=42)
    if engine == "xgboost":
        import xgboost as xgb
        from functools import partial

        def _quantile_obj(alpha_val, y_true, y_pred):
            err = y_true - y_pred
            grad = np.where(err >= 0, -alpha_val, -(alpha_val - 1))
            hess = np.ones_like(grad)
            return grad, hess

        return xgb.XGBRegressor(
            **q_hp,
            objective=partial(_quantile_obj, alpha),
            random_state=42,
        )
    raise ValueError(f"Unknown model engine: {engine}")


def _overall_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute MAE/RMSE for aligned series."""
    err = y_pred - y_true
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
    }


def _align_columns(X_test: pd.DataFrame, train_cols: list[str]) -> pd.DataFrame:
    """Ensure X_test has exactly the same columns as training data."""
    for c in train_cols:
        if c not in X_test.columns:
            X_test[c] = 0.0
    return X_test[train_cols]


def walk_forward_backtest(
    features_path: str | Path,
    config: dict[str, Any],
    test_year: int,
    train_start_year: int,
    output_dir: str | Path = "data/processed/backtest",
) -> dict[str, Any]:
    """Run round-by-round walk-forward test for a given season.

    Training set at round r: all rows with year < test_year plus rounds < r in test_year.
    Test set at round r: all rows in (test_year, round == r).

    Produces:
      - Point predictions (y_pred)
      - DNF probability (dnf_prob) from a separate classifier
      - Risk-adjusted prediction: y_pred * (1 - dnf_prob) + dnf_penalty * dnf_prob
      - Prediction intervals: y_pred_q10, y_pred_q90 (80% interval)
      - Baseline benchmarks for comparison
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    df = pd.read_parquet(features_path)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if "year" not in df.columns or "round" not in df.columns:
        raise ValueError("Features require year and round columns for walk-forward")

    df = df[df["session_type"] == "race"].copy()
    df = df.sort_values(["year", "round", "driver_code"]).reset_index(drop=True)

    # Create DNF binary target from status
    df[DNF_TARGET] = df["status"].astype(str).str.upper().isin(("DNF", "DSQ")).astype(int)

    # DNF penalty from config (race DNF/DSQ penalty, typically -20)
    dnf_penalty = config.get("scoring", {}).get("race", {}).get("dnf_dsq_penalty", -20)

    test_df_all = df[df["year"] == test_year].copy()
    if test_df_all.empty:
        raise ValueError(f"No rows for test year {test_year}")

    rounds = sorted(test_df_all["round"].dropna().astype(int).unique().tolist())
    if not rounds:
        raise ValueError(f"No rounds found in test year {test_year}")

    engine = config.get("model", {}).get("engine", "lightgbm")
    hp = config.get("model", {}).get("hyperparameters", {}).get(engine, {})
    quantiles = config.get("model", {}).get("quantiles", [0.1, 0.5, 0.9])
    q_low = min(q for q in quantiles if q < 0.5) if any(q < 0.5 for q in quantiles) else 0.1
    q_high = max(q for q in quantiles if q > 0.5) if any(q > 0.5 for q in quantiles) else 0.9
    all_feature_cols = FEATURE_COLS + CAT_COLS

    preds = []
    skipped_rounds: list[int] = []

    for r in rounds:
        train_df = df[
            (df["year"] >= train_start_year)
            & ((df["year"] < test_year) | ((df["year"] == test_year) & (df["round"] < r)))
        ].copy()
        test_df = test_df_all[test_df_all["round"] == r].copy()

        if train_df.empty or test_df.empty:
            skipped_rounds.append(int(r))
            continue

        X_train, train_cols = _prepare_xy(train_df, all_feature_cols, CAT_COLS)
        y_train = train_df[TARGET].astype(float)
        y_train_dnf = train_df[DNF_TARGET].astype(int)

        X_test, _ = _prepare_xy(test_df, all_feature_cols, CAT_COLS)
        X_test = _align_columns(X_test, train_cols)

        # 1) Main regressor
        model = _build_regressor(engine, hp)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 2) DNF risk classifier
        dnf_prob = np.zeros(len(X_test))
        if y_train_dnf.sum() >= 3:
            try:
                clf = _build_classifier(engine, hp)
                clf.fit(X_train, y_train_dnf)
                dnf_prob = clf.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        # Risk-adjusted: blend expected finish points with DNF penalty weighted by risk
        y_pred_risk_adj = y_pred * (1 - dnf_prob) + dnf_penalty * dnf_prob

        # 3) Quantile regression for uncertainty bands
        y_pred_q_low = y_pred.copy()
        y_pred_q_high = y_pred.copy()
        try:
            q_model_low = _build_quantile_regressor(engine, hp, q_low)
            q_model_low.fit(X_train, y_train)
            y_pred_q_low = q_model_low.predict(X_test)

            q_model_high = _build_quantile_regressor(engine, hp, q_high)
            q_model_high.fit(X_train, y_train)
            y_pred_q_high = q_model_high.predict(X_test)
        except Exception:
            pass

        round_out = test_df[["year", "round", "driver_code", "constructor_id", TARGET, DNF_TARGET]].copy()
        round_out = round_out.rename(columns={TARGET: "y_true"})
        round_out["y_pred"] = y_pred
        round_out["dnf_prob"] = dnf_prob
        round_out["y_pred_risk_adj"] = y_pred_risk_adj
        round_out[f"y_pred_q{int(q_low*100):02d}"] = y_pred_q_low
        round_out[f"y_pred_q{int(q_high*100):02d}"] = y_pred_q_high
        round_out["abs_error"] = (round_out["y_pred"] - round_out["y_true"]).abs()
        round_out["sq_error"] = (round_out["y_pred"] - round_out["y_true"]) ** 2
        round_out["abs_error_risk_adj"] = (round_out["y_pred_risk_adj"] - round_out["y_true"]).abs()
        round_out["sq_error_risk_adj"] = (round_out["y_pred_risk_adj"] - round_out["y_true"]) ** 2

        # Baseline benchmarks
        train_sorted = train_df.sort_values(["year", "round"])
        global_mean = float(train_sorted[TARGET].mean())
        prev_driver_map = train_sorted.groupby("driver_code")[TARGET].last().to_dict()
        roll3_driver_map = train_sorted.groupby("driver_code")[TARGET].apply(lambda s: float(s.tail(3).mean())).to_dict()
        constructor_mean_map = train_sorted.groupby("constructor_id")[TARGET].mean().to_dict()

        round_out["y_pred_prev_driver"] = round_out["driver_code"].map(prev_driver_map).fillna(global_mean)
        round_out["y_pred_roll3_driver"] = round_out["driver_code"].map(roll3_driver_map).fillna(global_mean)
        round_out["y_pred_constructor_mean"] = round_out["constructor_id"].map(constructor_mean_map).fillna(global_mean)

        for name in ["prev_driver", "roll3_driver", "constructor_mean"]:
            round_out[f"abs_error_{name}"] = (round_out[f"y_pred_{name}"] - round_out["y_true"]).abs()
            round_out[f"sq_error_{name}"] = (round_out[f"y_pred_{name}"] - round_out["y_true"]) ** 2

        preds.append(round_out)

    if not preds:
        raise ValueError("Walk-forward produced no predictions. Check data coverage.")

    pred_df = pd.concat(preds, ignore_index=True)
    overall_mae = float(pred_df["abs_error"].mean())
    overall_rmse = float(np.sqrt(pred_df["sq_error"].mean()))

    overall_mae_risk = float(pred_df["abs_error_risk_adj"].mean())
    overall_rmse_risk = float(np.sqrt(pred_df["sq_error_risk_adj"].mean()))

    # DNF classifier accuracy
    dnf_actual = pred_df[DNF_TARGET].sum()
    dnf_predicted_top = (pred_df["dnf_prob"] >= 0.5).sum()

    # Prediction interval coverage
    q_low_col = f"y_pred_q{int(q_low*100):02d}"
    q_high_col = f"y_pred_q{int(q_high*100):02d}"
    if q_low_col in pred_df.columns and q_high_col in pred_df.columns:
        in_interval = ((pred_df["y_true"] >= pred_df[q_low_col]) & (pred_df["y_true"] <= pred_df[q_high_col]))
        interval_coverage = float(in_interval.mean())
        interval_width = float((pred_df[q_high_col] - pred_df[q_low_col]).mean())
    else:
        interval_coverage = None
        interval_width = None

    per_round = (
        pred_df.groupby("round")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(s.mean()))),
            mae_risk_adj=("abs_error_risk_adj", "mean"),
            mae_prev_driver=("abs_error_prev_driver", "mean"),
            mae_roll3_driver=("abs_error_roll3_driver", "mean"),
            mae_constructor_mean=("abs_error_constructor_mean", "mean"),
            n=("driver_code", "count"),
        )
        .reset_index()
        .sort_values("round")
    )

    benchmark_overall = {
        "prev_driver": _overall_metrics(pred_df["y_true"], pred_df["y_pred_prev_driver"]),
        "roll3_driver": _overall_metrics(pred_df["y_true"], pred_df["y_pred_roll3_driver"]),
        "constructor_mean": _overall_metrics(pred_df["y_true"], pred_df["y_pred_constructor_mean"]),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"walk_forward_{test_year}_predictions.parquet"
    metrics_path = output_dir / f"walk_forward_{test_year}_metrics.json"

    pred_df.to_parquet(pred_path, index=False)
    metrics = {
        "test_year": int(test_year),
        "train_start_year": int(train_start_year),
        "overall": {"mae": overall_mae, "rmse": overall_rmse, "n": int(len(pred_df))},
        "overall_risk_adjusted": {"mae": overall_mae_risk, "rmse": overall_rmse_risk},
        "dnf_risk": {
            "actual_dnfs": int(dnf_actual),
            "predicted_dnf_ge50pct": int(dnf_predicted_top),
        },
        "uncertainty": {
            "quantiles": [q_low, q_high],
            "interval_coverage": interval_coverage,
            "mean_interval_width": interval_width,
        },
        "benchmark_overall": benchmark_overall,
        "per_round": per_round.to_dict(orient="records"),
        "skipped_rounds": skipped_rounds,
        "predictions_path": str(pred_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    log.info(
        "Walk-forward %s complete: MAE=%.2f (risk-adj=%.2f) RMSE=%.2f (rounds=%d, coverage=%.0f%%)",
        test_year,
        overall_mae,
        overall_mae_risk,
        overall_rmse,
        pred_df["round"].nunique(),
        (interval_coverage or 0) * 100,
    )

    return metrics
