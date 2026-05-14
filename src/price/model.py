from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceThreshold:
    terrible_max: float
    poor_max: float
    good_max: float


@dataclass(frozen=True)
class PriceUpdate:
    entity_type: str
    asset_id: str
    round_number: int
    old_price: float
    predicted_points: float
    band: str
    delta: float
    new_price: float
    threshold: PriceThreshold


# Round 1 official thresholds provided by user.
ROUND1_DRIVER_THRESHOLDS: dict[str, dict[str, float]] = {
    "VER": {"price": 27.7, "terrible_max": 16, "poor_max": 24, "good_max": 33},
    "RUS": {"price": 27.4, "terrible_max": 16, "poor_max": 24, "good_max": 32},
    "NOR": {"price": 27.2, "terrible_max": 16, "poor_max": 24, "good_max": 32},
    "PIA": {"price": 25.5, "terrible_max": 15, "poor_max": 22, "good_max": 30},
    "ANT": {"price": 23.2, "terrible_max": 13, "poor_max": 20, "good_max": 27},
    "LEC": {"price": 22.8, "terrible_max": 13, "poor_max": 20, "good_max": 27},
    "HAM": {"price": 22.5, "terrible_max": 13, "poor_max": 20, "good_max": 26},
    "HAD": {"price": 15.1, "terrible_max": 9, "poor_max": 13, "good_max": 18},
    "GAS": {"price": 12.0, "terrible_max": 7, "poor_max": 10, "good_max": 14},
    "SAI": {"price": 11.8, "terrible_max": 7, "poor_max": 10, "good_max": 14},
    "ALB": {"price": 11.6, "terrible_max": 6, "poor_max": 10, "good_max": 13},
    "ALO": {"price": 10.0, "terrible_max": 5, "poor_max": 8, "good_max": 11},
    "STR": {"price": 8.0, "terrible_max": 4, "poor_max": 7, "good_max": 9},
    "BEA": {"price": 7.4, "terrible_max": 4, "poor_max": 6, "good_max": 8},
    "OCO": {"price": 7.3, "terrible_max": 4, "poor_max": 7, "good_max": 8},
    "HUL": {"price": 6.8, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "LAW": {"price": 6.5, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "BOR": {"price": 6.4, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "COL": {"price": 6.2, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "LIN": {"price": 6.2, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "PER": {"price": 6.0, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "BOT": {"price": 5.9, "terrible_max": 3, "poor_max": 5, "good_max": 7},
}

ROUND1_CONSTRUCTOR_CN_THRESHOLDS: dict[str, dict[str, float]] = {
    "MER": {"price": 29.3, "terrible_max": 17, "poor_max": 26, "good_max": 35},
    "MCL": {"price": 28.9, "terrible_max": 17, "poor_max": 26, "good_max": 34},
    "RBR": {"price": 28.2, "terrible_max": 16, "poor_max": 25, "good_max": 33},
    "FER": {"price": 23.3, "terrible_max": 13, "poor_max": 20, "good_max": 27},
    "ALP": {"price": 12.5, "terrible_max": 7, "poor_max": 11, "good_max": 14},
    "WIL": {"price": 12.0, "terrible_max": 7, "poor_max": 10, "good_max": 12},
    "AST": {"price": 10.3, "terrible_max": 6, "poor_max": 9, "good_max": 12},
    "HAA": {"price": 7.4, "terrible_max": 4, "poor_max": 6, "good_max": 8},
    "AUD": {"price": 6.6, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "VRB": {"price": 6.3, "terrible_max": 3, "poor_max": 5, "good_max": 7},
    "CAD": {"price": 6.0, "terrible_max": 3, "poor_max": 5, "good_max": 7},
}

CONSTRUCTOR_CN_TO_ID = {
    "MER": "mercedes",
    "MCL": "mclaren",
    "RBR": "red_bull",
    "FER": "ferrari",
    "ALP": "alpine",
    "WIL": "williams",
    "AST": "aston_martin",
    "HAA": "haas",
    "AUD": "audi",
    "VRB": "racing_bulls",
    "CAD": "cadillac",
}


def _to_threshold_row(row: dict[str, float]) -> PriceThreshold:
    return PriceThreshold(
        terrible_max=float(row["terrible_max"]),
        poor_max=float(row["poor_max"]),
        good_max=float(row["good_max"]),
    )


class PriceModel:
    """Threshold-based Fantasy F1 price model.

    Round 1 uses explicit official thresholds for 2026 entities.
    For later rounds and backtests, thresholds are estimated from current price
    by interpolation against the Round 1 threshold table.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tier_cutoff = 18.5
        self.floor_price = 3.0

        self.driver_round1 = {k: _to_threshold_row(v) for k, v in ROUND1_DRIVER_THRESHOLDS.items()}
        self.constructor_round1 = {
            CONSTRUCTOR_CN_TO_ID[k]: _to_threshold_row(v)
            for k, v in ROUND1_CONSTRUCTOR_CN_THRESHOLDS.items()
        }

        self._driver_interp = self._build_interpolator(ROUND1_DRIVER_THRESHOLDS)
        constructor_interp_table = {
            CONSTRUCTOR_CN_TO_ID[k]: v for k, v in ROUND1_CONSTRUCTOR_CN_THRESHOLDS.items()
        }
        self._constructor_interp = self._build_interpolator(constructor_interp_table)

    @staticmethod
    def _build_interpolator(table: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
        rows = sorted(table.values(), key=lambda r: float(r["price"]))
        prices = np.array([float(r["price"]) for r in rows], dtype=float)
        terrible = np.array([float(r["terrible_max"]) for r in rows], dtype=float)
        poor = np.array([float(r["poor_max"]) for r in rows], dtype=float)
        good = np.array([float(r["good_max"]) for r in rows], dtype=float)
        return {
            "prices": prices,
            "terrible": terrible,
            "poor": poor,
            "good": good,
        }

    def _config_prices(self, entity_type: str) -> dict[str, float]:
        cfg_prices = self.config.get("prices", {})
        if entity_type == "driver":
            return {k: float(v["price"]) for k, v in cfg_prices.get("drivers", {}).items()}
        return {k: float(v["price"]) for k, v in cfg_prices.get("constructors", {}).items()}

    def _historical_price_overrides(self, entity_type: str, season_year: int) -> dict[str, float]:
        """Optional explicit historical prices from config.

        Expected shape in config:
        historical_prices:
          "2025":
            drivers: {VER: 30.5, ...}
            constructors: {mclaren: 33.3, ...}
        """
        hp = self.config.get("historical_prices", {})
        year_block = hp.get(str(season_year), hp.get(season_year, {}))
        if not isinstance(year_block, dict):
            return {}
        key = "drivers" if entity_type == "driver" else "constructors"
        raw = year_block.get(key, {})
        if not isinstance(raw, dict):
            return {}
        return {str(k): float(v) for k, v in raw.items()}

    @staticmethod
    def _pick_points_column(df: pd.DataFrame) -> str:
        for col in ("y_pred", "y_pred_risk_adj", "y_true"):
            if col in df.columns:
                return col
        raise ValueError("Predictions dataframe must include one of: y_pred, y_pred_risk_adj, y_true")

    def _map_rank_to_price_template(self, scores: pd.Series, template_prices_desc: np.ndarray) -> dict[str, float]:
        ranked = scores.sort_values(ascending=False)
        if ranked.empty:
            return {}

        template = template_prices_desc
        if template.size == 0:
            template = np.array([6.0], dtype=float)

        x = np.arange(len(template), dtype=float)
        n = len(ranked)
        out: dict[str, float] = {}
        for i, asset in enumerate(ranked.index.tolist()):
            q = i / max(1, n - 1)
            template_idx = q * max(1, len(template) - 1)
            price = float(np.interp(template_idx, x, template))
            out[str(asset)] = max(self.floor_price, price)
        return out

    def _historical_driver_scores(self, year_df: pd.DataFrame, score_col: str) -> pd.Series:
        # Opening context: blend driver expectation with team expectation at first race,
        # so driver transfers to stronger/weaker teams are reflected.
        opener_round = int(year_df["round"].min())
        opener = year_df[year_df["round"] == opener_round].copy()

        if opener.empty:
            return pd.Series(dtype=float)

        d_score = opener.groupby("driver_code")[score_col].mean()

        c_total = opener.groupby("constructor_id")[score_col].sum()
        c_driver_count = opener.groupby("constructor_id")["driver_code"].nunique().clip(lower=1)
        c_avg_per_driver = c_total / c_driver_count

        d_ctor = opener.groupby("driver_code")["constructor_id"].first()
        team_component = d_ctor.map(c_avg_per_driver).astype(float)
        team_component = team_component.fillna(float(d_score.mean()) if not d_score.empty else 0.0)

        return 0.75 * d_score + 0.25 * team_component

    def _historical_constructor_scores(self, year_df: pd.DataFrame, score_col: str) -> pd.Series:
        opener_round = int(year_df["round"].min())
        opener = year_df[year_df["round"] == opener_round].copy()
        if opener.empty:
            return pd.Series(dtype=float)
        return opener.groupby("constructor_id")[score_col].sum()

    def infer_opening_prices(
        self,
        predictions: pd.DataFrame,
        entity_type: str,
        season_year: int,
        use_official_for_config_season: bool = True,
    ) -> dict[str, float]:
        """Infer opening prices for a specific season.

        - For config season (2026), returns official configured prices.
        - For historical seasons, derives a season-specific opening price ranking
          from that season's own Round 1 projection context and maps it to the
          2026 opening price distribution.
        """
        config_year = int(self.config.get("season", {}).get("year", 2026))
        config_prices = self._config_prices(entity_type)
        explicit_overrides = self._historical_price_overrides(entity_type, season_year)

        year_df = predictions[predictions["year"] == season_year].copy()
        if explicit_overrides and year_df.empty:
            return explicit_overrides

        if use_official_for_config_season and season_year == config_year:
            if explicit_overrides:
                # Allow explicit overrides for config season if user provides them.
                out = dict(config_prices)
                out.update(explicit_overrides)
                return out
            return config_prices
        if year_df.empty:
            return explicit_overrides if explicit_overrides else config_prices

        score_col = self._pick_points_column(year_df)

        if entity_type == "driver":
            scores = self._historical_driver_scores(year_df, score_col)
            template = np.array(sorted(config_prices.values(), reverse=True), dtype=float)
            inferred = self._map_rank_to_price_template(scores, template)
        else:
            scores = self._historical_constructor_scores(year_df, score_col)
            template = np.array(sorted(config_prices.values(), reverse=True), dtype=float)
            inferred = self._map_rank_to_price_template(scores, template)

        if explicit_overrides:
            # Use explicit data where available; infer only missing season assets.
            inferred.update(explicit_overrides)
        return inferred if inferred else (explicit_overrides if explicit_overrides else config_prices)

    def seed_missing_prices_for_round(
        self,
        predictions: pd.DataFrame,
        entity_type: str,
        season_year: int,
        round_number: int,
        existing_prices: dict[str, float],
        score_col: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Assign prices to newly active assets that have no prior price.

        This supports mid-season driver swaps where an asset has no opening price
        because they were not active at the season start.
        """
        updated = dict(existing_prices)
        year_round = predictions[
            (predictions["year"] == season_year) & (predictions["round"] == round_number)
        ].copy()
        if year_round.empty:
            return updated, {}

        if entity_type == "driver":
            scores = year_round.groupby("driver_code")[score_col].mean()
        else:
            scores = year_round.groupby("constructor_id")[score_col].sum()
        if scores.empty:
            return updated, {}

        template = np.array(sorted(self._config_prices(entity_type).values(), reverse=True), dtype=float)
        rank_prices = self._map_rank_to_price_template(scores, template)

        seeded: dict[str, float] = {}
        for asset in scores.index.tolist():
            key = str(asset)
            if key in updated:
                continue
            seeded_price = float(rank_prices.get(key, self.floor_price))
            updated[key] = max(self.floor_price, seeded_price)
            seeded[key] = updated[key]

        return updated, seeded

    def estimate_threshold_from_price(self, entity_type: str, current_price: float) -> PriceThreshold:
        interp = self._driver_interp if entity_type == "driver" else self._constructor_interp
        p = float(current_price)
        t = float(np.interp(p, interp["prices"], interp["terrible"]))
        po = float(np.interp(p, interp["prices"], interp["poor"]))
        g = float(np.interp(p, interp["prices"], interp["good"]))

        # Keep monotonic and integer-like thresholds for stability.
        t_i = int(round(t))
        po_i = max(t_i + 1, int(round(po)))
        g_i = max(po_i + 1, int(round(g)))
        return PriceThreshold(float(t_i), float(po_i), float(g_i))

    def get_threshold(
        self,
        entity_type: str,
        asset_id: str,
        current_price: float,
        round_number: int,
    ) -> PriceThreshold:
        if round_number == 1:
            if entity_type == "driver" and asset_id in self.driver_round1:
                return self.driver_round1[asset_id]
            if entity_type == "constructor" and asset_id in self.constructor_round1:
                return self.constructor_round1[asset_id]
        return self.estimate_threshold_from_price(entity_type, current_price)

    @staticmethod
    def classify_band(points: float, threshold: PriceThreshold) -> str:
        p = float(points)
        if p <= threshold.terrible_max:
            return "terrible"
        if p <= threshold.poor_max:
            return "poor"
        if p <= threshold.good_max:
            return "good"
        return "great"

    def price_delta(self, current_price: float, band: str) -> float:
        high_tier = float(current_price) >= self.tier_cutoff
        if high_tier:
            mapping = {
                "terrible": -0.3,
                "poor": -0.1,
                "good": 0.1,
                "great": 0.3,
            }
        else:
            mapping = {
                "terrible": -0.6,
                "poor": -0.2,
                "good": 0.2,
                "great": 0.6,
            }
        return float(mapping[band])

    def apply_update(
        self,
        entity_type: str,
        asset_id: str,
        current_price: float,
        predicted_points: float,
        round_number: int,
    ) -> PriceUpdate:
        threshold = self.get_threshold(entity_type, asset_id, current_price, round_number)
        band = self.classify_band(predicted_points, threshold)
        raw_delta = self.price_delta(current_price, band)
        new_price = max(self.floor_price, float(current_price) + raw_delta)
        delta = new_price - float(current_price)
        return PriceUpdate(
            entity_type=entity_type,
            asset_id=asset_id,
            round_number=int(round_number),
            old_price=float(current_price),
            predicted_points=float(predicted_points),
            band=band,
            delta=float(delta),
            new_price=float(new_price),
            threshold=threshold,
        )

    def bootstrap_prices_for_unseen_entities(
        self,
        predictions: pd.DataFrame,
        entity_type: str,
        season_year: int,
    ) -> dict[str, float]:
        """Backward-compatible helper: infer prices for full seasonal universe."""
        return self.infer_opening_prices(
            predictions=predictions,
            entity_type=entity_type,
            season_year=season_year,
            use_official_for_config_season=True,
        )

    def constructor_points_from_driver_points(self, driver_round_df: pd.DataFrame, score_col: str) -> dict[str, float]:
        if driver_round_df.empty:
            return {}
        agg = driver_round_df.groupby("constructor_id")[score_col].sum()
        return {k: float(v) for k, v in agg.to_dict().items()}

    def simulate_round(
        self,
        round_number: int,
        driver_points: dict[str, float],
        constructor_points: dict[str, float],
        driver_prices: dict[str, float],
        constructor_prices: dict[str, float],
    ) -> tuple[list[PriceUpdate], dict[str, float], dict[str, float]]:
        updates: list[PriceUpdate] = []
        new_driver_prices = dict(driver_prices)
        new_constructor_prices = dict(constructor_prices)

        for code, pts in driver_points.items():
            if code not in new_driver_prices:
                continue
            upd = self.apply_update("driver", code, new_driver_prices[code], pts, round_number)
            updates.append(upd)
            new_driver_prices[code] = upd.new_price

        for cid, pts in constructor_points.items():
            if cid not in new_constructor_prices:
                continue
            upd = self.apply_update("constructor", cid, new_constructor_prices[cid], pts, round_number)
            updates.append(upd)
            new_constructor_prices[cid] = upd.new_price

        return updates, new_driver_prices, new_constructor_prices
