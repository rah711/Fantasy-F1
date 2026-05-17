from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd

from src.price.model import PriceModel


@dataclass
class TeamRecommendation:
    round_number: int
    drivers: list[str]
    constructors: list[str]
    drs_boost: str
    total_cost: float
    objective_score: float
    expected_points_next_race: float
    expected_price_gain_next_race: float


class TeamOptimizer:
    """Team selection and transfer optimizer for Fantasy F1."""

    def __init__(self, config: dict[str, Any], price_model: PriceModel | None = None) -> None:
        self.config = config
        self.price_model = price_model or PriceModel(config)

        fantasy_cfg = config.get("fantasy", {})
        self.budget = float(fantasy_cfg.get("budget", 100.0))
        self.num_drivers = int(fantasy_cfg.get("num_drivers", 5))
        self.num_constructors = int(fantasy_cfg.get("num_constructors", 2))
        self.drs_mult = float(fantasy_cfg.get("drs_boost_multiplier", 2.0))
        self.extra_transfer_cost = abs(float(fantasy_cfg.get("extra_transfer_cost", -10.0)))
        self.free_transfers_per_race = int(fantasy_cfg.get("free_transfers_per_race", 2))
        self.max_banked_transfers = int(fantasy_cfg.get("max_banked_transfers", 1))

        opt_cfg = config.get("optimizer", {})
        self.lookahead = int(opt_cfg.get("lookahead_races", 3))
        self.price_weight = float(opt_cfg.get("price_appreciation_weight", 0.3))
        self.risk_tolerance = str(opt_cfg.get("risk_tolerance", "moderate"))
        kpi_cfg = opt_cfg.get("initial_team_kpi", {})
        self.initial_kpi_enabled = bool(kpi_cfg.get("enabled", True))
        self.initial_kpi_num_alternatives = int(kpi_cfg.get("num_alternatives", 75))
        self.initial_kpi_random_seed = int(kpi_cfg.get("random_seed", 42))

    def _score_column(self, predictions: pd.DataFrame) -> str:
        if self.risk_tolerance == "conservative" and "y_pred_q10" in predictions.columns:
            return "y_pred_q10"
        if self.risk_tolerance == "aggressive" and "y_pred_q90" in predictions.columns:
            return "y_pred_q90"
        if "y_pred_risk_adj" in predictions.columns:
            return "y_pred_risk_adj"
        return "y_pred"

    def _initial_prices(
        self,
        predictions: pd.DataFrame,
        season_year: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        # Use official config prices for config season (2026) and
        # season-specific inferred opening prices for historical years.
        driver_prices = self.price_model.bootstrap_prices_for_unseen_entities(
            predictions,
            entity_type="driver",
            season_year=season_year,
        )
        constructor_prices = self.price_model.bootstrap_prices_for_unseen_entities(
            predictions,
            entity_type="constructor",
            season_year=season_year,
        )
        return driver_prices, constructor_prices

    def _asset_lookahead_utility(
        self,
        entity_type: str,
        asset_id: str,
        current_price: float,
        start_round: int,
        round_points: dict[int, dict[str, float]],
        horizon: int,
        price_weight: float,
        discount: float = 0.85,
    ) -> tuple[float, float, float, float]:
        points_total = 0.0
        price_gain_total = 0.0
        next_round_points = 0.0
        next_round_gain = 0.0
        p = float(current_price)

        for k in range(horizon):
            rnd = start_round + k
            pts = float(round_points.get(rnd, {}).get(asset_id, 0.0))
            d = discount ** k
            points_total += d * pts
            upd = self.price_model.apply_update(entity_type, asset_id, p, pts, rnd)
            price_gain_total += d * upd.delta
            if k == 0:
                next_round_points = pts
                next_round_gain = upd.delta
            p = upd.new_price

        objective = points_total + price_weight * price_gain_total
        return objective, next_round_points, price_gain_total, next_round_gain

    def _build_asset_tables(
        self,
        predictions: pd.DataFrame,
        season_year: int,
        start_round: int,
        driver_prices: dict[str, float],
        constructor_prices: dict[str, float],
        lookahead: int,
        price_weight: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        score_col = self._score_column(predictions)
        year_df = predictions[predictions["year"] == season_year].copy()

        if year_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        driver_round_points: dict[int, dict[str, float]] = {}
        constructor_round_points: dict[int, dict[str, float]] = {}

        for rnd in sorted(year_df["round"].astype(int).unique().tolist()):
            rdf = year_df[year_df["round"] == rnd]
            driver_round_points[rnd] = {
                k: float(v) for k, v in rdf.groupby("driver_code")[score_col].mean().to_dict().items()
            }
            constructor_round_points[rnd] = self.price_model.constructor_points_from_driver_points(rdf, score_col)

        active_driver_ids = set(driver_round_points.get(start_round, {}).keys())
        active_constructor_ids = set(constructor_round_points.get(start_round, {}).keys())

        driver_rows = []
        for code, price in driver_prices.items():
            if active_driver_ids and code not in active_driver_ids:
                continue
            obj, next_pts, total_gain, next_gain = self._asset_lookahead_utility(
                entity_type="driver",
                asset_id=code,
                current_price=price,
                start_round=start_round,
                round_points=driver_round_points,
                horizon=lookahead,
                price_weight=price_weight,
            )
            driver_rows.append(
                {
                    "asset": code,
                    "price": float(price),
                    "objective": obj,
                    "next_points": next_pts,
                    "price_gain_total": total_gain,
                    "next_price_gain": next_gain,
                }
            )

        constructor_rows = []
        for cid, price in constructor_prices.items():
            if active_constructor_ids and cid not in active_constructor_ids:
                continue
            obj, next_pts, total_gain, next_gain = self._asset_lookahead_utility(
                entity_type="constructor",
                asset_id=cid,
                current_price=price,
                start_round=start_round,
                round_points=constructor_round_points,
                horizon=lookahead,
                price_weight=price_weight,
            )
            constructor_rows.append(
                {
                    "asset": cid,
                    "price": float(price),
                    "objective": obj,
                    "next_points": next_pts,
                    "price_gain_total": total_gain,
                    "next_price_gain": next_gain,
                }
            )

        return pd.DataFrame(driver_rows), pd.DataFrame(constructor_rows)

    def _count_transfers(
        self,
        new_drivers: tuple[str, ...],
        new_ctors: tuple[str, ...],
        current_drivers: set[str],
        current_ctors: set[str],
    ) -> int:
        incoming_drivers = len(set(new_drivers) - current_drivers)
        incoming_ctors = len(set(new_ctors) - current_ctors)
        return incoming_drivers + incoming_ctors

    def _enumerate_feasible_teams(
        self,
        driver_df: pd.DataFrame,
        constructor_df: pd.DataFrame,
        budget: float,
    ) -> list[dict[str, Any]]:
        """Enumerate all feasible initial teams under budget."""
        drivers = [
            (
                str(row.asset),
                float(row.price),
                float(row.objective),
                float(row.next_points),
            )
            for row in driver_df.itertuples(index=False)
        ]
        ctors = [
            (
                str(row.asset),
                float(row.price),
                float(row.objective),
                float(row.next_points),
            )
            for row in constructor_df.itertuples(index=False)
        ]

        ctor_combos = []
        for combo in combinations(ctors, self.num_constructors):
            ids = tuple(sorted(c[0] for c in combo))
            cost = sum(c[1] for c in combo)
            obj = sum(c[2] for c in combo)
            ctor_combos.append((ids, cost, obj))

        teams: list[dict[str, Any]] = []
        for dcombo in combinations(drivers, self.num_drivers):
            d_ids = tuple(sorted(d[0] for d in dcombo))
            d_cost = sum(d[1] for d in dcombo)
            if d_cost > budget:
                continue
            d_obj = sum(d[2] for d in dcombo)
            drs_driver, drs_pts = max(((d[0], d[3]) for d in dcombo), key=lambda x: x[1])
            drs_bonus = (self.drs_mult - 1.0) * drs_pts

            for c_ids, c_cost, c_obj in ctor_combos:
                total_cost = d_cost + c_cost
                if total_cost > budget:
                    continue
                teams.append(
                    {
                        "drivers": list(d_ids),
                        "constructors": list(c_ids),
                        "drs_boost": drs_driver,
                        "total_cost": float(total_cost),
                        "objective": float(d_obj + c_obj + drs_bonus),
                    }
                )
        return teams

    def _best_team_under_budget(
        self,
        driver_df: pd.DataFrame,
        constructor_df: pd.DataFrame,
        budget: float,
        current_drivers: set[str] | None = None,
        current_ctors: set[str] | None = None,
        free_transfers: int = 0,
    ) -> TeamRecommendation:
        current_drivers = current_drivers or set()
        current_ctors = current_ctors or set()

        drivers = [
            (
                str(row.asset),
                float(row.price),
                float(row.objective),
                float(row.next_points),
                float(row.next_price_gain),
            )
            for row in driver_df.itertuples(index=False)
        ]
        ctors = [
            (
                str(row.asset),
                float(row.price),
                float(row.objective),
                float(row.next_points),
                float(row.next_price_gain),
            )
            for row in constructor_df.itertuples(index=False)
        ]

        ctor_combos = []
        for combo in combinations(ctors, self.num_constructors):
            ids = tuple(sorted(c[0] for c in combo))
            cost = sum(c[1] for c in combo)
            obj = sum(c[2] for c in combo)
            next_pts = sum(c[3] for c in combo)
            next_gain = sum(c[4] for c in combo)
            ctor_combos.append((ids, cost, obj, next_pts, next_gain))

        best = None
        best_score = float("-inf")

        for dcombo in combinations(drivers, self.num_drivers):
            d_ids = tuple(sorted(d[0] for d in dcombo))
            d_cost = sum(d[1] for d in dcombo)
            if d_cost > budget:
                continue
            d_obj = sum(d[2] for d in dcombo)
            d_next_pts = sum(d[3] for d in dcombo)
            d_next_gain = sum(d[4] for d in dcombo)

            drs_driver, drs_pts = max(((d[0], d[3]) for d in dcombo), key=lambda x: x[1])
            drs_bonus = (self.drs_mult - 1.0) * drs_pts

            for c_ids, c_cost, c_obj, c_next_pts, c_next_gain in ctor_combos:
                total_cost = d_cost + c_cost
                if total_cost > budget:
                    continue

                transfers = self._count_transfers(d_ids, c_ids, current_drivers, current_ctors)
                penalty = max(0, transfers - free_transfers) * self.extra_transfer_cost

                objective = d_obj + c_obj + drs_bonus - penalty
                if objective > best_score:
                    best_score = objective
                    best = TeamRecommendation(
                        round_number=0,
                        drivers=list(d_ids),
                        constructors=list(c_ids),
                        drs_boost=drs_driver,
                        total_cost=float(total_cost),
                        objective_score=float(objective),
                        expected_points_next_race=float(d_next_pts + c_next_pts + drs_bonus - penalty),
                        expected_price_gain_next_race=float(d_next_gain + c_next_gain),
                    )

        if best is None:
            raise ValueError("No feasible team under budget with current constraints.")
        return best

    def recommend_initial_team(
        self,
        predictions: pd.DataFrame,
        season_year: int,
        round_number: int = 1,
        budget: float | None = None,
        lookahead: int | None = None,
        price_weight: float | None = None,
    ) -> TeamRecommendation:
        budget = float(self.budget if budget is None else budget)
        lookahead = int(self.lookahead if lookahead is None else lookahead)
        price_weight = float(self.price_weight if price_weight is None else price_weight)

        driver_prices, constructor_prices = self._initial_prices(predictions, season_year)
        driver_df, constructor_df = self._build_asset_tables(
            predictions=predictions,
            season_year=season_year,
            start_round=round_number,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            lookahead=lookahead,
            price_weight=price_weight,
        )

        rec = self._best_team_under_budget(driver_df, constructor_df, budget)
        rec.round_number = int(round_number)
        return rec

    def recommend_transfers(
        self,
        predictions: pd.DataFrame,
        season_year: int,
        round_number: int,
        current_team: dict[str, Any],
        driver_prices: dict[str, float],
        constructor_prices: dict[str, float],
        lookahead: int | None = None,
        price_weight: float | None = None,
    ) -> dict[str, Any]:
        lookahead = int(self.lookahead if lookahead is None else lookahead)
        price_weight = float(self.price_weight if price_weight is None else price_weight)
        score_col = self._score_column(predictions)

        # `current_team.budget` in config means "bank remaining" (cash leftover
        # after buying the current team). The optimizer's cap is the *total*
        # spending power for the new team = current team's sell-value at present
        # prices + bank. Falls back to the fantasy cap (£100M) if no team is set.
        bank = float(current_team.get("budget", 0.0))
        current_drivers = set(current_team.get("drivers", []))
        current_ctors = set(current_team.get("constructors", []))
        free_transfers = int(current_team.get("free_transfers", self.free_transfers_per_race))
        banked = int(current_team.get("banked_transfers", 0))
        free_allowance = min(
            free_transfers + banked,
            self.free_transfers_per_race + self.max_banked_transfers,
        )

        if current_drivers or current_ctors:
            current_team_value = (
                sum(float(driver_prices.get(d, 0.0)) for d in current_drivers)
                + sum(float(constructor_prices.get(c, 0.0)) for c in current_ctors)
            )
            budget = current_team_value + bank
        else:
            budget = self.budget

        driver_prices, _ = self.price_model.seed_missing_prices_for_round(
            predictions=predictions,
            entity_type="driver",
            season_year=season_year,
            round_number=round_number,
            existing_prices=driver_prices,
            score_col=score_col,
        )
        constructor_prices, _ = self.price_model.seed_missing_prices_for_round(
            predictions=predictions,
            entity_type="constructor",
            season_year=season_year,
            round_number=round_number,
            existing_prices=constructor_prices,
            score_col=score_col,
        )

        driver_df, constructor_df = self._build_asset_tables(
            predictions=predictions,
            season_year=season_year,
            start_round=round_number,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            lookahead=lookahead,
            price_weight=price_weight,
        )

        rec = self._best_team_under_budget(
            driver_df=driver_df,
            constructor_df=constructor_df,
            budget=budget,
            current_drivers=current_drivers,
            current_ctors=current_ctors,
            free_transfers=free_allowance,
        )
        rec.round_number = int(round_number)

        new_drivers = set(rec.drivers)
        new_ctors = set(rec.constructors)
        drivers_out = sorted(current_drivers - new_drivers)
        drivers_in = sorted(new_drivers - current_drivers)
        ctors_out = sorted(current_ctors - new_ctors)
        ctors_in = sorted(new_ctors - current_ctors)
        transfers = len(drivers_in) + len(ctors_in)
        penalty_points = max(0, transfers - free_allowance) * self.extra_transfer_cost

        return {
            "recommendation": rec,
            "drivers_out": drivers_out,
            "drivers_in": drivers_in,
            "constructors_out": ctors_out,
            "constructors_in": ctors_in,
            "num_transfers": transfers,
            "free_transfer_allowance": free_allowance,
            "transfer_penalty_points": penalty_points,
        }

    def _simulate_strategy(
        self,
        predictions: pd.DataFrame,
        season_year: int,
        current_team: dict[str, Any] | None,
        optimize_transfers: bool,
        lookahead: int,
        price_weight: float,
    ) -> dict[str, Any]:
        year_df = predictions[predictions["year"] == season_year].copy()
        if year_df.empty:
            raise ValueError(f"No predictions found for year {season_year}")

        rounds = sorted(year_df["round"].astype(int).unique().tolist())
        score_col = self._score_column(year_df)
        actual_col = "y_true" if "y_true" in year_df.columns else score_col

        driver_prices, constructor_prices = self._initial_prices(predictions, season_year)

        team_state = (current_team or self.config.get("current_team") or {}).copy()
        if not team_state.get("drivers") or not team_state.get("constructors"):
            init_rec = self.recommend_initial_team(
                predictions=predictions,
                season_year=season_year,
                round_number=rounds[0],
                budget=float(team_state.get("budget", self.budget)),
                lookahead=lookahead,
                price_weight=price_weight,
            )
            team_state["drivers"] = init_rec.drivers
            team_state["constructors"] = init_rec.constructors
            team_state["drs_boost"] = init_rec.drs_boost

        team_state["budget"] = float(team_state.get("budget", self.budget))
        team_state["free_transfers"] = int(team_state.get("free_transfers", self.free_transfers_per_race))
        team_state["banked_transfers"] = int(team_state.get("banked_transfers", 0))

        history: list[dict[str, Any]] = []
        total_points = 0.0

        for rnd in rounds:
            driver_prices, _ = self.price_model.seed_missing_prices_for_round(
                predictions=predictions,
                entity_type="driver",
                season_year=season_year,
                round_number=rnd,
                existing_prices=driver_prices,
                score_col=score_col,
            )
            constructor_prices, _ = self.price_model.seed_missing_prices_for_round(
                predictions=predictions,
                entity_type="constructor",
                season_year=season_year,
                round_number=rnd,
                existing_prices=constructor_prices,
                score_col=score_col,
            )

            transfer_info = None
            if rnd != rounds[0] and optimize_transfers:
                transfer_info = self.recommend_transfers(
                    predictions=predictions,
                    season_year=season_year,
                    round_number=rnd,
                    current_team=team_state,
                    driver_prices=driver_prices,
                    constructor_prices=constructor_prices,
                    lookahead=lookahead,
                    price_weight=price_weight,
                )
                rec: TeamRecommendation = transfer_info["recommendation"]
                team_state["drivers"] = rec.drivers
                team_state["constructors"] = rec.constructors
                team_state["drs_boost"] = rec.drs_boost
                penalty = float(transfer_info["transfer_penalty_points"])

                used_free = min(transfer_info["num_transfers"], transfer_info["free_transfer_allowance"])
                remaining_free = max(0, transfer_info["free_transfer_allowance"] - used_free)
                team_state["banked_transfers"] = min(self.max_banked_transfers, remaining_free)
                team_state["free_transfers"] = self.free_transfers_per_race
            else:
                penalty = 0.0

            round_df = year_df[year_df["round"] == rnd]
            d_actual_map = round_df.groupby("driver_code")[actual_col].mean().to_dict()
            d_pred_map = round_df.groupby("driver_code")[score_col].mean().to_dict()
            c_actual_map = round_df.groupby("constructor_id")[actual_col].sum().to_dict()
            c_pred_map = round_df.groupby("constructor_id")[score_col].sum().to_dict()

            drivers = team_state["drivers"]
            ctors = team_state["constructors"]
            drv_points = sum(float(d_actual_map.get(d, 0.0)) for d in drivers)
            c_points = sum(float(c_actual_map.get(c, 0.0)) for c in ctors)
            drs_driver = str(team_state.get("drs_boost") or max(drivers, key=lambda x: d_pred_map.get(x, 0.0)))
            drs_points = float(d_actual_map.get(drs_driver, 0.0)) * (self.drs_mult - 1.0)
            round_points = drv_points + c_points + drs_points - penalty
            total_points += round_points

            held_driver_pred = {d: float(d_pred_map.get(d, 0.0)) for d in drivers}
            held_ctor_pred = {c: float(c_pred_map.get(c, 0.0)) for c in ctors}
            updates, driver_prices, constructor_prices = self.price_model.simulate_round(
                round_number=rnd,
                driver_points=held_driver_pred,
                constructor_points=held_ctor_pred,
                driver_prices=driver_prices,
                constructor_prices=constructor_prices,
            )

            budget_delta = 0.0
            for u in updates:
                if u.entity_type == "driver" and u.asset_id in drivers:
                    budget_delta += u.delta
                if u.entity_type == "constructor" and u.asset_id in ctors:
                    budget_delta += u.delta
            team_state["budget"] = float(team_state.get("budget", self.budget)) + budget_delta

            round_out = {
                "round": int(rnd),
                "drivers": list(drivers),
                "constructors": list(ctors),
                "drs_boost": drs_driver,
                "round_points": float(round_points),
                "cumulative_points": float(total_points),
                "round_budget_delta": float(budget_delta),
                "budget_after_round": float(team_state["budget"]),
                "transfer_penalty": float(penalty),
            }
            if transfer_info is not None:
                round_out["num_transfers"] = int(transfer_info["num_transfers"])
                round_out["drivers_in"] = transfer_info["drivers_in"]
                round_out["drivers_out"] = transfer_info["drivers_out"]
                round_out["constructors_in"] = transfer_info["constructors_in"]
                round_out["constructors_out"] = transfer_info["constructors_out"]
            history.append(round_out)

        return {
            "season_year": int(season_year),
            "total_points": float(total_points),
            "ending_budget": float(team_state.get("budget", self.budget)),
            "final_team": {
                "drivers": list(team_state.get("drivers", [])),
                "constructors": list(team_state.get("constructors", [])),
                "drs_boost": team_state.get("drs_boost"),
            },
            "history": history,
        }

    def evaluate_prediction_decision_metrics(
        self,
        predictions: pd.DataFrame,
        season_year: int,
    ) -> dict[str, Any]:
        """Prediction quality metrics aligned with decision usefulness."""
        year_df = predictions[predictions["year"] == season_year].copy()
        if year_df.empty:
            return {}

        score_col = self._score_column(year_df)
        if "y_true" not in year_df.columns:
            return {"note": "y_true not available; decision metrics skipped"}

        rounds = sorted(year_df["round"].astype(int).unique().tolist())
        k = self.num_drivers
        rank_corrs: list[float] = []
        topk_hits: list[float] = []
        drs_hits: list[float] = []
        ctor_top2_hits: list[float] = []

        for rnd in rounds:
            rdf = year_df[year_df["round"] == rnd].copy()
            if rdf.empty:
                continue

            d_pred = rdf.groupby("driver_code")[score_col].mean()
            d_true = rdf.groupby("driver_code")["y_true"].mean()
            aligned = pd.concat([d_pred.rename("pred"), d_true.rename("true")], axis=1).dropna()
            if len(aligned) >= 2:
                corr = aligned["pred"].corr(aligned["true"], method="spearman")
                if corr is not None and pd.notna(corr):
                    rank_corrs.append(float(corr))

                pred_topk = set(aligned.sort_values("pred", ascending=False).head(k).index.tolist())
                true_topk = set(aligned.sort_values("true", ascending=False).head(k).index.tolist())
                if pred_topk:
                    topk_hits.append(float(len(pred_topk & true_topk) / len(pred_topk)))

                pred_top1 = aligned.sort_values("pred", ascending=False).index[0]
                true_top1 = aligned.sort_values("true", ascending=False).index[0]
                drs_hits.append(1.0 if pred_top1 == true_top1 else 0.0)

            c_pred = rdf.groupby("constructor_id")[score_col].sum()
            c_true = rdf.groupby("constructor_id")["y_true"].sum()
            c_aligned = pd.concat([c_pred.rename("pred"), c_true.rename("true")], axis=1).dropna()
            if len(c_aligned) >= 2:
                pred_top2 = set(c_aligned.sort_values("pred", ascending=False).head(2).index.tolist())
                true_top2 = set(c_aligned.sort_values("true", ascending=False).head(2).index.tolist())
                if pred_top2:
                    ctor_top2_hits.append(float(len(pred_top2 & true_top2) / len(pred_top2)))

        out: dict[str, Any] = {
            "driver_rank_spearman_mean": float(sum(rank_corrs) / len(rank_corrs)) if rank_corrs else None,
            "driver_top5_hit_rate": float(sum(topk_hits) / len(topk_hits)) if topk_hits else None,
            "drs_top1_hit_rate": float(sum(drs_hits) / len(drs_hits)) if drs_hits else None,
            "constructor_top2_hit_rate": float(sum(ctor_top2_hits) / len(ctor_top2_hits)) if ctor_top2_hits else None,
        }

        if "dnf_prob" in year_df.columns and "is_dnf" in year_df.columns:
            d = year_df[["dnf_prob", "is_dnf"]].dropna()
            if not d.empty:
                brier = ((d["dnf_prob"].astype(float) - d["is_dnf"].astype(float)) ** 2).mean()
                out["dnf_brier_score"] = float(brier)
            else:
                out["dnf_brier_score"] = None

        return out

    def evaluate_initial_team_kpi(
        self,
        predictions: pd.DataFrame,
        season_year: int,
        num_alternatives: int = 250,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Main KPI: chosen initial team vs feasible alternatives under 100M budget.

        Every alternative in this KPI is constrained to <= starting fantasy budget.
        """
        budget = float(self.config.get("fantasy", {}).get("budget", self.budget))

        driver_prices, constructor_prices = self._initial_prices(predictions, season_year)
        driver_df, constructor_df = self._build_asset_tables(
            predictions=predictions,
            season_year=season_year,
            start_round=1,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            lookahead=self.lookahead,
            price_weight=self.price_weight,
        )
        feasible = self._enumerate_feasible_teams(driver_df, constructor_df, budget)
        if not feasible:
            return {"note": "No feasible opening teams under budget"}

        chosen = self.recommend_initial_team(
            predictions=predictions,
            season_year=season_year,
            round_number=1,
            budget=budget,
            lookahead=self.lookahead,
            price_weight=self.price_weight,
        )
        chosen_key = (
            tuple(sorted(chosen.drivers)),
            tuple(sorted(chosen.constructors)),
        )

        chosen_bt = self._simulate_strategy(
            predictions=predictions,
            season_year=season_year,
            current_team={
                "drivers": list(chosen.drivers),
                "constructors": list(chosen.constructors),
                "drs_boost": chosen.drs_boost,
                "budget": budget,
                "free_transfers": self.free_transfers_per_race,
                "banked_transfers": 0,
            },
            optimize_transfers=True,
            lookahead=self.lookahead,
            price_weight=self.price_weight,
        )
        chosen_points = float(chosen_bt["total_points"])

        alt_pool = [
            t for t in feasible
            if (tuple(sorted(t["drivers"])), tuple(sorted(t["constructors"]))) != chosen_key
        ]
        rng = random.Random(random_seed)
        if len(alt_pool) > num_alternatives:
            alt_sorted = sorted(alt_pool, key=lambda x: x["objective"], reverse=True)
            n_top = min(max(20, num_alternatives // 3), len(alt_sorted))
            top = alt_sorted[:n_top]
            rest = alt_sorted[n_top:]
            n_rand = max(0, num_alternatives - len(top))
            sampled = top + (rng.sample(rest, n_rand) if n_rand and len(rest) >= n_rand else rest[:n_rand])
        else:
            sampled = alt_pool

        alt_points: list[float] = []
        for alt in sampled:
            alt_bt = self._simulate_strategy(
                predictions=predictions,
                season_year=season_year,
                current_team={
                    "drivers": list(alt["drivers"]),
                    "constructors": list(alt["constructors"]),
                    "drs_boost": str(alt["drs_boost"]),
                    "budget": budget,
                    "free_transfers": self.free_transfers_per_race,
                    "banked_transfers": 0,
                },
                optimize_transfers=True,
                lookahead=self.lookahead,
                price_weight=self.price_weight,
            )
            alt_points.append(float(alt_bt["total_points"]))

        if not alt_points:
            return {
                "chosen_team_points": chosen_points,
                "chosen_team_cost": float(chosen.total_cost),
                "num_feasible_alternatives_tested": 0,
                "note": "No alternative teams available",
            }

        better = sum(1 for p in alt_points if p < chosen_points)
        equal = sum(1 for p in alt_points if p == chosen_points)
        percentile = (better + 0.5 * equal) / len(alt_points)
        best_alt = max(alt_points)
        avg_alt = sum(alt_points) / len(alt_points)

        return {
            "starting_budget_constraint": budget,
            "chosen_team": {
                "drivers": list(chosen.drivers),
                "constructors": list(chosen.constructors),
                "drs_boost": chosen.drs_boost,
                "cost": float(chosen.total_cost),
            },
            "chosen_team_points": chosen_points,
            "num_feasible_total": int(len(feasible)),
            "num_feasible_alternatives_tested": int(len(alt_points)),
            "percentile_vs_feasible_alternatives": float(percentile),
            "avg_alternative_points": float(avg_alt),
            "best_alternative_points": float(best_alt),
            "delta_vs_avg_alternative": float(chosen_points - avg_alt),
            "delta_vs_best_alternative": float(chosen_points - best_alt),
        }

    def backtest_season(
        self,
        predictions: pd.DataFrame,
        season_year: int,
        current_team: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        optimized = self._simulate_strategy(
            predictions=predictions,
            season_year=season_year,
            current_team=current_team,
            optimize_transfers=True,
            lookahead=self.lookahead,
            price_weight=self.price_weight,
        )
        hold = self._simulate_strategy(
            predictions=predictions,
            season_year=season_year,
            current_team=current_team,
            optimize_transfers=False,
            lookahead=self.lookahead,
            price_weight=self.price_weight,
        )
        points_only = self._simulate_strategy(
            predictions=predictions,
            season_year=season_year,
            current_team=current_team,
            optimize_transfers=True,
            lookahead=1,
            price_weight=0.0,
        )

        optimized["baselines"] = {
            "hold_team": {
                "total_points": hold["total_points"],
                "ending_budget": hold["ending_budget"],
            },
            "points_only": {
                "total_points": points_only["total_points"],
                "ending_budget": points_only["ending_budget"],
            },
            "deltas_vs_hold": {
                "points": optimized["total_points"] - hold["total_points"],
                "ending_budget": optimized["ending_budget"] - hold["ending_budget"],
            },
            "deltas_vs_points_only": {
                "points": optimized["total_points"] - points_only["total_points"],
                "ending_budget": optimized["ending_budget"] - points_only["ending_budget"],
            },
        }
        if self.initial_kpi_enabled:
            optimized["initial_team_kpi"] = self.evaluate_initial_team_kpi(
                predictions=predictions,
                season_year=season_year,
                num_alternatives=self.initial_kpi_num_alternatives,
                random_seed=self.initial_kpi_random_seed,
            )
        else:
            optimized["initial_team_kpi"] = {"enabled": False}
        optimized["prediction_decision_metrics"] = self.evaluate_prediction_decision_metrics(
            predictions=predictions,
            season_year=season_year,
        )
        return optimized
