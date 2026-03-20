"""
Monthly fundamental clearing price engine.

This module consumes standardized monthly market inputs and plant-level data
from the LiveSheet extractor and produces monthly clearing prices that can be
compared with the forecast-driven pricing workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import JapanRegion


THERMAL_TECHS = {"gas", "coal", "oil"}


@dataclass
class FundamentalRunResult:
    monthly_prices: pd.DataFrame
    monthly_stacks: pd.DataFrame


class MonthlyFundamentalEngine:
    """Plant-level monthly clearing price engine."""

    def __init__(self, region: JapanRegion):
        self.region = region

    def run(
        self,
        plant_master: pd.DataFrame,
        monthly_market_inputs: pd.DataFrame,
    ) -> FundamentalRunResult:
        if plant_master.empty:
            raise ValueError("plant_master is empty")
        if monthly_market_inputs.empty:
            raise ValueError("monthly_market_inputs is empty")

        market_inputs = monthly_market_inputs.sort_values("month").reset_index(drop=True)
        relevant_plants = plant_master[
            plant_master["market_side"].str.lower() == market_inputs["market_side"].iloc[0].lower()
        ].copy()

        if relevant_plants.empty:
            raise ValueError(
                f"No plants found for market side '{market_inputs['market_side'].iloc[0]}'"
            )

        tech_efficiency_reference = (
            relevant_plants.groupby("technology")["efficiency"]
            .mean()
            .replace(0, pd.NA)
            .to_dict()
        )

        monthly_results = []
        stack_results = []

        for monthly_input in market_inputs.to_dict("records"):
            month = pd.Timestamp(monthly_input["month"]).normalize()
            active = self._build_active_stack(
                plants=relevant_plants,
                monthly_input=monthly_input,
                tech_efficiency_reference=tech_efficiency_reference,
                month=month,
            )
            if active.empty:
                continue

            demand_mw = float(monthly_input["demand_mw"])
            active = active.sort_values(
                ["srmc_jpy_kwh", "effective_capacity_mw", "plant_id"],
                ascending=[True, False, True],
            ).reset_index(drop=True)
            active["cumulative_capacity_mw"] = active["effective_capacity_mw"].cumsum()

            marginal_candidates = active[active["cumulative_capacity_mw"] >= demand_mw]
            if marginal_candidates.empty:
                marginal_unit = active.iloc[-1]
                clearing_price = max(float(marginal_unit["srmc_jpy_kwh"]) * 1.25, 100.0)
                scarcity = True
            else:
                marginal_unit = marginal_candidates.iloc[0]
                clearing_price = float(marginal_unit["srmc_jpy_kwh"])
                scarcity = False

            total_capacity = float(active["effective_capacity_mw"].sum())
            reserve_margin = ((total_capacity - demand_mw) / demand_mw) if demand_mw else 0.0

            monthly_results.append(
                {
                    "month": month,
                    "region": self.region.value,
                    "market_side": monthly_input["market_side"],
                    "contract_type": monthly_input["contract_type"],
                    "weather_year": monthly_input["weather_year"],
                    "demand_mw": demand_mw,
                    "available_capacity_mw": total_capacity,
                    "clearing_price_jpy_kwh": clearing_price,
                    "marginal_plant_id": marginal_unit["plant_id"],
                    "marginal_technology": marginal_unit["technology"],
                    "marginal_cost_jpy_kwh": float(marginal_unit["srmc_jpy_kwh"]),
                    "reserve_margin": reserve_margin,
                    "scarcity_flag": scarcity,
                }
            )

            active["month"] = month
            active["region"] = self.region.value
            active["demand_mw"] = demand_mw
            stack_results.append(active)

        monthly_df = pd.DataFrame(monthly_results).sort_values("month").reset_index(drop=True)
        stack_df = (
            pd.concat(stack_results, ignore_index=True)
            if stack_results
            else pd.DataFrame()
        )

        return FundamentalRunResult(monthly_prices=monthly_df, monthly_stacks=stack_df)

    def _build_active_stack(
        self,
        plants: pd.DataFrame,
        monthly_input: Dict[str, object],
        tech_efficiency_reference: Dict[str, float],
        month: pd.Timestamp,
    ) -> pd.DataFrame:
        month_end = month + pd.offsets.MonthEnd(0)

        active = plants[
            (plants["start_date"].isna() | (plants["start_date"] <= month_end))
            & (plants["end_date"].isna() | (plants["end_date"] >= month))
        ].copy()

        active["availability"] = active["technology"].map(
            lambda tech: float(monthly_input.get(f"availability_{tech}", 1.0) or 1.0)
        )
        active["effective_capacity_mw"] = active["capacity_mw"] * active["availability"]

        active["srmc_jpy_kwh"] = active.apply(
            lambda row: self._compute_plant_cost(
                technology=row["technology"],
                efficiency=row["efficiency"],
                tech_efficiency_reference=tech_efficiency_reference,
                monthly_input=monthly_input,
            ),
            axis=1,
        )

        return active[active["effective_capacity_mw"] > 0].copy()

    def _compute_plant_cost(
        self,
        technology: str,
        efficiency: float,
        tech_efficiency_reference: Dict[str, float],
        monthly_input: Dict[str, object],
    ) -> float:
        base_cost_mwh = float(monthly_input.get(f"cost_jpy_mwh_{technology}", 0.0) or 0.0)
        if base_cost_mwh <= 0:
            vom_fallback = float(monthly_input.get(f"vom_jpy_mwh_{technology}", 0.0) or 0.0)
            base_cost_mwh = vom_fallback

        adjusted_cost_mwh = base_cost_mwh
        reference_efficiency = tech_efficiency_reference.get(technology)

        if (
            technology in THERMAL_TECHS
            and reference_efficiency
            and efficiency
            and efficiency > 0
        ):
            adjusted_cost_mwh = base_cost_mwh * (reference_efficiency / efficiency)

        return adjusted_cost_mwh / 1000.0
