"""
PPA Pricing Engine - Calculates fixed electricity prices for PPA contracts.

This module takes price forecasts and calculates fixed prices for different
supply scenarios, considering the contracted volume distribution.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from config.settings import JapanRegion
from src.core.pricing.config import PPAPricingConfig
from src.core.profiles.supply_profile import SupplyProfileManager

logger = logging.getLogger(__name__)


@dataclass
class PPAPricingResult:
    """
    Container for PPA pricing results.
    
    Attributes:
        scenario_key: Scenario identifier
        scenario_name: Human-readable scenario name
        contract_period: Contract period string
        fixed_price: Calculated fixed price (JPY/kWh)
        total_volume_mwh: Total contracted volume
        total_revenue: Expected total revenue (JPY)
        avg_spot_price: Average spot price during supply periods
        capture_ratio: Fixed price / Average spot price
        num_supply_periods: Number of supply periods
        price_lower_bound: Lower bound of price confidence interval
        price_upper_bound: Upper bound of price confidence interval
    """
    scenario_key: str
    scenario_name: str
    contract_period: str
    fixed_price: float
    total_volume_mwh: float
    total_revenue: float
    avg_spot_price: float
    capture_ratio: float
    num_supply_periods: int
    price_lower_bound: Optional[float] = None
    price_upper_bound: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenario_key': self.scenario_key,
            'scenario_name': self.scenario_name,
            'contract_period': self.contract_period,
            'fixed_price_jpy_kwh': self.fixed_price,
            'total_volume_mwh': self.total_volume_mwh,
            'total_revenue_jpy': self.total_revenue,
            'avg_spot_price': self.avg_spot_price,
            'capture_ratio': self.capture_ratio,
            'num_supply_periods': self.num_supply_periods,
            'price_lower_bound': self.price_lower_bound,
            'price_upper_bound': self.price_upper_bound,
        }


class PPAPricingEngine:
    """
    PPA Pricing Engine - Calculates fixed prices for different supply scenarios.
    
    The engine takes price forecasts and calculates the fixed price that would
    result in equivalent expected value over the contract period.
    
    Pricing Formula:
        Fixed_Price = Σ(Predicted_Price_t × Volume_t) / Σ(Volume_t)
    
    For uniform volume distribution:
        Fixed_Price = Σ(Predicted_Price_t) / N_periods (simple average)
    
    This is equivalent to the capture price concept used in renewable energy.
    """
    
    def __init__(self,
                 config: PPAPricingConfig = None,
                 region: JapanRegion = None,
                 forecast_months: int = None,
                 total_volume_mwh: float = None):
        """
        Initialize pricing engine.
        
        Args:
            config: PPAPricingConfig object (overrides other parameters)
            region: Target region (used if config not provided)
            forecast_months: Forecast months (used if config not provided)
            total_volume_mwh: Total volume (used if config not provided)
        """
        if config is not None:
            self.config = config
        else:
            self.config = PPAPricingConfig(
                region=region or JapanRegion.TOKYO,
                forecast_months=forecast_months or 3,
                total_volume_mwh=total_volume_mwh or 2.0
            )
        
        self.supply_manager = SupplyProfileManager(
            peak_hour_start=self.config.peak_hour_start,
            peak_hour_end=self.config.peak_hour_end
        )
    
    def calculate_ppa_price(self,
                            forecast: pd.DataFrame,
                            scenario_key: str,
                            datetime_col: str = 'datetime',
                            price_col: str = 'forecast',
                            lower_col: str = None,
                            upper_col: str = None) -> PPAPricingResult:
        """
        Calculate PPA fixed price for a specific supply scenario.
        
        Args:
            forecast: DataFrame with price forecasts
            scenario_key: Scenario identifier ('24_7', 'peak_only', 'off_peak')
            datetime_col: Name of datetime column
            price_col: Name of forecast price column
            lower_col: Name of lower bound column (optional)
            upper_col: Name of upper bound column (optional)
            
        Returns:
            PPAPricingResult with calculated price
        """
        # Use full forecast period (no contract filtering)
        df = forecast.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        if len(df) == 0:
            raise ValueError("No forecast data provided")
        
        # Get forecast period for reporting
        forecast_start = df[datetime_col].min()
        forecast_end = df[datetime_col].max()
        logger.info(f"Calculating PPA price for period: {forecast_start.date()} to {forecast_end.date()}")
        
        # Get scenario info
        scenario = self.supply_manager.get_scenario(scenario_key)
        
        # Apply scenario filter
        scenario_mask = self.supply_manager.get_scenario_mask(
            df, scenario_key, datetime_col
        )
        supply_df = df[scenario_mask].copy()
        
        if len(supply_df) == 0:
            raise ValueError(f"No supply periods for scenario {scenario_key}")
        
        num_periods = len(supply_df)
        
        # Calculate volume per period (uniform distribution)
        # Convert MWh to volume per 30-minute period
        volume_per_period_mwh = self.config.total_volume_mwh / num_periods
        
        # Calculate fixed price (weighted average)
        # For uniform distribution, this is just the simple average
        fixed_price = supply_df[price_col].mean()
        
        # Calculate average spot price (same as fixed_price for uniform distribution)
        avg_spot_price = supply_df[price_col].mean()
        
        # Capture ratio
        capture_ratio = fixed_price / avg_spot_price if avg_spot_price > 0 else 1.0
        
        # Total revenue (price in JPY/kWh, volume in MWh, convert to kWh)
        total_revenue = fixed_price * self.config.total_volume_mwh * 1000
        
        # Confidence interval (if bounds provided)
        price_lower = None
        price_upper = None
        
        if lower_col and lower_col in supply_df.columns:
            price_lower = supply_df[lower_col].mean()
        if upper_col and upper_col in supply_df.columns:
            price_upper = supply_df[upper_col].mean()
        
        logger.info(f"Scenario {scenario.name}: Fixed Price = {fixed_price:.3f} JPY/kWh, "
                    f"Periods = {num_periods}, Capture Ratio = {capture_ratio:.3f}")
        
        # Get forecast period for result
        forecast_period = f"{forecast_start.date()} to {forecast_end.date()}"
        
        return PPAPricingResult(
            scenario_key=scenario_key,
            scenario_name=scenario.name,
            contract_period=forecast_period,
            fixed_price=fixed_price,
            total_volume_mwh=self.config.total_volume_mwh,
            total_revenue=total_revenue,
            avg_spot_price=avg_spot_price,
            capture_ratio=capture_ratio,
            num_supply_periods=num_periods,
            price_lower_bound=price_lower,
            price_upper_bound=price_upper,
            metadata={
                'region': self.config.region.value,
                'volume_per_period_mwh': volume_per_period_mwh,
            }
        )
    
    def calculate_all_scenarios(self,
                                 forecast: pd.DataFrame,
                                 datetime_col: str = 'datetime',
                                 price_col: str = 'forecast',
                                 lower_col: str = None,
                                 upper_col: str = None) -> Dict[str, PPAPricingResult]:
        """
        Calculate PPA prices for all supply scenarios.
        
        Args:
            forecast: DataFrame with price forecasts
            datetime_col: Name of datetime column
            price_col: Name of forecast price column
            lower_col: Name of lower bound column (optional)
            upper_col: Name of upper bound column (optional)
            
        Returns:
            Dictionary mapping scenario key to PPAPricingResult
        """
        results = {}
        
        for scenario_key in self.supply_manager.get_all_scenarios().keys():
            try:
                result = self.calculate_ppa_price(
                    forecast=forecast,
                    scenario_key=scenario_key,
                    datetime_col=datetime_col,
                    price_col=price_col,
                    lower_col=lower_col,
                    upper_col=upper_col
                )
                results[scenario_key] = result
            except Exception as e:
                logger.error(f"Error calculating price for {scenario_key}: {str(e)}")
        
        return results
    
    def generate_pricing_report(self,
                                 results: Dict[str, PPAPricingResult]) -> pd.DataFrame:
        """
        Generate pricing report as DataFrame.
        
        Args:
            results: Dictionary of pricing results
            
        Returns:
            DataFrame with pricing summary
        """
        records = [result.to_dict() for result in results.values()]
        return pd.DataFrame(records)
    
    def generate_pricing_report_text(self,
                                      results: Dict[str, PPAPricingResult]) -> str:
        """
        Generate formatted pricing report as text.
        
        Args:
            results: Dictionary of pricing results
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"PPA PRICING REPORT ({self.config.region.value.upper()} Region)")
        lines.append("=" * 80)
        lines.append("")
        
        # Get forecast period from first result
        first_result = list(results.values())[0]
        lines.append(f"Forecast Period: {first_result.contract_period}")
        lines.append(f"Forecast Months: {self.config.forecast_months} months")
        lines.append(f"Total Volume: {self.config.total_volume_mwh} MWh")
        lines.append(f"Peak Hours: {self.config.peak_hour_start}:00 - {self.config.peak_hour_end}:00")
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"{'Scenario':<20} {'Fixed Price':<15} {'Periods':<12} {'Capture':<10} {'Revenue (JPY)':<15}")
        lines.append("-" * 80)
        
        for result in results.values():
            price_str = f"{result.fixed_price:.3f} JPY/kWh"
            capture_str = f"{result.capture_ratio:.3f}"
            revenue_str = f"{result.total_revenue:,.0f}"
            
            lines.append(f"{result.scenario_name:<20} {price_str:<15} {result.num_supply_periods:<12} {capture_str:<10} {revenue_str:<15}")
            
            if result.price_lower_bound and result.price_upper_bound:
                ci_str = f"  └─ 90% CI: [{result.price_lower_bound:.3f}, {result.price_upper_bound:.3f}]"
                lines.append(ci_str)
        
        lines.append("-" * 80)
        lines.append("")
        lines.append("Notes:")
        lines.append("- Fixed Price: Weighted average forecast price during supply periods")
        lines.append("- Capture Ratio: Fixed Price / Average Spot Price (1.0 = break-even)")
        lines.append("- Revenue: Fixed Price × Total Volume × 1000 (kWh)")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def compare_scenarios(self,
                           results: Dict[str, PPAPricingResult]) -> Dict[str, Any]:
        """
        Compare pricing across scenarios.
        
        Args:
            results: Dictionary of pricing results
            
        Returns:
            Comparison summary
        """
        if not results:
            return {}
        
        prices = {k: v.fixed_price for k, v in results.items()}
        
        return {
            'highest_price_scenario': max(prices, key=prices.get),
            'lowest_price_scenario': min(prices, key=prices.get),
            'price_range': max(prices.values()) - min(prices.values()),
            'peak_vs_offpeak_spread': (
                results.get('peak_only', results.get('24_7')).fixed_price -
                results.get('off_peak', results.get('24_7')).fixed_price
            ) if 'peak_only' in results and 'off_peak' in results else None,
            'scenarios': prices
        }
