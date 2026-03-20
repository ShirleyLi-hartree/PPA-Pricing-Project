"""
Configuration module for PPA pricing.

Defines configuration classes for PPA contracts, supply scenarios,
and regional settings.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import JapanRegion


class SupplyScenario(Enum):
    """Supply scenario types for PPA contracts."""
    FULL_DAY = "24_7"  # 24/7 continuous supply
    PEAK_ONLY = "peak_only"  # Peak hours only (8:00-22:00)
    OFF_PEAK_ONLY = "off_peak"  # Off-peak hours (22:00-8:00 + weekends)


@dataclass
class PPAPricingConfig:
    """
    Configuration for PPA pricing model.

    Attributes:
        region: Target Japanese electricity market region
        forecast_months: Number of months to forecast from latest data (default: 3)
        total_volume_mwh: Total contracted supply volume in MWh
        train_window_years: Training window size in years
        scenarios: List of supply scenarios to price
    """
    # Region
    region: JapanRegion = JapanRegion.TOKYO
    
    # Forecast parameters
    forecast_months: int = 3  # Forecast N months from latest data
    total_volume_mwh: float = 2.0

    # Rolling window parameters
    train_window_years: int = 3
    predict_horizon_years: int = 1  # For validation only

    # Supply scenarios to calculate
    scenarios: List[SupplyScenario] = field(default_factory=lambda: [
        SupplyScenario.FULL_DAY,
        SupplyScenario.PEAK_ONLY,
        SupplyScenario.OFF_PEAK_ONLY,
    ])

    # Peak hours definition (8:00-22:00)
    peak_hour_start: int = 8
    peak_hour_end: int = 22

    # Data parameters
    datetime_col: str = 'datetime'
    price_col: str = 'price'

    # Model parameters
    confidence_level: float = 0.90

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.train_window_years < 1:
            raise ValueError("train_window_years must be at least 1")
        if self.predict_horizon_years < 1:
            raise ValueError("predict_horizon_years must be at least 1")
        if self.forecast_months < 1:
            raise ValueError("forecast_months must be at least 1")
        if self.total_volume_mwh <= 0:
            raise ValueError("total_volume_mwh must be positive")
        if not 0 <= self.peak_hour_start < 24:
            raise ValueError("peak_hour_start must be between 0 and 23")
        if not 0 < self.peak_hour_end <= 24:
            raise ValueError("peak_hour_end must be between 1 and 24")
        if self.peak_hour_start >= self.peak_hour_end:
            raise ValueError("peak_hour_start must be less than peak_hour_end")

    @property
    def periods_per_day(self) -> int:
        """Number of 30-minute periods per day."""
        return 48

    @property
    def periods_per_year(self) -> int:
        """Approximate number of periods per year."""
        return 365 * self.periods_per_day

    @property
    def train_window_periods(self) -> int:
        """Training window size in number of periods."""
        return self.train_window_years * self.periods_per_year

    @property
    def predict_horizon_periods(self) -> int:
        """Prediction horizon in number of periods."""
        return self.predict_horizon_years * self.periods_per_year