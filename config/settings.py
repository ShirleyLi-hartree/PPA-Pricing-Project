"""
Configuration settings for PPA Pricing system.

Contains market configuration, region definitions, and other system-wide settings.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class JapanRegion(Enum):
    """Japanese electricity market regions."""
    HOKKAIDO = 'hokkaido'
    TOHOKU = 'tohoku'
    TOKYO = 'tokyo'
    CHUBU = 'chubu'
    HOKURIKU = 'hokuriku'
    KANSAI = 'kansai'
    CHUGOKU = 'chugoku'
    SHIKOKU = 'shikoku'
    KYUSHU = 'kyushu'
    OKINAWA = 'okinawa'
    SYSTEM = 'system'  # System-wide price


@dataclass
class MarketConfig:
    """Configuration for electricity market settings."""
    
    # Trading periods per day (30-minute intervals)
    periods_per_day: int = 48
    
    # Peak hours definition
    peak_start_hour: int = 8
    peak_end_hour: int = 22
    
    # Price units
    price_unit: str = 'JPY/kWh'
    
    # Default region
    default_region: JapanRegion = JapanRegion.TOKYO
    
    # Data frequency in minutes
    data_frequency_minutes: int = 30
    
    # Timezone
    timezone: str = 'Asia/Tokyo'


@dataclass  
class ModelConfig:
    """Configuration for forecasting models."""
    
    # Default training window in years
    default_train_years: int = 3
    
    # Default validation window in months
    default_val_months: int = 6
    
    # Default test window in months
    default_test_months: int = 1
    
    # Default step size in months
    default_step_month: int = 1
    
    # Confidence interval level
    confidence_level: float = 0.90


# Default configurations
DEFAULT_MARKET_CONFIG = MarketConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
