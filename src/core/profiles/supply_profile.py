"""
Supply Profile Manager - Defines supply scenarios for PPA contracts.

This module defines different power supply patterns:
- 24/7 (Full Day): Continuous supply across all periods
- Peak Only: Supply during peak hours (8:00-22:00)
- Off-Peak Only: Supply during off-peak hours (22:00-8:00 + weekends)
"""
from dataclasses import dataclass
from typing import Dict, Callable, List
import pandas as pd
import numpy as np


@dataclass
class SupplyScenario:
    """
    Definition of a power supply scenario.
    
    Attributes:
        key: Unique identifier for the scenario
        name: Human-readable name
        description: Detailed description
        filter_func: Function that returns boolean mask for applicable periods
    """
    key: str
    name: str
    description: str
    filter_func: Callable[[pd.DataFrame, str, int, int], np.ndarray]


class SupplyProfileManager:
    """
    Supply Profile Manager - Manages different supply scenarios.
    
    Defines three standard supply scenarios for PPA pricing:
    1. 24/7: Continuous power supply (all 48 half-hourly periods)
    2. Peak Only: Supply during daytime hours (8:00-22:00)
    3. Off-Peak Only: Supply during night (22:00-8:00) and weekends
    
    The peak/off-peak boundaries are configurable.
    """
    
    def __init__(self,
                 peak_hour_start: int = 8,
                 peak_hour_end: int = 22):
        """
        Initialize supply profile manager.
        
        Args:
            peak_hour_start: Start of peak hours (inclusive), default 8
            peak_hour_end: End of peak hours (exclusive), default 22
        """
        self.peak_hour_start = peak_hour_start
        self.peak_hour_end = peak_hour_end
        
        # Define standard scenarios
        self._scenarios = self._create_standard_scenarios()
    
    def _create_standard_scenarios(self) -> Dict[str, SupplyScenario]:
        """Create standard supply scenarios."""
        return {
            '24_7': SupplyScenario(
                key='24_7',
                name='24/7 Full Day',
                description='Continuous power supply across all 48 half-hourly periods per day',
                filter_func=self._scenario_24_7
            ),
            'peak_only': SupplyScenario(
                key='peak_only',
                name='Peak Hours Only',
                description=f'Power supply during peak hours ({self.peak_hour_start}:00-{self.peak_hour_end}:00)',
                filter_func=self._scenario_peak_only
            ),
            'off_peak': SupplyScenario(
                key='off_peak',
                name='Off-Peak Hours Only',
                description=f'Power supply during off-peak hours ({self.peak_hour_end}:00-{self.peak_hour_start}:00) and weekends',
                filter_func=self._scenario_off_peak
            )
        }
    
    def _scenario_24_7(self,
                       df: pd.DataFrame,
                       datetime_col: str = 'datetime',
                       peak_start: int = None,
                       peak_end: int = None) -> np.ndarray:
        """
        24/7 Full Day scenario - all periods included.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            peak_start: Ignored for this scenario
            peak_end: Ignored for this scenario
            
        Returns:
            Boolean array with all True values
        """
        return np.ones(len(df), dtype=bool)
    
    def _scenario_peak_only(self,
                            df: pd.DataFrame,
                            datetime_col: str = 'datetime',
                            peak_start: int = None,
                            peak_end: int = None) -> np.ndarray:
        """
        Peak Hours Only scenario - daytime hours only.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            peak_start: Start of peak hours (uses default if None)
            peak_end: End of peak hours (uses default if None)
            
        Returns:
            Boolean array indicating peak hour periods
        """
        peak_start = peak_start or self.peak_hour_start
        peak_end = peak_end or self.peak_hour_end
        
        dt = pd.to_datetime(df[datetime_col])
        hour = dt.dt.hour
        
        # Peak hours: 8:00-22:00 (or configured range)
        return ((hour >= peak_start) & (hour < peak_end)).values
    
    def _scenario_off_peak(self,
                           df: pd.DataFrame,
                           datetime_col: str = 'datetime',
                           peak_start: int = None,
                           peak_end: int = None) -> np.ndarray:
        """
        Off-Peak Hours Only scenario - night hours and weekends.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            peak_start: Start of peak hours (uses default if None)
            peak_end: End of peak hours (uses default if None)
            
        Returns:
            Boolean array indicating off-peak periods
        """
        peak_start = peak_start or self.peak_hour_start
        peak_end = peak_end or self.peak_hour_end
        
        dt = pd.to_datetime(df[datetime_col])
        hour = dt.dt.hour
        is_weekend = dt.dt.dayofweek >= 5
        
        # Night hours: 22:00-8:00 (or outside peak range)
        is_night = (hour >= peak_end) | (hour < peak_start)
        
        # Off-peak: night hours OR weekends
        return (is_weekend | is_night).values
    
    def get_scenario(self, key: str) -> SupplyScenario:
        """
        Get a supply scenario by key.
        
        Args:
            key: Scenario key ('24_7', 'peak_only', 'off_peak')
            
        Returns:
            SupplyScenario object
        """
        if key not in self._scenarios:
            raise ValueError(f"Unknown scenario: {key}. "
                             f"Available: {list(self._scenarios.keys())}")
        return self._scenarios[key]
    
    def get_all_scenarios(self) -> Dict[str, SupplyScenario]:
        """Get all available scenarios."""
        return self._scenarios.copy()
    
    def apply_scenario(self,
                       df: pd.DataFrame,
                       scenario_key: str,
                       datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        Apply a supply scenario filter to data.
        
        Args:
            df: DataFrame to filter
            scenario_key: Key of scenario to apply
            datetime_col: Name of datetime column
            
        Returns:
            Filtered DataFrame containing only periods matching the scenario
        """
        scenario = self.get_scenario(scenario_key)
        mask = scenario.filter_func(
            df, datetime_col, self.peak_hour_start, self.peak_hour_end
        )
        return df[mask].copy()
    
    def get_scenario_mask(self,
                          df: pd.DataFrame,
                          scenario_key: str,
                          datetime_col: str = 'datetime') -> np.ndarray:
        """
        Get boolean mask for a scenario without filtering.
        
        Args:
            df: DataFrame with datetime column
            scenario_key: Key of scenario
            datetime_col: Name of datetime column
            
        Returns:
            Boolean numpy array
        """
        scenario = self.get_scenario(scenario_key)
        return scenario.filter_func(
            df, datetime_col, self.peak_hour_start, self.peak_hour_end
        )
    
    def count_periods_by_scenario(self,
                                   df: pd.DataFrame,
                                   datetime_col: str = 'datetime') -> Dict[str, int]:
        """
        Count number of periods for each scenario.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            
        Returns:
            Dictionary mapping scenario key to period count
        """
        counts = {}
        for key, scenario in self._scenarios.items():
            mask = scenario.filter_func(
                df, datetime_col, self.peak_hour_start, self.peak_hour_end
            )
            counts[key] = mask.sum()
        return counts
    
    def get_scenario_summary(self,
                              df: pd.DataFrame,
                              datetime_col: str = 'datetime',
                              price_col: str = 'price') -> pd.DataFrame:
        """
        Get summary statistics for each scenario.
        
        Args:
            df: DataFrame with datetime and price columns
            datetime_col: Name of datetime column
            price_col: Name of price column
            
        Returns:
            DataFrame with scenario statistics
        """
        summaries = []
        
        for key, scenario in self._scenarios.items():
            mask = scenario.filter_func(
                df, datetime_col, self.peak_hour_start, self.peak_hour_end
            )
            scenario_data = df[mask]
            
            if len(scenario_data) > 0 and price_col in scenario_data.columns:
                summary = {
                    'scenario': key,
                    'name': scenario.name,
                    'num_periods': len(scenario_data),
                    'pct_of_total': len(scenario_data) / len(df) * 100,
                    'avg_price': scenario_data[price_col].mean(),
                    'min_price': scenario_data[price_col].min(),
                    'max_price': scenario_data[price_col].max(),
                    'std_price': scenario_data[price_col].std(),
                }
            else:
                summary = {
                    'scenario': key,
                    'name': scenario.name,
                    'num_periods': 0,
                    'pct_of_total': 0,
                    'avg_price': np.nan,
                    'min_price': np.nan,
                    'max_price': np.nan,
                    'std_price': np.nan,
                }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def __repr__(self) -> str:
        return (f"SupplyProfileManager(peak_hours={self.peak_hour_start}:00-{self.peak_hour_end}:00, "
                f"scenarios={list(self._scenarios.keys())})")
