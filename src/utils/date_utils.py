"""
Japanese date utilities for PPA pricing.

Provides functionality for:
- Japanese public holiday detection
- Business day calculations
- Seasonal patterns specific to Japanese electricity market
"""
from datetime import datetime, date, timedelta
from typing import List, Optional, Set
import numpy as np
import pandas as pd
try:
    import jpholiday
except ImportError:  # pragma: no cover - optional dependency fallback
    jpholiday = None


class JapanDateUtils:
    """
    Utility class for Japanese date-related operations.
    
    Handles Japanese public holidays, business day calculations,
    and seasonal patterns relevant to the electricity market.
    """
    
    # Major Japanese holiday periods (approximate peak dates)
    GOLDEN_WEEK_DAYS = list(range(29, 32)) + [3, 4, 5]  # Apr 29 - May 5
    GOLDEN_WEEK_MONTHS = [4, 5]
    OBON_DAYS = list(range(13, 17))  # Aug 13-16
    OBON_MONTH = 8
    NEW_YEAR_DAYS = list(range(1, 4))  # Jan 1-3
    NEW_YEAR_MONTH = 1
    
    # Seasons for Japanese electricity demand
    SEASONS = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11]
    }
    
    # Peak demand periods (high AC/heating load)
    PEAK_SUMMER_MONTHS = [7, 8]
    PEAK_WINTER_MONTHS = [1, 2, 12]
    
    def __init__(self):
        """Initialize the date utilities."""
        self._holiday_cache: Set[date] = set()
    
    def is_holiday(self, dt: date) -> bool:
        """
        Check if a date is a Japanese public holiday.
        
        Args:
            dt: Date to check
            
        Returns:
            True if the date is a holiday
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        if jpholiday is not None:
            return jpholiday.is_holiday(dt)
        return self._is_basic_holiday(dt)
    
    def _is_basic_holiday(self, dt: date) -> bool:
        """
        Basic holiday detection without jpholiday library.
        
        Covers major fixed holidays only.
        """
        month, day = dt.month, dt.day
        
        # Fixed holidays
        fixed_holidays = [
            (1, 1),   # New Year's Day
            (2, 11),  # Foundation Day
            (2, 23),  # Emperor's Birthday (since 2020)
            (4, 29),  # Showa Day
            (5, 3),   # Constitution Day
            (5, 4),   # Greenery Day
            (5, 5),   # Children's Day
            (8, 11),  # Mountain Day
            (11, 3),  # Culture Day
            (11, 23), # Labor Thanksgiving Day
        ]
        
        return (month, day) in fixed_holidays
    
    def is_weekend(self, dt: date) -> bool:
        """
        Check if a date is a weekend (Saturday or Sunday).
        
        Args:
            dt: Date to check
            
        Returns:
            True if weekend
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt.weekday() >= 5
    
    def is_business_day(self, dt: date) -> bool:
        """
        Check if a date is a business day.
        
        Args:
            dt: Date to check
            
        Returns:
            True if it's a business day (not weekend and not holiday)
        """
        return not self.is_weekend(dt) and not self.is_holiday(dt)
    
    def is_golden_week(self, dt: date) -> bool:
        """
        Check if a date falls within Golden Week period.
        
        Args:
            dt: Date to check
            
        Returns:
            True if during Golden Week
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return (dt.month == 4 and dt.day >= 29) or (dt.month == 5 and dt.day <= 5)
    
    def is_obon(self, dt: date) -> bool:
        """
        Check if a date falls within Obon period.
        
        Args:
            dt: Date to check
            
        Returns:
            True if during Obon
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt.month == self.OBON_MONTH and dt.day in self.OBON_DAYS
    
    def is_new_year_period(self, dt: date) -> bool:
        """
        Check if a date falls within New Year period.
        
        Args:
            dt: Date to check
            
        Returns:
            True if during New Year
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt.month == self.NEW_YEAR_MONTH and dt.day in self.NEW_YEAR_DAYS
    
    def is_special_period(self, dt: date) -> bool:
        """
        Check if a date is in a special holiday period.
        
        Special periods include Golden Week, Obon, and New Year
        when electricity demand patterns differ significantly.
        
        Args:
            dt: Date to check
            
        Returns:
            True if in special period
        """
        return self.is_golden_week(dt) or self.is_obon(dt) or self.is_new_year_period(dt)
    
    def get_season(self, dt: date) -> str:
        """
        Get the season for a given date.
        
        Args:
            dt: Date to check
            
        Returns:
            Season name ('winter', 'spring', 'summer', 'autumn')
        """
        if isinstance(dt, datetime):
            dt = dt.date()
            
        for season, months in self.SEASONS.items():
            if dt.month in months:
                return season
        return 'unknown'
    
    def is_peak_demand_period(self, dt: date) -> bool:
        """
        Check if a date is in a peak demand period.
        
        Peak periods are summer (high AC load) and winter (heating).
        
        Args:
            dt: Date to check
            
        Returns:
            True if in peak demand period
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt.month in self.PEAK_SUMMER_MONTHS or dt.month in self.PEAK_WINTER_MONTHS
    
    def get_holidays_in_range(self, start_date: date, end_date: date) -> List[date]:
        """
        Get all holidays in a date range.
        
        Args:
            start_date: Start of range
            end_date: End of range
            
        Returns:
            List of holiday dates
        """
        holidays = []
        current = start_date
        while current <= end_date:
            if self.is_holiday(current):
                holidays.append(current)
            current += timedelta(days=1)
        return holidays
    
    def count_business_days(self, start_date: date, end_date: date) -> int:
        """
        Count business days between two dates.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Number of business days
        """
        count = 0
        current = start_date
        while current <= end_date:
            if self.is_business_day(current):
                count += 1
            current += timedelta(days=1)
        return count
    
    def add_business_days(self, dt: date, days: int) -> date:
        """
        Add business days to a date.
        
        Args:
            dt: Starting date
            days: Number of business days to add
            
        Returns:
            Resulting date
        """
        if isinstance(dt, datetime):
            dt = dt.date()
            
        direction = 1 if days >= 0 else -1
        remaining = abs(days)
        
        while remaining > 0:
            dt += timedelta(days=direction)
            if self.is_business_day(dt):
                remaining -= 1
                
        return dt
    
    def get_date_features(self, dt: date) -> dict:
        """
        Extract date-based features for machine learning.
        
        Args:
            dt: Date to analyze
            
        Returns:
            Dictionary of features
        """
        if isinstance(dt, datetime):
            dt = dt.date()
            
        return {
            'day_of_week': dt.weekday(),
            'day_of_month': dt.day,
            'month': dt.month,
            'quarter': (dt.month - 1) // 3 + 1,
            'day_of_year': dt.timetuple().tm_yday,
            'week_of_year': dt.isocalendar()[1],
            'is_weekend': self.is_weekend(dt),
            'is_holiday': self.is_holiday(dt),
            'is_business_day': self.is_business_day(dt),
            'is_golden_week': self.is_golden_week(dt),
            'is_obon': self.is_obon(dt),
            'is_new_year': self.is_new_year_period(dt),
            'is_special_period': self.is_special_period(dt),
            'season': self.get_season(dt),
            'is_peak_demand': self.is_peak_demand_period(dt),
        }
    
    @staticmethod
    def get_settlement_period(dt: datetime) -> int:
        """
        Get the JEPX settlement period (1-48) for a datetime.
        
        JEPX uses 30-minute settlement periods.
        Period 1: 00:00-00:30
        Period 48: 23:30-24:00
        
        Args:
            dt: Datetime
            
        Returns:
            Settlement period (1-48)
        """
        return dt.hour * 2 + (1 if dt.minute < 30 else 2)
    
    @staticmethod
    def get_period_start_time(period: int) -> tuple:
        """
        Get the start time (hour, minute) for a settlement period.
        
        Args:
            period: Settlement period (1-48)
            
        Returns:
            Tuple of (hour, minute)
        """
        if period < 1 or period > 48:
            raise ValueError(f"Period must be 1-48, got {period}")
        
        period_index = period - 1
        hour = period_index // 2
        minute = (period_index % 2) * 30
        return hour, minute
