"""
Feature engineering for electricity price forecasting.

Creates features for machine learning models including:
- Temporal features
- Lagged price features
- Rolling statistics
- External data features
"""
from datetime import datetime
from typing import Optional, Dict, List, Union, Tuple
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils.date_utils import JapanDateUtils


class FeatureEngineer:
    """
    Feature engineering for electricity price forecasting ML models.
    
    Creates a comprehensive feature set including:
    - Temporal features (hour, day, month, cyclical encodings)
    - Lagged price features (autocorrelation capture)
    - Rolling statistics (trend capture)
    - External features (fuel prices, weather if available)
    """
    
    def __init__(self, 
                 lags: List[int] = [48, 96, 336, 672, 1440, 2016], # Removed 1, 2
                 rolling_windows: List[int] = [24, 48, 168, 720]):
        """
        Initialize feature engineer.
        
        Args:
            lags: List of lag periods for lagged features
            rolling_windows: List of rolling window sizes for statistics
        """
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.date_utils = JapanDateUtils()
        self._feature_names: List[str] = []
        self._hour_means: Optional[pd.Series] = None
        self._hour_dow_means: Optional[pd.Series] = None
    
    def fit_statistics(self, df: pd.DataFrame, target_col: str = 'price'):
        """
        Calculate global statistics from training data (to avoid leakage).
        
        Args:
            df: Training DataFrame
            target_col: Target column
        """
        temp_df = df.copy()
        temp_df['hour'] = pd.to_datetime(temp_df['datetime']).dt.hour
        temp_df['day_of_week'] = pd.to_datetime(temp_df['datetime']).dt.dayofweek
        
        self._hour_means = temp_df.groupby('hour')[target_col].mean()
        self._hour_dow_means = temp_df.groupby(['hour', 'day_of_week'])[target_col].mean()
        
    def create_temporal_features(self,
                                  df: pd.DataFrame,
                                  datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        Create temporal features from datetime column.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with temporal features
        """
        result = df.copy()
        dt = pd.to_datetime(result[datetime_col])
        
        # Basic temporal features
        result['hour'] = dt.dt.hour
        result['minute'] = dt.dt.minute
        result['period'] = result['hour'] * 2 + result['minute'] // 30  # 30-min period (0-47)
        result['day_of_week'] = dt.dt.dayofweek
        result['day_of_month'] = dt.dt.day
        result['month'] = dt.dt.month
        result['quarter'] = dt.dt.quarter
        result['day_of_year'] = dt.dt.dayofyear
        result['week_of_year'] = dt.dt.isocalendar().week.astype(int)
        result['year'] = dt.dt.year
        
        # Binary features
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        result['is_morning_peak'] = ((result['hour'] >= 8) & (result['hour'] <= 10)).astype(int)
        result['is_evening_peak'] = ((result['hour'] >= 17) & (result['hour'] <= 20)).astype(int)
        result['is_night'] = ((result['hour'] >= 22) | (result['hour'] <= 5)).astype(int)
        
        # Japanese-specific features
        result['is_holiday'] = dt.dt.date.apply(self.date_utils.is_holiday).astype(int)
        result['is_golden_week'] = dt.dt.date.apply(self.date_utils.is_golden_week).astype(int)
        result['is_obon'] = dt.dt.date.apply(self.date_utils.is_obon).astype(int)
        result['is_new_year'] = dt.dt.date.apply(self.date_utils.is_new_year_period).astype(int)
        result['is_special_period'] = (
            result['is_golden_week'] | result['is_obon'] | result['is_new_year']
        ).astype(int)
        
        # Seasonal features
        result['is_summer'] = result['month'].isin([6, 7, 8]).astype(int)
        result['is_winter'] = result['month'].isin([12, 1, 2]).astype(int)
        result['is_peak_demand_season'] = result['month'].isin([1, 2, 7, 8, 12]).astype(int)
        
        # Japan-specific weather/seasonal features
        # Tsuyu (rainy season): mid-June to mid-July - affects solar generation
        result['is_tsuyu_season'] = (
            ((result['month'] == 6) & (result['day_of_month'] >= 10)) |
            ((result['month'] == 7) & (result['day_of_month'] <= 20))
        ).astype(int)
        
        # Typhoon season: August to October - affects wind and demand patterns
        result['is_typhoon_season'] = result['month'].isin([8, 9, 10]).astype(int)
        
        # Industrial activity patterns
        # Week start (Monday morning) - industrial ramp-up
        result['is_week_start'] = (
            (result['day_of_week'] == 0) & (result['hour'] < 10)
        ).astype(int)
        
        # Week end (Friday afternoon) - industrial wind-down
        result['is_week_end'] = (
            (result['day_of_week'] == 4) & (result['hour'] >= 15)
        ).astype(int)
        
        # Fiscal year end (March) - increased industrial activity
        result['is_fiscal_year_end'] = (result['month'] == 3).astype(int)
        
        # Cyclical encoding (sin/cos for periodic features)
        result['hour_sin'] = np.sin(2 * np.pi * result['period'] / 48)
        result['hour_cos'] = np.cos(2 * np.pi * result['period'] / 48)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * (result['month'] - 1) / 12)
        result['month_cos'] = np.cos(2 * np.pi * (result['month'] - 1) / 12)
        result['doy_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
        result['doy_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
        
        # Interaction features (Hour x DayOfWeek) - very important for electricity
        result['hour_dow'] = result['hour'] * 7 + result['day_of_week']
        result['is_weekend_hour'] = result['is_weekend'] * result['hour']
        
        # One-hot encoding for hour (to force model to see each hour distinctly)
        for h in range(24):
            result[f'is_hour_{h}'] = (result['hour'] == h).astype(int)
        
        # Global hour-of-day mean (to provide strong baseline pattern)
        if self._hour_means is not None:
            result['global_hour_mean'] = result['hour'].map(self._hour_means)
        else:
            result['global_hour_mean'] = 10.0 # fallback
            
        if self._hour_dow_means is not None:
            # Map the multi-index correctly
            hour_dow_map = self._hour_dow_means.to_dict()
            result['global_hour_dow_mean'] = [
                hour_dow_map.get((h, d), 10.0) 
                for h, d in zip(result['hour'], result['day_of_week'])
            ]
        else:
            result['global_hour_dow_mean'] = 10.0 # fallback
        
        return result
    
    def create_lagged_features(self,
                               df: pd.DataFrame,
                               target_col: str = 'price',
                               lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create lagged features from target variable.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            lags: List of lag periods (uses default if None)
            
        Returns:
            DataFrame with lagged features
        """
        result = df.copy()
        lags = lags or self.lags
        
        for lag in lags:
            result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
        
        # Differences (price changes) - use shift(1) to avoid leakage
        target_shifted = result[target_col].shift(1)
        # result[f'{target_col}_diff_1'] = target_shifted.diff(1) # Removed short-term diff
        result[f'{target_col}_diff_48'] = target_shifted.diff(48)  # Daily diff
        
        # Same period yesterday/last week
        result[f'{target_col}_same_hour_yesterday'] = result[target_col].shift(48)
        result[f'{target_col}_same_hour_last_week'] = result[target_col].shift(336)
        
        return result
    
    def create_rolling_features(self,
                                df: pd.DataFrame,
                                target_col: str = 'price',
                                windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features
        """
        result = df.copy()
        windows = windows or self.rolling_windows
        
        # Shift target by 1 to avoid data leakage
        target_shifted = result[target_col].shift(1)
        
        for window in windows:
            # Rolling mean
            result[f'{target_col}_rolling_mean_{window}'] = (
                target_shifted.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            result[f'{target_col}_rolling_std_{window}'] = (
                target_shifted.rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min/max
            result[f'{target_col}_rolling_min_{window}'] = (
                target_shifted.rolling(window=window, min_periods=1).min()
            )
            result[f'{target_col}_rolling_max_{window}'] = (
                target_shifted.rolling(window=window, min_periods=1).max()
            )
            
            # Rolling quantiles
            result[f'{target_col}_rolling_q25_{window}'] = (
                target_shifted.rolling(window=window, min_periods=1).quantile(0.25)
            )
            result[f'{target_col}_rolling_q75_{window}'] = (
                target_shifted.rolling(window=window, min_periods=1).quantile(0.75)
            )
        
        # Exponential moving averages
        for span in [24, 48]: # Removed 12
            result[f'{target_col}_ema_{span}'] = (
                target_shifted.ewm(span=span, adjust=False).mean()
            )
        
        return result
    
    def create_volatility_features(self,
                                   df: pd.DataFrame,
                                   target_col: str = 'price') -> pd.DataFrame:
        """
        Create volatility and price change features for medium-term forecasting.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with volatility features
        """
        result = df.copy()
        
        # Shift target by 1 to avoid data leakage
        target_shifted = result[target_col].shift(1)
        
        # Week-over-week change rate
        result[f'{target_col}_wow_change'] = (
            target_shifted / target_shifted.shift(336).replace(0, np.nan) - 1
        )
        
        # Month-over-month change rate (30 days)
        result[f'{target_col}_mom_change'] = (
            target_shifted / target_shifted.shift(1440).replace(0, np.nan) - 1
        )
        
        # Rolling coefficient of variation (volatility relative to mean)
        # 30-day window (1440 periods)
        rolling_mean_30d = target_shifted.rolling(window=1440, min_periods=48).mean()
        rolling_std_30d = target_shifted.rolling(window=1440, min_periods=48).std()
        result[f'{target_col}_cv_30d'] = rolling_std_30d / rolling_mean_30d.replace(0, np.nan)
        
        # 7-day window (336 periods)
        rolling_mean_7d = target_shifted.rolling(window=336, min_periods=48).mean()
        rolling_std_7d = target_shifted.rolling(window=336, min_periods=48).std()
        result[f'{target_col}_cv_7d'] = rolling_std_7d / rolling_mean_7d.replace(0, np.nan)
        
        # Price range (max - min) over rolling windows
        result[f'{target_col}_range_7d'] = (
            target_shifted.rolling(window=336, min_periods=48).max() -
            target_shifted.rolling(window=336, min_periods=48).min()
        )
        
        result[f'{target_col}_range_30d'] = (
            target_shifted.rolling(window=1440, min_periods=48).max() -
            target_shifted.rolling(window=1440, min_periods=48).min()
        )
        
        # Year-over-year change rate (if enough data available)
        yoy_lag = 48 * 365  # 1 year in 30-min periods
        if len(result) > yoy_lag:
            result[f'{target_col}_yoy_change'] = (
                target_shifted / target_shifted.shift(yoy_lag).replace(0, np.nan) - 1
            )
        
        return result
    
    def create_external_features(self,
                                  df: pd.DataFrame,
                                  fuel_prices: Optional[pd.DataFrame] = None,
                                  weather: Optional[pd.DataFrame] = None,
                                  datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        Add external features (fuel prices, weather).
        
        Args:
            df: Base DataFrame
            fuel_prices: DataFrame with fuel price data
            weather: DataFrame with weather data
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with external features
        """
        result = df.copy()
        result[datetime_col] = pd.to_datetime(result[datetime_col])
        
        # Add fuel prices if provided
        if fuel_prices is not None:
            fuel_df = fuel_prices.copy()
            fuel_df[datetime_col] = pd.to_datetime(fuel_df[datetime_col])
            
            # Merge and forward-fill (fuel prices often monthly)
            result = result.set_index(datetime_col)
            fuel_df = fuel_df.set_index(datetime_col)
            
            for col in ['lng_price', 'coal_price', 'oil_price']:
                if col in fuel_df.columns:
                    result = result.join(fuel_df[[col]], how='left')
                    result[col] = result[col].ffill()
            
            result = result.reset_index()
            
            # Create fuel price features
            if 'lng_price' in result.columns and 'coal_price' in result.columns:
                result['lng_coal_ratio'] = result['lng_price'] / result['coal_price'].replace(0, 1)
                result['lng_coal_spread'] = result['lng_price'] - result['coal_price']
        
        # Add weather if provided
        if weather is not None:
            weather_df = weather.copy()
            weather_df[datetime_col] = pd.to_datetime(weather_df[datetime_col])
            
            result = result.set_index(datetime_col)
            weather_df = weather_df.set_index(datetime_col)
            
            for col in ['temperature', 'solar_irradiance', 'wind_speed']:
                if col in weather_df.columns:
                    result = result.join(weather_df[[col]], how='left')
                    result[col] = result[col].interpolate()
            
            result = result.reset_index()
            
            # Create weather-derived features
            if 'temperature' in result.columns:
                result['heating_degree_days'] = np.maximum(18 - result['temperature'], 0)
                result['cooling_degree_days'] = np.maximum(result['temperature'] - 26, 0)
        
        return result
    
    def create_all_features(self,
                            df: pd.DataFrame,
                            target_col: str = 'price',
                            datetime_col: str = 'datetime',
                            fuel_prices: Optional[pd.DataFrame] = None,
                            weather: Optional[pd.DataFrame] = None,
                            drop_na: bool = True,
                            extra_lag_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create all features.
        
        Args:
            df: Base DataFrame
            target_col: Name of target column
            datetime_col: Name of datetime column
            fuel_prices: Optional fuel price data
            weather: Optional weather data
            drop_na: Whether to drop rows with NaN
            extra_lag_cols: List of additional columns to create lag/rolling features for
            
        Returns:
            DataFrame with all features
        """
        result = df.copy()
        
        # Temporal features
        result = self.create_temporal_features(result, datetime_col)
        
        # Lagged features
        result = self.create_lagged_features(result, target_col)
        
        # Rolling features
        result = self.create_rolling_features(result, target_col)
        
        # Volatility features (for medium-term forecasting)
        result = self.create_volatility_features(result, target_col)
        
        # Extra lagged/rolling features for other columns (e.g., volume)
        if extra_lag_cols:
            for col in extra_lag_cols:
                if col in result.columns:
                    # Use standard lags/windows for these too
                    result = self.create_lagged_features(result, col)
                    result = self.create_rolling_features(result, col)
        
        # External features
        result = self.create_external_features(result, fuel_prices, weather, datetime_col)
        
        # Drop NaN if requested
        if drop_na:
            result = result.dropna()
        
        # Store feature names
        exclude_cols = [datetime_col, target_col, 'date', 'region']
        self._feature_names = [col for col in result.columns if col not in exclude_cols]
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names
    
    def prepare_for_training(self,
                              df: pd.DataFrame,
                              target_col: str = 'price',
                              datetime_col: str = 'datetime',
                              test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for ML training.
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
            datetime_col: Name of datetime column
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        # Create features
        featured_df = self.create_all_features(df, target_col, datetime_col)
        
        # Get feature columns
        exclude_cols = [datetime_col, target_col, 'date', 'region']
        feature_cols = [col for col in featured_df.columns if col not in exclude_cols]
        
        X = featured_df[feature_cols].values
        y = featured_df[target_col].values
        
        # Time series split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def create_forecast_features(self,
                                  historical_df: pd.DataFrame,
                                  forecast_datetimes: pd.DatetimeIndex,
                                  target_col: str = 'price',
                                  datetime_col: str = 'datetime',
                                  extra_lag_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create features for forecasting (future timestamps).
        
        Args:
            historical_df: Historical data with prices
            forecast_datetimes: Future timestamps to forecast
            target_col: Name of target column
            datetime_col: Name of datetime column
            extra_lag_cols: List of additional columns to create lag/rolling features for
            
        Returns:
            DataFrame with features for forecasting
        """
        # Create statistics_v1 DataFrame for forecast periods
        forecast_df = pd.DataFrame({datetime_col: forecast_datetimes})
        
        # Add temporal features (no dependency on target)
        forecast_df = self.create_temporal_features(forecast_df, datetime_col)
        
        # For lagged features, need historical data
        # Combine historical and forecast for lag calculation
        # We need to include extra_lag_cols in the combination
        cols_to_keep = [datetime_col, target_col]
        if extra_lag_cols:
            cols_to_keep.extend([col for col in extra_lag_cols if col in historical_df.columns])
            
        combined = pd.concat([
            historical_df[cols_to_keep],
            forecast_df
        ], ignore_index=True)
        
        combined = combined.sort_values(datetime_col)
        combined = self.create_lagged_features(combined, target_col)
        combined = self.create_rolling_features(combined, target_col)
        combined = self.create_volatility_features(combined, target_col)
        
        if extra_lag_cols:
             for col in extra_lag_cols:
                 if col in combined.columns:
                     combined = self.create_lagged_features(combined, col)
                     combined = self.create_rolling_features(combined, col)
        
        # Extract only forecast rows
        forecast_df = combined[combined[datetime_col].isin(forecast_datetimes)].copy()
        
        return forecast_df
