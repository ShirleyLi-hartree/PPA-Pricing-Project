"""
Prophet forecaster for electricity price prediction.

Implements Facebook Prophet model with Japanese market-specific
holiday effects and seasonality configurations for medium-term
electricity price forecasting.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import itertools
import logging
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.base.model_interface import BasePriceModel, ModelConfig, ForecastResult
from src.utils.date_utils import JapanDateUtils

logger = logging.getLogger(__name__)


class ProphetForecaster(BasePriceModel):
    """
    Prophet-based electricity price forecaster.
    
    Prophet is particularly well-suited for:
    - Strong seasonal patterns (daily, weekly, yearly)
    - Holiday effects (Japanese specific holidays)
    - Missing data and outliers
    - Medium-term forecasting (1-3 months)
    
    Features:
    - Automatic trend and seasonality detection
    - Japanese holiday integration (Golden Week, Obon, New Year)
    - Uncertainty intervals
    - Multiplicative seasonality for price data
    """
    
    def __init__(self,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True,
                 seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 config: Optional[ModelConfig] = None):
        """
        Initialize Prophet forecaster.
        
        Args:
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality (for 30-min data)
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Flexibility of seasonality
            holidays_prior_scale: Flexibility of holiday effects
            config: Model configuration
        """
        default_config = ModelConfig(
            name="ProphetForecaster",
            version="1.0.0",
            description="Prophet for electricity price forecasting with Japanese holidays"
        )
        super().__init__(config or default_config)
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        
        self._model = None
        self._date_utils = JapanDateUtils()
        self._training_end_date = None
    
    def _create_japan_holidays_df(self, 
                                   start_date: datetime,
                                   end_date: datetime) -> pd.DataFrame:
        """
        Create DataFrame of Japanese holidays for Prophet.
        
        Args:
            start_date: Start date for holiday range
            end_date: End date for holiday range
            
        Returns:
            DataFrame with holiday information for Prophet
        """
        holidays_list = []
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for date in date_range:
            date_obj = date.date()
            
            # Check for public holidays
            if self._date_utils.is_holiday(date_obj):
                holidays_list.append({
                    'holiday': 'japan_public_holiday',
                    'ds': date,
                    'lower_window': 0,
                    'upper_window': 0,
                })
            
            # Golden Week (April 29 - May 5)
            if self._date_utils.is_golden_week(date_obj):
                holidays_list.append({
                    'holiday': 'golden_week',
                    'ds': date,
                    'lower_window': 0,
                    'upper_window': 0,
                })
            
            # Obon (August 13-16)
            if self._date_utils.is_obon(date_obj):
                holidays_list.append({
                    'holiday': 'obon',
                    'ds': date,
                    'lower_window': 0,
                    'upper_window': 0,
                })
            
            # New Year (January 1-3)
            if self._date_utils.is_new_year_period(date_obj):
                holidays_list.append({
                    'holiday': 'new_year',
                    'ds': date,
                    'lower_window': 0,
                    'upper_window': 0,
                })
        
        if not holidays_list:
            return None
        
        holidays_df = pd.DataFrame(holidays_list)
        
        # Remove duplicates (a date might be both public holiday and golden week)
        holidays_df = holidays_df.drop_duplicates(subset=['ds', 'holiday'])
        
        return holidays_df
    
    def fit(self,
            train_data: pd.DataFrame,
            datetime_col: str = 'datetime',
            price_col: str = 'price',
            **kwargs) -> 'ProphetForecaster':
        """
        Fit Prophet model.
        
        Args:
            train_data: DataFrame with datetime and price
            datetime_col: Name of datetime column
            price_col: Name of price column
            
        Returns:
            Self for method chaining
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet required. Install with: pip install prophet")
        
        # Prepare data in Prophet format (ds, y)
        df = train_data.copy()
        df['ds'] = pd.to_datetime(df[datetime_col])
        df['y'] = df[price_col]
        
        # Store training end date
        self._training_end_date = df['ds'].max()
        
        # Get date range for holidays
        start_date = df['ds'].min()
        # Extend end date for future predictions
        end_date = self._training_end_date + pd.Timedelta(days=365)
        
        # Create Japanese holidays DataFrame
        holidays_df = self._create_japan_holidays_df(start_date, end_date)
        
        # Initialize Prophet model
        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,  # We'll add custom daily seasonality for 30-min data
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            holidays=holidays_df,
            interval_width=0.90,
        )
        
        # Add custom daily seasonality for 30-minute data (48 periods per day)
        if self.daily_seasonality:
            self._model.add_seasonality(
                name='daily',
                period=1,  # 1 day
                fourier_order=10,  # Higher order for 30-min resolution
            )
        
        # Add Japanese-specific seasonal patterns
        # Summer peak (July-August AC demand)
        self._model.add_seasonality(
            name='summer_peak',
            period=365.25,
            fourier_order=3,
            condition_name='is_summer'
        )
        
        # Winter peak (December-February heating demand)
        self._model.add_seasonality(
            name='winter_peak',
            period=365.25,
            fourier_order=3,
            condition_name='is_winter'
        )
        
        # Add condition columns
        df['is_summer'] = df['ds'].dt.month.isin([7, 8])
        df['is_winter'] = df['ds'].dt.month.isin([12, 1, 2])
        
        # Fit model
        self._model.fit(df[['ds', 'y', 'is_summer', 'is_winter']])
        
        self._training_data = df
        self._is_fitted = True
        
        # Store model parameters
        self._model_params = {
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'seasonality_mode': self.seasonality_mode,
            'training_periods': len(df),
        }
        
        return self
    
    def predict(self,
                forecast_horizon: int,
                start_date: Optional[datetime] = None,
                historical_data: Optional[pd.DataFrame] = None,
                **kwargs) -> ForecastResult:
        """
        Generate price forecasts.
        
        Args:
            forecast_horizon: Number of periods to forecast
            start_date: Start date for forecast
            historical_data: Not used (Prophet handles internally)
            
        Returns:
            ForecastResult with forecasts
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if start_date is None:
            start_date = self._training_end_date + pd.Timedelta(minutes=30)
        
        # Generate forecast datetimes
        datetime_index = pd.date_range(
            start=start_date,
            periods=forecast_horizon,
            freq='30min'
        )
        
        # Create future DataFrame for Prophet
        future = pd.DataFrame({'ds': datetime_index})
        
        # Add seasonal condition columns
        future['is_summer'] = future['ds'].dt.month.isin([7, 8])
        future['is_winter'] = future['ds'].dt.month.isin([12, 1, 2])
        
        # Generate forecast
        forecast = self._model.predict(future)
        
        # Extract results
        point_forecast = forecast['yhat'].values
        lower_bound = forecast['yhat_lower'].values
        upper_bound = forecast['yhat_upper'].values
        
        # Ensure non-negative prices
        point_forecast = np.maximum(0, point_forecast)
        lower_bound = np.maximum(0, lower_bound)
        upper_bound = np.maximum(0, upper_bound)
        
        return ForecastResult(
            datetime=datetime_index,
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=0.90,
            metadata={
                'model': 'Prophet',
                'trend': forecast['trend'].values[-1] if len(forecast) > 0 else None,
            }
        )
    
    def get_seasonality_components(self) -> Optional[pd.DataFrame]:
        """
        Get decomposed seasonality components from the fitted model.
        
        Returns:
            DataFrame with seasonality components or None if not fitted
        """
        if not self._is_fitted or self._model is None:
            return None
        
        # Create a year's worth of data points for decomposition
        future = self._model.make_future_dataframe(
            periods=48 * 365,  # 1 year of 30-min data
            freq='30min'
        )
        
        future['is_summer'] = future['ds'].dt.month.isin([7, 8])
        future['is_winter'] = future['ds'].dt.month.isin([12, 1, 2])
        
        forecast = self._model.predict(future)
        
        components = ['trend', 'yhat']
        
        # Add available seasonality components
        for col in forecast.columns:
            if col.endswith('_yearly') or col.endswith('_weekly') or col == 'daily':
                components.append(col)
        
        return forecast[['ds'] + [c for c in components if c in forecast.columns]]
    
    def get_trend(self) -> Optional[pd.DataFrame]:
        """
        Get trend component from fitted model.
        
        Returns:
            DataFrame with trend values
        """
        if not self._is_fitted:
            return None
        
        future = pd.DataFrame({'ds': self._training_data['ds']})
        future['is_summer'] = future['ds'].dt.month.isin([7, 8])
        future['is_winter'] = future['ds'].dt.month.isin([12, 1, 2])
        
        forecast = self._model.predict(future)
        
        return forecast[['ds', 'trend']]
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'seasonality_mode': self.seasonality_mode,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale,
            **self._model_params
        }
    
    def set_params(self, **params) -> 'ProphetForecaster':
        """Set model parameters."""
        valid_params = [
            'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
            'seasonality_mode', 'changepoint_prior_scale', 
            'seasonality_prior_scale', 'holidays_prior_scale'
        ]
        
        for param, value in params.items():
            if param in valid_params:
                setattr(self, param, value)
        
        return self
    
    def plot_components(self, save_path: Optional[str] = None):
        """
        Plot forecast components (trend, seasonality, holidays).
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        try:
            from prophet.plot import plot_components
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        future = self._model.make_future_dataframe(
            periods=48 * 30,  # 30 days
            freq='30min'
        )
        future['is_summer'] = future['ds'].dt.month.isin([7, 8])
        future['is_winter'] = future['ds'].dt.month.isin([12, 1, 2])
        
        forecast = self._model.predict(future)
        
        fig = self._model.plot_components(forecast)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def structure_search(
        self,
        train_data: pd.DataFrame,
        datetime_col: str = 'datetime',
        price_col: str = 'price',
        representative_years: int = 4,
        val_months: int = 6,
        changepoint_prior_candidates: Optional[List[float]] = None,
        seasonality_prior_candidates: Optional[List[float]] = None,
        holidays_prior_candidates: Optional[List[float]] = None,
        mae_weight: float = 0.5,
        worst_week_weight: float = 0.3,
        bias_weight: float = 0.2,
    ) -> 'ProphetForecaster':
        """
        One-time structural search: find optimal Prophet hyperparameters.
        
        Call ONCE before rolling-window training; subsequent ``fit()`` calls
        use the optimal parameters found here. This method searches over
        prior scale parameters (changepoint, seasonality, holidays) using
        grid search with Loss_GS objective function.
        
        Loss_GS = mae_weight*MAE + worst_week_weight*WorstWeekMAE + bias_weight*|Bias|
        
        Args:
            train_data: DataFrame with datetime and price columns
            datetime_col: Name of datetime column
            price_col: Name of price column
            representative_years: Years of data for training in search
            val_months: Months of data for validation in search
            changepoint_prior_candidates: List of changepoint_prior_scale values to test
            seasonality_prior_candidates: List of seasonality_prior_scale values to test
            holidays_prior_candidates: List of holidays_prior_scale values to test
            mae_weight: Weight for MAE in Loss_GS (default 0.5)
            worst_week_weight: Weight for worst week MAE in Loss_GS (default 0.3)
            bias_weight: Weight for absolute bias in Loss_GS (default 0.2)
            
        Returns:
            Self with optimal parameters set
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet required. Install with: pip install prophet")
        
        from src.models.base.metrics import PriceMetrics
        
        logger.info("=" * 60)
        logger.info("PROPHET STRUCTURE SEARCH (one-time)")
        logger.info("=" * 60)
        
        # Default parameter candidates if not provided
        if changepoint_prior_candidates is None:
            changepoint_prior_candidates = [0.001, 0.01, 0.05, 0.1, 0.5]
        if seasonality_prior_candidates is None:
            seasonality_prior_candidates = [0.01, 0.1, 1.0, 10.0]
        if holidays_prior_candidates is None:
            holidays_prior_candidates = [0.01, 0.1, 1.0, 10.0]
        
        # --- Step 1: Prepare data windows ---
        df = train_data[[datetime_col, price_col]].copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col)
        
        # Split into train and validation periods
        data_end = df[datetime_col].max()
        val_start = data_end - pd.DateOffset(months=val_months)
        train_start = val_start - pd.DateOffset(years=representative_years)
        
        # Filter to search window
        df = df[df[datetime_col] >= train_start]
        
        train_df = df[df[datetime_col] < val_start].copy()
        val_df = df[df[datetime_col] >= val_start].copy()
        
        logger.info(f"Training window: {train_df[datetime_col].min().date()} to "
                    f"{train_df[datetime_col].max().date()} ({len(train_df)} samples)")
        logger.info(f"Validation window: {val_df[datetime_col].min().date()} to "
                    f"{val_df[datetime_col].max().date()} ({len(val_df)} samples)")
        
        if len(train_df) < 48 * 30 or len(val_df) < 48 * 7:  # Min 30 days train, 7 days val
            logger.warning("Insufficient data for structure search. Using default parameters.")
            self._structure_fixed = True
            return self
        
        # --- Step 2: Prepare Prophet format data ---
        train_prophet = pd.DataFrame({
            'ds': train_df[datetime_col],
            'y': train_df[price_col]
        })
        train_prophet['is_summer'] = train_prophet['ds'].dt.month.isin([7, 8])
        train_prophet['is_winter'] = train_prophet['ds'].dt.month.isin([12, 1, 2])
        
        val_prophet = pd.DataFrame({
            'ds': val_df[datetime_col],
            'y': val_df[price_col]
        })
        val_prophet['is_summer'] = val_prophet['ds'].dt.month.isin([7, 8])
        val_prophet['is_winter'] = val_prophet['ds'].dt.month.isin([12, 1, 2])
        
        # Create holidays DataFrame for search period
        holidays_df = self._create_japan_holidays_df(
            train_prophet['ds'].min(),
            val_prophet['ds'].max() + pd.Timedelta(days=30)
        )
        
        # --- Step 3: Grid search over parameter combinations ---
        param_combinations = list(itertools.product(
            changepoint_prior_candidates,
            seasonality_prior_candidates,
            holidays_prior_candidates
        ))
        
        n_combinations = len(param_combinations)
        logger.info(f"Testing {n_combinations} parameter combinations...")
        
        best_loss = float('inf')
        best_params = {
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale,
        }
        trial_results: List[Dict[str, Any]] = []
        
        for i, (cp_prior, seas_prior, hol_prior) in enumerate(param_combinations):
            logger.info(f"Trial {i+1}/{n_combinations}: cp={cp_prior}, "
                        f"seas={seas_prior}, hol={hol_prior}")
            
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Initialize and fit Prophet model
                    model = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality,
                        daily_seasonality=False,
                        seasonality_mode=self.seasonality_mode,
                        changepoint_prior_scale=cp_prior,
                        seasonality_prior_scale=seas_prior,
                        holidays_prior_scale=hol_prior,
                        holidays=holidays_df,
                        interval_width=0.90,
                    )
                    
                    # Add custom seasonalities
                    if self.daily_seasonality:
                        model.add_seasonality(name='daily', period=1, fourier_order=10)
                    model.add_seasonality(
                        name='summer_peak', period=365.25, fourier_order=3,
                        condition_name='is_summer'
                    )
                    model.add_seasonality(
                        name='winter_peak', period=365.25, fourier_order=3,
                        condition_name='is_winter'
                    )
                    
                    # Fit model
                    model.fit(train_prophet[['ds', 'y', 'is_summer', 'is_winter']])
                    
                    # Predict on validation set
                    future = val_prophet[['ds', 'is_summer', 'is_winter']].copy()
                    forecast = model.predict(future)
                    
                    # Calculate metrics
                    actual = val_prophet['y'].values
                    predicted = forecast['yhat'].values
                    errors = actual - predicted
                    
                    mae = PriceMetrics.mae(actual, predicted)
                    bias = PriceMetrics.mean_bias_error(actual, predicted)
                    worst_week_mae = PriceMetrics.worst_week_mae(
                        actual, predicted, pd.DatetimeIndex(val_prophet['ds'])
                    )
                    
                    # Calculate Loss_GS
                    loss_gs = (
                        mae_weight * mae +
                        worst_week_weight * worst_week_mae +
                        bias_weight * abs(bias)
                    )
                    
                    trial_result = {
                        'changepoint_prior_scale': cp_prior,
                        'seasonality_prior_scale': seas_prior,
                        'holidays_prior_scale': hol_prior,
                        'mae': mae,
                        'bias': bias,
                        'worst_week_mae': worst_week_mae,
                        'loss_gs': loss_gs,
                        'success': True,
                    }
                    trial_results.append(trial_result)
                    
                    logger.info(f"  Loss_GS={loss_gs:.4f} (MAE={mae:.4f}, "
                                f"WorstWeek={worst_week_mae:.4f}, Bias={bias:.4f})")
                    
                    if loss_gs < best_loss:
                        best_loss = loss_gs
                        best_params = {
                            'changepoint_prior_scale': cp_prior,
                            'seasonality_prior_scale': seas_prior,
                            'holidays_prior_scale': hol_prior,
                        }
                        logger.info(f"  -> New best configuration!")
                        
            except Exception as e:
                logger.warning(f"  Trial failed: {e}")
                trial_results.append({
                    'changepoint_prior_scale': cp_prior,
                    'seasonality_prior_scale': seas_prior,
                    'holidays_prior_scale': hol_prior,
                    'loss_gs': float('inf'),
                    'success': False,
                    'error': str(e),
                })
                continue
        
        # --- Step 4: Apply optimal parameters ---
        logger.info("=" * 60)
        logger.info("STRUCTURE SEARCH COMPLETE")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best Loss_GS: {best_loss:.4f}")
        logger.info("=" * 60)
        
        self.changepoint_prior_scale = best_params['changepoint_prior_scale']
        self.seasonality_prior_scale = best_params['seasonality_prior_scale']
        self.holidays_prior_scale = best_params['holidays_prior_scale']
        
        # Store search results
        self._structure_search_results = {
            'best_params': best_params,
            'best_loss': best_loss,
            'all_trials': trial_results,
            'n_successful_trials': sum(1 for t in trial_results if t.get('success', False)),
        }
        self._structure_fixed = True
        
        return self
    
    def get_structure_search_results(self) -> Optional[Dict[str, Any]]:
        """
        Get results from structure search if performed.
        
        Returns:
            Dictionary with search results or None if not performed
        """
        return getattr(self, '_structure_search_results', None)
