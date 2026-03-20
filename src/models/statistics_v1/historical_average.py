"""
Historical Average Model for electricity price prediction.

A simple baseline model using hourly and monthly historical averages.
"""
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models.base.model_interface import BasePriceModel, ModelConfig, ForecastResult


class HistoricalAverageModel(BasePriceModel):
    """
    Historical Average Model - Simple baseline using hour/month patterns.
    
    Strategy:
    - Computes average price by hour of day and month
    - Predictions are based on looking up the historical average
    - Provides uncertainty intervals based on historical std
    
    Advantages: Simple, stable, interpretable
    Limitations: Cannot capture trends or external factors
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize historical average model."""
        default_config = ModelConfig(
            name="HistoricalAverage",
            version="1.0.0",
            description="Historical average by hour and month"
        )
        super().__init__(config or default_config)
        
        self._hourly_monthly_avg: Optional[pd.Series] = None
        self._hourly_monthly_std: Optional[pd.Series] = None
        self._overall_avg: float = 0.0
        self._overall_std: float = 1.0
    
    def fit(self,
            train_data: pd.DataFrame,
            datetime_col: str = 'datetime',
            price_col: str = 'price',
            **kwargs) -> 'HistoricalAverageModel':
        """
        Fit model by computing historical averages.
        
        Args:
            train_data: Training data with datetime and price
            datetime_col: Name of datetime column
            price_col: Name of price column
            
        Returns:
            Self for method chaining
        """
        df = train_data.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df['hour'] = df[datetime_col].dt.hour
        df['month'] = df[datetime_col].dt.month
        
        # Compute averages by hour and month
        self._hourly_monthly_avg = df.groupby(['hour', 'month'])[price_col].mean()
        self._hourly_monthly_std = df.groupby(['hour', 'month'])[price_col].std()
        
        # Overall statistics as fallback
        self._overall_avg = df[price_col].mean()
        self._overall_std = df[price_col].std()
        
        self._is_fitted = True
        self._training_data = train_data
        self._model_params = {
            'n_samples': len(train_data),
            'overall_avg': self._overall_avg,
            'overall_std': self._overall_std,
        }
        
        return self
    
    def predict(self,
                forecast_horizon: int,
                start_date: Optional[datetime] = None,
                **kwargs) -> ForecastResult:
        """
        Generate predictions using historical averages.
        
        Args:
            forecast_horizon: Number of periods to forecast
            start_date: Start date for forecast
            
        Returns:
            ForecastResult with predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if start_date is None:
            start_date = self._training_data['datetime'].max() + pd.Timedelta(minutes=30)
        
        # Generate forecast timestamps
        datetime_range = pd.date_range(
            start=start_date,
            periods=forecast_horizon,
            freq='30min'
        )
        
        # Look up historical averages
        forecasts = []
        stds = []
        
        for dt in datetime_range:
            hour = dt.hour
            month = dt.month
            
            if (hour, month) in self._hourly_monthly_avg.index:
                forecasts.append(self._hourly_monthly_avg.loc[(hour, month)])
                stds.append(self._hourly_monthly_std.loc[(hour, month)])
            else:
                forecasts.append(self._overall_avg)
                stds.append(self._overall_std)
        
        forecasts = np.array(forecasts)
        stds = np.array(stds)
        
        # 90% confidence interval
        z = 1.645
        lower_bound = forecasts - z * stds
        upper_bound = forecasts + z * stds
        
        return ForecastResult(
            datetime=datetime_range,
            point_forecast=forecasts,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=0.90,
            metadata={'model': 'HistoricalAverage'}
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self._model_params
    
    def set_params(self, **params) -> 'HistoricalAverageModel':
        """Set model parameters."""
        self._model_params.update(params)
        return self
