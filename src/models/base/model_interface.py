"""
Base model interface for price forecasting models.

Defines abstract statistics_v1 class that all price forecasting models must implement.
"""
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ModelConfig:
    """Base configuration for pricing models."""
    name: str
    version: str = "1.0.0"
    description: str = ""


@dataclass
class ForecastResult:
    """Container for forecast results."""
    datetime: pd.DatetimeIndex
    point_forecast: np.ndarray  # Point forecast (e.g., mean)
    lower_bound: Optional[np.ndarray] = None  # Lower prediction interval
    upper_bound: Optional[np.ndarray] = None  # Upper prediction interval
    confidence_level: float = 0.90  # Confidence level for intervals
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast result to DataFrame."""
        df = pd.DataFrame({
            'datetime': self.datetime,
            'forecast': self.point_forecast,
        })
        
        if self.lower_bound is not None:
            df['lower_bound'] = self.lower_bound
        if self.upper_bound is not None:
            df['upper_bound'] = self.upper_bound
            
        return df


class BasePriceModel(ABC):
    """
    Abstract statistics_v1 class for electricity price forecasting models.
    
    All price forecasting models (V1 statistical, V2 fundamental_v2, V3 ML)
    should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize statistics_v1 model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig(name="BaseModel")
        self._is_fitted = False
        self._training_data: Optional[pd.DataFrame] = None
        self._model_params: Dict[str, Any] = {}
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    @abstractmethod
    def fit(self, 
            train_data: pd.DataFrame,
            **kwargs) -> 'BasePriceModel':
        """
        Fit the model to training data.
        
        Args:
            train_data: DataFrame with 'datetime' and 'price' columns
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self,
                forecast_horizon: int,
                start_date: Optional[datetime] = None,
                **kwargs) -> ForecastResult:
        """
        Generate price forecasts.
        
        Args:
            forecast_horizon: Number of periods to forecast
            start_date: Start date for forecast (defaults to end of training data)
            **kwargs: Additional model-specific parameters
            
        Returns:
            ForecastResult with point forecasts and optional intervals
        """
        pass
    
    def predict_range(self,
                      start_date: Union[str, datetime, date],
                      end_date: Union[str, datetime, date],
                      freq: str = '30min',
                      **kwargs) -> ForecastResult:
        """
        Generate forecasts for a specific date range.
        
        Args:
            start_date: Start date
            end_date: End date
            freq: Frequency of forecasts
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult for the date range
        """
        datetime_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        horizon = len(datetime_range)
        
        return self.predict(
            forecast_horizon=horizon,
            start_date=pd.Timestamp(start_date),
            **kwargs
        )
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'BasePriceModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If data is invalid
        """
        required_cols = ['datetime', 'price']
        
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if data['price'].isna().any():
            raise ValueError("Price column contains NaN values")
        
        return True
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        model_state = {
            'config': self.config,
            'is_fitted': self._is_fitted,
            'model_params': self._model_params,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BasePriceModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        instance = cls(config=model_state['config'])
        instance._is_fitted = model_state['is_fitted']
        instance._model_params = model_state['model_params']
        
        return instance
    
    def cross_validate(self,
                       data: pd.DataFrame,
                       n_splits: int = 5,
                       test_size: int = 48 * 30,  # 30 days
                       gap: int = 0) -> List[Dict[str, float]]:
        """
        Perform time series cross-validation.
        
        Uses expanding window approach suitable for time series.
        
        Args:
            data: Full dataset
            n_splits: Number of CV splits
            test_size: Size of test set in periods
            gap: Gap between train and test (to avoid leakage)
            
        Returns:
            List of metrics dictionaries for each fold
        """
        from src.models.base.metrics import PricingMetrics
        
        n_samples = len(data)
        results = []
        
        # Calculate split points
        total_test_size = n_splits * test_size
        train_start_size = n_samples - total_test_size - (n_splits - 1) * gap
        
        if train_start_size < test_size:
            raise ValueError("Not enough data for requested CV configuration")
        
        for i in range(n_splits):
            # Training data ends at different points for each fold
            train_end = train_start_size + i * (test_size + gap)
            test_start = train_end + gap
            test_end = test_start + test_size
            
            train_data = data.iloc[:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Fit and predict
            self.fit(train_data)
            forecast = self.predict(
                forecast_horizon=len(test_data),
                start_date=test_data['datetime'].iloc[0]
            )
            
            # Calculate metrics
            actual = test_data['price'].values
            predicted = forecast.point_forecast
            
            metrics = PricingMetrics.calculate_all_metrics(actual, predicted)
            metrics['fold'] = i
            results.append(metrics)
        
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(config={self.config}, fitted={self._is_fitted})"
