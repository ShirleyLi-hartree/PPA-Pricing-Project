from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np
from src.models.base.metrics import PricingMetrics
from src.utils.date_utils import JapanDateUtils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """
    Rolling window backtester for time series models.
    
    Supports both expanding window and rolling window training strategies,
    with configurable refit frequency to balance accuracy and computation.
    """
    
    def __init__(self,
                 model_class: Any,
                 model_params: Dict[str, Any],
                 feature_engineer: Any,
                 target_col: str = 'price',
                 datetime_col: str = 'datetime',
                 initial_train_size: int = 365 * 48 * 3,  # 3 years of half-hourly data
                 window_size: int = 90 * 48,              # 90 days (3 months) forecast
                 step_size: int = 30 * 48,                # 30 days step
                 refit_frequency: Optional[int] = None,   # Refit every N folds (None = every fold)
                 use_expanding_window: bool = True,       # True = expanding, False = rolling
                 ):
        """
        Initialize backtester.
        
        Args:
            model_class: Class of the model to instantiate (must have fit/predict methods)
            model_params: Parameters for model instantiation
            feature_engineer: FeatureEngineer instance
            target_col: Target column name
            datetime_col: Datetime column name
            initial_train_size: Initial training window size (number of rows)
                Default: 3 years (365*48*3) for medium-term forecasting
            window_size: Forecast window size (number of rows) for each step
                Default: 90 days (90*48) for 3-month PPA pricing
            step_size: Step size for moving the window (number of rows)
                Default: 30 days (30*48) for monthly rolling
            refit_frequency: Refit model every N folds (None = every fold)
                Set to higher value to reduce computation time
            use_expanding_window: If True, use all historical data for training.
                If False, use fixed-size rolling window.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.feature_engineer = feature_engineer
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.initial_train_size = initial_train_size
        self.window_size = window_size
        self.step_size = step_size
        self.refit_frequency = refit_frequency
        self.use_expanding_window = use_expanding_window
        self.results = []
        self._current_model = None  # Cache model for refit_frequency
        self._date_utils = JapanDateUtils()
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtesting simulation.
        
        Args:
            df: DataFrame containing all data (will be split into train/test)
            
        Returns:
            DataFrame with backtest results (metrics per fold)
        """
        df = df.sort_values(self.datetime_col).reset_index(drop=True)
        n_samples = len(df)
        
        current_idx = self.initial_train_size
        
        logger.info(f"Starting backtest with {n_samples} samples...")
        logger.info(f"Config: train_size={self.initial_train_size}, window={self.window_size}, step={self.step_size}")
        
        fold = 0
        self._current_model = None
        
        while current_idx + self.window_size <= n_samples:
            # Define train indices based on window strategy
            if self.use_expanding_window:
                # Expanding window: use all data up to current_idx
                train_df = df.iloc[:current_idx].copy()
            else:
                # Rolling window: use fixed size window
                train_start = max(0, current_idx - self.initial_train_size)
                train_df = df.iloc[train_start:current_idx].copy()
            
            test_df = df.iloc[current_idx : current_idx + self.window_size].copy()
            
            test_start = test_df[self.datetime_col].min()
            test_end = test_df[self.datetime_col].max()
            
            logger.info(f"Fold {fold}: Train size {len(train_df)}, Test range {test_start} - {test_end}")
            
            # Determine if we need to refit the model
            need_refit = (
                self._current_model is None or
                self.refit_frequency is None or
                fold % self.refit_frequency == 0
            )
            
            try:
                if need_refit:
                    # Initialize and train new model
                    model = self.model_class(**self.model_params)
                    model.fit(
                        train_data=train_df,
                        datetime_col=self.datetime_col,
                        price_col=self.target_col
                    )
                    self._current_model = model
                    logger.info(f"Fold {fold}: Model refitted")
                else:
                    model = self._current_model
                    logger.info(f"Fold {fold}: Using cached model")
                
                # Predict
                if hasattr(model, 'predict') and 'forecast_horizon' in model.predict.__code__.co_varnames:
                    forecast_result = model.predict(
                        forecast_horizon=len(test_df),
                        start_date=test_start,
                        historical_data=train_df
                    )
                    y_pred = forecast_result.point_forecast
                else:
                    y_pred = model.predict(test_df)
                
                y_true = test_df[self.target_col].values
                
                # Calculate metrics
                mae = PricingMetrics.mae(y_true, y_pred)
                rmse = PricingMetrics.rmse(y_true, y_pred)
                r2 = PricingMetrics.r_squared(y_true, y_pred)
                
                # Calculate additional metrics
                mbe = PricingMetrics.mean_bias_error(y_true, y_pred)
                
                self.results.append({
                    'fold': fold,
                    'test_start': test_start,
                    'test_end': test_end,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mbe': mbe,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'refitted': need_refit,
                })
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {str(e)}")
            
            current_idx += self.step_size
            fold += 1
            
        return pd.DataFrame(self.results)
    
    def seasonal_backtest(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run backtest separately for each season to assess model stability.
        
        This helps identify if the model performs differently across seasons,
        which is critical for Japanese electricity market with distinct
        summer (AC) and winter (heating) demand patterns.
        
        Args:
            df: DataFrame containing all data
            
        Returns:
            Dictionary mapping season names to their backtest results
        """
        df = df.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df['month'] = df[self.datetime_col].dt.month
        
        seasonal_results = {}
        
        for season_name, months in self._date_utils.SEASONS.items():
            logger.info(f"Running seasonal backtest for {season_name} (months: {months})")
            
            # Filter data to only include the target season months
            season_df = df[df['month'].isin(months)].copy()
            
            if len(season_df) < self.initial_train_size + self.window_size:
                logger.warning(f"Insufficient data for {season_name} backtest, skipping...")
                continue
            
            # Reset results for this season
            self.results = []
            
            # Run backtest on seasonal data
            results_df = self.run(season_df)
            results_df['season'] = season_name
            
            seasonal_results[season_name] = results_df
            
        return seasonal_results
    
    def get_seasonal_summary(self, seasonal_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics comparing model performance across seasons.
        
        Args:
            seasonal_results: Output from seasonal_backtest()
            
        Returns:
            DataFrame with summary metrics per season
        """
        summaries = []
        
        for season_name, results_df in seasonal_results.items():
            if len(results_df) == 0:
                continue
                
            summary = {
                'season': season_name,
                'n_folds': len(results_df),
                'mae_mean': results_df['mae'].mean(),
                'mae_std': results_df['mae'].std(),
                'rmse_mean': results_df['rmse'].mean(),
                'rmse_std': results_df['rmse'].std(),
                'r2_mean': results_df['r2'].mean(),
                'mbe_mean': results_df['mbe'].mean() if 'mbe' in results_df.columns else None,
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def get_backtest_summary(self) -> Dict[str, float]:
        """
        Get summary statistics from the backtest results.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        return {
            'n_folds': len(results_df),
            'mae_mean': results_df['mae'].mean(),
            'mae_std': results_df['mae'].std(),
            'mae_min': results_df['mae'].min(),
            'mae_max': results_df['mae'].max(),
            'rmse_mean': results_df['rmse'].mean(),
            'rmse_std': results_df['rmse'].std(),
            'r2_mean': results_df['r2'].mean(),
            'r2_std': results_df['r2'].std(),
            'mbe_mean': results_df['mbe'].mean() if 'mbe' in results_df.columns else None,
        }
