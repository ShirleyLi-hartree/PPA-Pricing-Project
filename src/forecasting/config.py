"""
Configuration module for training, grid search, and phase-based workflows.

This module provides configuration dataclasses for:
- Rolling window training (RollingWindowConfig)
- Validation metrics (ValidationConfig)
- Grid search parameter spaces (GridSearchConfig)
- Phase 1 structure search (Phase1Config)
- Phase 2 rolling backtest (Phase2Config)
"""
from dataclasses import dataclass, field
from typing import Optional, List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



@dataclass
class RollingWindowConfig:
    """
    Configuration for rolling window training-validation.
    
    Attributes:
        train_years: Number of years in training window
        val_months: Number of months in validation window
        test_months: Number of months in test (backtest) window
        step_month: Step size for rolling window in months
        min_windows: Minimum number of windows required
    """
    train_years: int = 3
    val_months: int = 6
    test_months: int = 1
    step_month: int = 1  # 1-month step for higher resolution
    min_windows: int = 3
    
    def calculate_num_windows(self, data_years: int) -> int:
        """
        Calculate number of rolling windows given data span.
        
        Args:
            data_years: Total years of data available
            
        Returns:
            Number of possible rolling windows
        """
        # Approximate calculation for estimation only
        total_months = data_years * 12
        window_size_months = (self.train_years * 12) + self.val_months + self.test_months
        available_months = total_months - window_size_months
        
        if available_months < 0:
            return 0
            
        # Convert step to months approx
        return int(available_months / self.step_month) + 1


@dataclass  
class ValidationConfig:
    """
    Configuration for validation metrics.
    
    Attributes:
        metrics: List of metrics to compute
        granularities: Time granularities for analysis
        scenarios: Supply scenarios to validate
    """
    metrics: List[str] = field(default_factory=lambda: [
        'MAE', 'RMSE', 'MAPE', 'sMAPE', 'R2', 'MBE'
    ])
    granularities: List[str] = field(default_factory=lambda: [
        'overall', 'hourly', 'daily', 'monthly', 'quarterly'
    ])
    scenarios: List[str] = field(default_factory=lambda: [
        '24_7', 'peak_only', 'off_peak'
    ])


@dataclass
class GridSearchConfig:
    """
    Configuration for Grid Search parameter spaces.
    
    Defines the hyperparameter search spaces for each model type.
    These are used in Phase 1 structure search to find optimal model configurations.
    
    Attributes:
        sarimax_order_*: ARIMA order parameters (p, d, q)
        sarimax_seasonal_*: Seasonal ARIMA parameters (P, D, Q, s)
        prophet_*_prior: Prophet prior scale parameters
        xgb_*: XGBoost hyperparameters
    """
    # SARIMAX parameter space
    # Note: SimplifiedSARIMAX uses daily aggregation internally,
    # so seasonal_s=7 (weekly seasonality), NOT 48 (half-hourly daily cycle)
    sarimax_order_p: List[int] = field(default_factory=lambda: [0, 1, 2])
    sarimax_order_d: List[int] = field(default_factory=lambda: [0, 1])
    sarimax_order_q: List[int] = field(default_factory=lambda: [0, 1, 2])
    sarimax_seasonal_P: List[int] = field(default_factory=lambda: [0, 1, 2])
    sarimax_seasonal_D: List[int] = field(default_factory=lambda: [0, 1])
    sarimax_seasonal_Q: List[int] = field(default_factory=lambda: [0, 1, 2])
    sarimax_seasonal_s: int = 7  # Weekly cycle for daily-aggregated data
    
    # Prophet parameter space
    prophet_changepoint_prior: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.05, 0.1, 0.5])
    prophet_seasonality_prior: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    prophet_holidays_prior: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    
    # XGBoost parameter space
    xgb_max_depth: List[int] = field(default_factory=lambda: [3, 5, 7, 10])
    xgb_learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.1])
    xgb_subsample: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 1.0])
    xgb_colsample_bytree: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 1.0])
    xgb_n_estimators: int = 3000  # Increased from 1000
    xgb_early_stopping: int = 50  # Fixed
    
    def get_sarimax_param_grid(self) -> dict:
        """Get SARIMAX parameter grid for grid search."""
        return {
            'order_p': self.sarimax_order_p,
            'order_d': self.sarimax_order_d,
            'order_q': self.sarimax_order_q,
            'seasonal_P': self.sarimax_seasonal_P,
            'seasonal_D': self.sarimax_seasonal_D,
            'seasonal_Q': self.sarimax_seasonal_Q,
            'seasonal_s': [self.sarimax_seasonal_s],
        }
    
    def get_prophet_param_grid(self) -> dict:
        """Get Prophet parameter grid for grid search."""
        return {
            'changepoint_prior_scale': self.prophet_changepoint_prior,
            'seasonality_prior_scale': self.prophet_seasonality_prior,
            'holidays_prior_scale': self.prophet_holidays_prior,
        }
    
    def get_xgboost_param_grid(self) -> dict:
        """Get XGBoost parameter grid for grid search."""
        return {
            'max_depth': self.xgb_max_depth,
            'learning_rate': self.xgb_learning_rate,
            'subsample': self.xgb_subsample,
            'colsample_bytree': self.xgb_colsample_bytree,
        }


@dataclass
class Phase1Config:
    """
    Configuration for Phase 1 structure search.
    
    Phase 1 uses a 5-year data window (4 years training + 6 months validation)
    to search for optimal model hyperparameters.
    
    Attributes:
        search_years: Total years of data to use for search
        train_years: Years for training within search window
        val_months: Months for validation within search window
        loss_type: Loss function type ('mae', 'rmse', 'mape', 'hybrid')
        mae_weight: Weight for MAE/RMSE in Loss_GS
        worst_week_weight: Weight for worst week loss in Loss_GS
        bias_weight: Weight for absolute bias in Loss_GS
        output_dir: Directory to save optimal parameters
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    search_years: int = 5
    train_years: int = 4
    val_months: int = 6
    
    # Loss function type: 'rmse' (recommended) or 'mae' or 'hybrid'
    # RMSE penalizes large errors more than MAE
    loss_type: str = 'rmse'
    
    # Improved loss weights:
    # - Reduce worst_week weight to allow more variation
    # - Increase bias weight to ensure overall direction is correct
    # Loss_GS = loss_weight*Loss + worst_week_weight*WorstWeekMAE + bias_weight*|Bias|
    loss_weight: float = 1.0     # Primary loss (RMSE or MAE)
    worst_week_weight: float = 0  # Reduced from 0.2 to allow extreme predictions
    bias_weight: float = 0.0       # Keep bias check
    
    # Output configuration
    output_dir: str = 'config/optimal_params'
    
    # Parallel processing
    n_jobs: int = -1
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.search_years < self.train_years:
            raise ValueError("search_years must be >= train_years")
        
        if self.loss_type not in ['mae', 'rmse', 'mape', 'hybrid']:
            raise ValueError(f"loss_type must be one of ['mae', 'rmse', 'mape', 'hybrid'], got {self.loss_type}")
        
        total_weight = self.loss_weight + self.worst_week_weight + self.bias_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Loss weights must sum to 1.0, got {total_weight}")


@dataclass
class Phase2Config:
    """
    Configuration for Phase 2 rolling backtest.
    
    Phase 2 performs rolling window training and evaluation using
    the optimal parameters from Phase 1.
    
    Uses last 5 years of data (consistent with Phase 1 search window)
    with 3-month rolling step for efficiency.
    
    Attributes:
        data_years: Years of data to use (from most recent)
        train_years: Training window in years
        test_months: Test window in months
        step_month: Rolling step size in months
        tail_risk_weight: Weight for CVaR95 in composite score
        stability_weight: Weight for parameter stability in composite score
        coverage_weight: Weight for 90% coverage in composite score
        mae_weight: Weight for MAE in composite score
    """
    # Data range: use last N years (consistent with Phase 1)
    data_years: int = 5
    
    # Window configuration
    train_years: int = 3
    test_months: int = 1
    step_month: int = 1 
    
    # Composite score weights
    # CompositeScore = tail_risk_weight*CVaR + stability_weight*Stability + coverage_weight*Coverage + mae_weight*MAE
    tail_risk_weight: float = 0.4
    stability_weight: float = 0.3
    coverage_weight: float = 0.2
    mae_weight: float = 0.1
    
    def validate(self) -> None:
        """Validate configuration values."""
        total_weight = (self.tail_risk_weight + self.stability_weight + 
                       self.coverage_weight + self.mae_weight)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Composite score weights must sum to 1.0, got {total_weight}")
