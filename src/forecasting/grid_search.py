"""
Grid Search Engine for Phase 1 Structure Search.

This module provides a unified interface for hyperparameter optimization
across all supported model types (SARIMAX, Prophet, XGBoost).
"""
import os
import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Tuple

import numpy as np
import pandas as pd

# Optional yaml support - fall back to json if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.base.model_interface import BasePriceModel
from src.models.base.metrics import PricingMetrics
from src.forecasting.config import GridSearchConfig, Phase1Config

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """
    Result of a single grid search trial.
    
    Attributes:
        params: Parameter combination tested
        loss_gs: Combined loss value (Loss_GS)
        mae: Mean absolute error
        worst_week_mae: Worst week MAE
        bias: Mean bias error
        training_time: Time taken to train (seconds)
        converged: Whether model converged successfully
        metadata: Additional model-specific metrics
    """
    params: Dict[str, Any]
    loss_gs: float
    mae: float
    worst_week_mae: float
    bias: float
    training_time: float = 0.0
    converged: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridSearchResult:
    """
    Result of grid search optimization.
    
    Attributes:
        model_type: Type of model searched
        optimal_params: Best parameter combination found
        optimal_loss: Loss value of best parameters
        all_trials: List of all trial results
        search_time_seconds: Total search time
        data_range: Date range of data used
        validation_range: Date range of validation data
    """
    model_type: str
    optimal_params: Dict[str, Any]
    optimal_loss: float
    all_trials: List[TrialResult]
    search_time_seconds: float
    data_range: str
    validation_range: str
    search_metrics: Dict[str, Any] = field(default_factory=dict)


class GridSearchEngine:
    """
    Unified Grid Search Engine for model hyperparameter optimization.
    
    Supports three model types:
    - SARIMAX: Order (p,d,q) and seasonal order (P,D,Q,s) search
    - Prophet: Prior scale parameter search
    - XGBoost: Tree and boosting parameter search
    
    Uses Loss_GS = 0.5*MAE + 0.3*WorstWeekMAE + 0.2*|Bias| as objective.
    """
    
    def __init__(self,
                 phase1_config: Optional[Phase1Config] = None,
                 grid_config: Optional[GridSearchConfig] = None):
        """
        Initialize Grid Search Engine.
        
        Args:
            phase1_config: Phase 1 configuration (loss weights, output dir, etc.)
            grid_config: Grid search parameter space configuration
        """
        self.phase1_config = phase1_config or Phase1Config()
        self.grid_config = grid_config or GridSearchConfig()
        
        # Validate configuration
        self.phase1_config.validate()
    
    def search(self,
               model_type: str,
               train_data: pd.DataFrame,
               val_data: pd.DataFrame,
               datetime_col: str = 'datetime',
               price_col: str = 'price') -> GridSearchResult:
        """
        Execute grid search for specified model type.
        
        Args:
            model_type: One of 'sarimax', 'prophet', 'xgboost'
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            datetime_col: Name of datetime column
            price_col: Name of price column
            
        Returns:
            GridSearchResult with optimal parameters and all trials
        """
        import time
        start_time = time.time()
        
        model_type = model_type.lower()
        logger.info(f"Starting grid search for {model_type.upper()}")
        
        # Get parameter grid
        if model_type == 'sarimax':
            param_grid = self.grid_config.get_sarimax_param_grid()
            search_func = self._search_sarimax
        elif model_type == 'prophet':
            param_grid = self.grid_config.get_prophet_param_grid()
            search_func = self._search_prophet
        elif model_type == 'xgboost':
            param_grid = self.grid_config.get_xgboost_param_grid()
            search_func = self._search_xgboost
        else:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Supported: sarimax, prophet, xgboost")
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        logger.info(f"Total parameter combinations: {len(param_combinations)}")
        
        # Execute search
        all_trials = search_func(
            param_combinations,
            train_data,
            val_data,
            datetime_col,
            price_col
        )
        
        # Filter successful trials
        successful_trials = [t for t in all_trials if t.converged and not np.isnan(t.loss_gs)]
        
        if not successful_trials:
            logger.error("No successful trials! Using first trial as fallback.")
            successful_trials = all_trials[:1] if all_trials else []
        
        # Find optimal
        if successful_trials:
            best_trial = min(successful_trials, key=lambda t: t.loss_gs)
            optimal_params = best_trial.params
            optimal_loss = best_trial.loss_gs
        else:
            optimal_params = param_combinations[0] if param_combinations else {}
            optimal_loss = float('inf')
        
        total_time = time.time() - start_time
        
        # Prepare result
        data_start = train_data[datetime_col].min()
        data_end = val_data[datetime_col].max()
        val_start = val_data[datetime_col].min()
        val_end = val_data[datetime_col].max()
        
        result = GridSearchResult(
            model_type=model_type.upper(),
            optimal_params=optimal_params,
            optimal_loss=optimal_loss,
            all_trials=all_trials,
            search_time_seconds=total_time,
            data_range=f"{data_start.date()} to {data_end.date()}",
            validation_range=f"{val_start.date()} to {val_end.date()}",
            search_metrics={
                'total_trials': len(all_trials),
                'successful_trials': len(successful_trials),
                'convergence_rate': len(successful_trials) / len(all_trials) if all_trials else 0,
            }
        )
        
        logger.info(f"Grid search complete. Best loss: {optimal_loss:.4f}")
        logger.info(f"Optimal params: {optimal_params}")
        
        return result
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _search_sarimax(self,
                        param_combinations: List[Dict],
                        train_data: pd.DataFrame,
                        val_data: pd.DataFrame,
                        datetime_col: str,
                        price_col: str) -> List[TrialResult]:
        """Execute SARIMAX grid search."""
        from src.models.ml_enhanced_v3.sarimax_model import SimplifiedSARIMAX
        
        trials = []
        total = len(param_combinations)
        
        for i, params in enumerate(param_combinations):
            if (i + 1) % 10 == 0:
                logger.info(f"SARIMAX trial {i + 1}/{total}")
            
            try:
                import time
                start = time.time()
                
                # Build order tuples
                order = (params['order_p'], params['order_d'], params['order_q'])
                seasonal_order = (
                    params['seasonal_P'],
                    params['seasonal_D'],
                    params['seasonal_Q'],
                    params['seasonal_s']
                )
                
                # Create and train model
                model = SimplifiedSARIMAX(order=order, seasonal_order=seasonal_order)
                model.fit(train_data, datetime_col=datetime_col, price_col=price_col)
                
                training_time = time.time() - start
                
                # Predict on validation
                forecast = model.predict(
                    forecast_horizon=len(val_data),
                    start_date=val_data[datetime_col].iloc[0],
                    historical_data=train_data
                )
                
                # Calculate metrics
                y_true = val_data[price_col].values
                y_pred = forecast.point_forecast[:len(y_true)]
                datetime_index = pd.DatetimeIndex(val_data[datetime_col])
                
                # Calculate multiple loss metrics
                mae = PricingMetrics.mae(y_true, y_pred)
                rmse = PricingMetrics.rmse(y_true, y_pred)
                mape = PricingMetrics.mape(y_true, y_pred) if not np.any(y_true == 0) else mae
                worst_week = PricingMetrics.worst_week_mae(y_true, y_pred, datetime_index)
                bias = PricingMetrics.mean_bias_error(y_true, y_pred)
                
                # Select primary loss based on config
                loss_type = getattr(self.phase1_config, 'loss_type', 'mae')
                if loss_type == 'rmse':
                    primary_loss = rmse
                elif loss_type == 'mape':
                    primary_loss = mape
                elif loss_type == 'hybrid':
                    # Hybrid: combine RMSE and MAE
                    primary_loss = 0.5 * rmse + 0.5 * mae
                else:  # mae (default)
                    primary_loss = mae
                
                # Calculate Loss_GS with configurable weights
                loss_gs = primary_loss
                
                # Get model-specific metrics
                metadata = {}
                if hasattr(model, '_model') and model._model is not None:
                    try:
                        metadata['aic'] = model._model.aic
                    except:
                        pass
                
                trials.append(TrialResult(
                    params={'order': order, 'seasonal_order': seasonal_order},
                    loss_gs=loss_gs,
                    mae=mae,
                    worst_week_mae=worst_week,
                    bias=bias,
                    training_time=training_time,
                    converged=True,
                    metadata=metadata
                ))
                
            except Exception as e:
                logger.warning(f"SARIMAX trial failed: order={params}, error={str(e)[:100]}")
                trials.append(TrialResult(
                    params=params,
                    loss_gs=float('inf'),
                    mae=float('inf'),
                    worst_week_mae=float('inf'),
                    bias=0.0,
                    converged=False,
                    metadata={'error': str(e)[:200]}
                ))
        
        return trials
    
    def _search_prophet(self,
                        param_combinations: List[Dict],
                        train_data: pd.DataFrame,
                        val_data: pd.DataFrame,
                        datetime_col: str,
                        price_col: str) -> List[TrialResult]:
        """Execute Prophet grid search."""
        from src.models.ml_enhanced_v3.prophet_forecaster import ProphetForecaster
        
        trials = []
        total = len(param_combinations)
        
        for i, params in enumerate(param_combinations):
            if (i + 1) % 10 == 0:
                logger.info(f"Prophet trial {i + 1}/{total}")
            
            try:
                import time
                start = time.time()
                
                # Create model with parameters
                model = ProphetForecaster(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    holidays_prior_scale=params['holidays_prior_scale']
                )
                
                model.fit(train_data, datetime_col=datetime_col, price_col=price_col)
                
                training_time = time.time() - start
                
                # Predict on validation
                forecast = model.predict(
                    forecast_horizon=len(val_data),
                    start_date=val_data[datetime_col].iloc[0],
                    historical_data=train_data
                )
                
                # Calculate metrics
                y_true = val_data[price_col].values
                y_pred = forecast.point_forecast[:len(y_true)]
                datetime_index = pd.DatetimeIndex(val_data[datetime_col])
                
                # Calculate multiple loss metrics
                mae = PricingMetrics.mae(y_true, y_pred)
                rmse = PricingMetrics.rmse(y_true, y_pred)
                mape = PricingMetrics.mape(y_true, y_pred) if not np.any(y_true == 0) else mae
                worst_week = PricingMetrics.worst_week_mae(y_true, y_pred, datetime_index)
                bias = PricingMetrics.mean_bias_error(y_true, y_pred)
                
                # Select primary loss based on config
                loss_type = getattr(self.phase1_config, 'loss_type', 'mae')
                if loss_type == 'rmse':
                    primary_loss = rmse
                elif loss_type == 'mape':
                    primary_loss = mape
                elif loss_type == 'hybrid':
                    # Hybrid: combine RMSE and MAE
                    primary_loss = 0.5 * rmse + 0.5 * mae
                else:  # mae (default)
                    primary_loss = mae
                
                # Calculate Loss_GS with configurable weights
                loss_gs = primary_loss
                
                # Get model-specific metrics
                metadata = {}
                if hasattr(model, '_model') and model._model is not None:
                    try:
                        metadata['n_changepoints'] = len(model._model.changepoints)
                    except:
                        pass
                
                trials.append(TrialResult(
                    params=params,
                    loss_gs=loss_gs,
                    mae=mae,
                    worst_week_mae=worst_week,
                    bias=bias,
                    training_time=training_time,
                    converged=True,
                    metadata=metadata
                ))
                
            except Exception as e:
                logger.warning(f"Prophet trial failed: params={params}, error={str(e)[:100]}")
                trials.append(TrialResult(
                    params=params,
                    loss_gs=float('inf'),
                    mae=float('inf'),
                    worst_week_mae=float('inf'),
                    bias=0.0,
                    converged=False,
                    metadata={'error': str(e)[:200]}
                ))
        
        return trials
    
    def _search_xgboost(self,
                        param_combinations: List[Dict],
                        train_data: pd.DataFrame,
                        val_data: pd.DataFrame,
                        datetime_col: str,
                        price_col: str) -> List[TrialResult]:
        """Execute XGBoost grid search."""
        from src.models.ml_enhanced_v3.xgboost_forecaster import XGBoostForecaster
        
        trials = []
        total = len(param_combinations)
        
        for i, params in enumerate(param_combinations):
            if (i + 1) % 10 == 0:
                logger.info(f"XGBoost trial {i + 1}/{total}")
            
            try:
                import time
                start = time.time()
                
                # Create model with parameters
                model = XGBoostForecaster(
                    n_estimators=self.grid_config.xgb_n_estimators,
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate']
                )
                
                # Set additional params via model params
                model._model_params['subsample'] = params['subsample']
                model._model_params['colsample_bytree'] = params['colsample_bytree']
                
                model.fit(
                    train_data,
                    datetime_col=datetime_col,
                    price_col=price_col,
                    validation_data=val_data
                )
                
                training_time = time.time() - start
                
                # Predict on validation
                forecast = model.predict(
                    forecast_horizon=len(val_data),
                    start_date=val_data[datetime_col].iloc[0],
                    historical_data=train_data
                )
                
                # Calculate metrics
                y_true = val_data[price_col].values
                y_pred = forecast.point_forecast[:len(y_true)]
                datetime_index = pd.DatetimeIndex(val_data[datetime_col])
                
                # Calculate multiple loss metrics
                mae = PricingMetrics.mae(y_true, y_pred)
                rmse = PricingMetrics.rmse(y_true, y_pred)
                mape = PricingMetrics.mape(y_true, y_pred) if not np.any(y_true == 0) else mae
                worst_week = PricingMetrics.worst_week_mae(y_true, y_pred, datetime_index)
                bias = PricingMetrics.mean_bias_error(y_true, y_pred)
                
                # Select primary loss based on config
                loss_type = getattr(self.phase1_config, 'loss_type', 'mae')
                if loss_type == 'rmse':
                    primary_loss = rmse
                elif loss_type == 'mape':
                    primary_loss = mape
                elif loss_type == 'hybrid':
                    # Hybrid: combine RMSE and MAE
                    primary_loss = 0.5 * rmse + 0.5 * mae
                else:  # mae (default)
                    primary_loss = mae
                
                # Calculate Loss_GS with configurable weights
                loss_gs = primary_loss
                
                # Get model-specific metrics
                metadata = {
                    'n_estimators_used': self.grid_config.xgb_n_estimators
                }
                if hasattr(model, '_model') and model._model is not None:
                    try:
                        metadata['best_iteration'] = model._model.best_iteration
                    except:
                        pass
                
                # Get feature importance
                try:
                    importance = model.get_feature_importance(top_n=10)
                    if importance is not None:
                        metadata['top_features'] = importance['feature'].tolist()[:5]
                except:
                    pass
                
                trials.append(TrialResult(
                    params=params,
                    loss_gs=loss_gs,
                    mae=mae,
                    worst_week_mae=worst_week,
                    bias=bias,
                    training_time=training_time,
                    converged=True,
                    metadata=metadata
                ))
                
            except Exception as e:
                logger.warning(f"XGBoost trial failed: params={params}, error={str(e)[:100]}")
                trials.append(TrialResult(
                    params=params,
                    loss_gs=float('inf'),
                    mae=float('inf'),
                    worst_week_mae=float('inf'),
                    bias=0.0,
                    converged=False,
                    metadata={'error': str(e)[:200]}
                ))
        
        return trials


def save_optimal_params(result: GridSearchResult,
                        output_dir: str = 'config/optimal_params') -> str:
    """
    Save optimal parameters to YAML file.
    
    Args:
        result: GridSearchResult from grid search
        output_dir: Directory to save file
        
    Returns:
        Path to saved file
    """
    # Create directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    file_ext = 'yaml' if YAML_AVAILABLE else 'json'
    filename = f"{result.model_type.lower()}_optimal_{timestamp}.{file_ext}"
    filepath = output_path / filename
    
    # Prepare data for serialization
    data = {
        'metadata': {
            'model': result.model_type,
            'search_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_range': result.data_range,
            'validation_range': result.validation_range,
            'loss_gs': float(result.optimal_loss),
            'search_time_seconds': float(result.search_time_seconds),
        },
        'optimal_params': _convert_params_for_yaml(result.optimal_params),
        'search_metrics': result.search_metrics,
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        if YAML_AVAILABLE:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved optimal params to: {filepath}")
    return str(filepath)


def load_optimal_params(model_type: str,
                        input_dir: str = 'config/optimal_params',
                        date: Optional[str] = None) -> Dict[str, Any]:
    """
    Load optimal parameters from YAML or JSON file.
    
    Args:
        model_type: Model type (sarimax, prophet, xgboost)
        input_dir: Directory containing parameter files
        date: Specific date to load (YYYYMMDD format), None for latest
        
    Returns:
        Dictionary with optimal parameters
    """
    input_path = Path(input_dir)
    model_type = model_type.lower()
    
    if date:
        # Try both yaml and json extensions
        for ext in ['yaml', 'json']:
            filename = f"{model_type}_optimal_{date}.{ext}"
            filepath = input_path / filename
            if filepath.exists():
                break
        else:
            raise FileNotFoundError(f"Parameter file not found for {model_type} on {date}")
    else:
        # Find latest file (yaml or json)
        files = []
        for ext in ['yaml', 'json']:
            pattern = f"{model_type}_optimal_*.{ext}"
            files.extend(list(input_path.glob(pattern)))
        if not files:
            raise FileNotFoundError(f"No parameter files found for {model_type} in {input_dir}")
        filepath = max(files, key=lambda f: f.stat().st_mtime)
    
    with open(filepath, 'r') as f:
        if filepath.suffix == '.yaml':
            if YAML_AVAILABLE:
                data = yaml.safe_load(f)
            else:
                raise ImportError("YAML file found but PyYAML not installed")
        else:
            data = json.load(f)
    
    logger.info(f"Loaded optimal params from: {filepath}")
    return data


def _convert_params_for_yaml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parameters to YAML-serializable format."""
    result = {}
    for key, value in params.items():
        if isinstance(value, (tuple, list)):
            result[key] = list(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            result[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            result[key] = float(value)
        else:
            result[key] = value
    return result
