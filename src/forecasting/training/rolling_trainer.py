"""
Rolling Window Trainer - Training module completely separated from validation.

This module handles the generation of rolling windows and model training.
It does NOT perform any validation or metric computation.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import logging
import copy
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import JapanRegion
from src.models.base.model_interface import BasePriceModel

logger = logging.getLogger(__name__)


@dataclass
class TrainWindow:
    """
    Container for a single training window.
    
    Attributes:
        window_id: Unique identifier for this window
        train_start: Start date of training period
        train_end: End date of training period
        val_start: Start date of validation period
        val_end: End date of validation period
        test_start: Start date of test (backtest) period
        test_end: End date of test (backtest) period
        train_data: Training DataFrame
        val_data: Validation DataFrame (for early stopping/tuning)
        test_data: Test DataFrame (for backtest metrics)
    """
    window_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: Optional[pd.DataFrame] = None
    val_data: Optional[pd.DataFrame] = None
    test_data: Optional[pd.DataFrame] = None
    
    def __repr__(self) -> str:
        return (f"TrainWindow(id={self.window_id}, "
                f"train=[{self.train_start.date()} to {self.train_end.date()}], "
                f"val=[{self.val_start.date()} to {self.val_end.date()}], "
                f"test=[{self.test_start.date()} to {self.test_end.date()}])")
    
    @property
    def train_size(self) -> int:
        """Number of samples in training data."""
        return len(self.train_data) if self.train_data is not None else 0
    
    @property
    def val_size(self) -> int:
        """Number of samples in validation data."""
        return len(self.val_data) if self.val_data is not None else 0
        
    @property
    def test_size(self) -> int:
        """Number of samples in test data."""
        return len(self.test_data) if self.test_data is not None else 0


@dataclass
class TrainedModelResult:
    """
    Container for trained model results.
    
    Attributes:
        window: The training window used
        model: The trained model instance
        model_name: Name of the model
        training_time_seconds: Time taken to train
        metadata: Additional training metadata
    """
    window: TrainWindow
    model: BasePriceModel
    model_name: str
    training_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self.model.is_fitted if hasattr(self.model, 'is_fitted') else True


class RollingWindowTrainer:
    """
    Rolling Window Trainer - Responsible ONLY for training.
    
    This class implements rolling window training strategy where:
    - Train: 3 years
    - Test: 1 month (for backtesting)
    - Step: 1 month
    
    IMPORTANT: This class does NOT perform metric evaluation.
    Validation/Evaluation is handled by the separate ValidationMatrix class.
    """
    
    def __init__(self,
                 train_years: int = 3,
                 val_months: int = 0,
                 test_months: int = 1,
                 step_month: int = 3,
                 region: JapanRegion = JapanRegion.TOKYO):
        """
        Initialize rolling window trainer.
        
        Args:
            train_years: Number of years in training window
            val_months: Number of months in validation window
            test_months: Number of months in test window
            step_month: Step size in months (default 1 for higher resolution)
            region: Target Japanese electricity market region
        """
        self.train_years = train_years
        self.val_months = val_months
        self.test_months = test_months
        self.step_month = step_month
        self.region = region
        
        # Will be set after window generation
        self.windows: List[TrainWindow] = []
    
    def generate_windows(self,
                         data: pd.DataFrame,
                         datetime_col: str = 'datetime') -> List[TrainWindow]:
        """
        Generate all rolling windows from the data.
        
        Args:
            data: Full historical data
            datetime_col: Name of datetime column
            
        Returns:
            List of TrainWindow objects with train/val/test data assigned
        """
        data = data.copy()
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        data = data.sort_values(datetime_col).reset_index(drop=True)
        
        # Get data range
        data_start = data[datetime_col].min()
        data_end = data[datetime_col].max()
        
        logger.info(f"Data range: {data_start.date()} to {data_end.date()}")
        logger.info(f"Generating rolling windows with {self.train_years}y train, "
                    f"{self.val_months}m val, {self.test_months}m test, {self.step_month}m step")
        
        windows = []
        window_id = 0
        
        # Start from the earliest possible window
        current_train_start = data_start
        
        while True:
            # Calculate window boundaries
            train_start = current_train_start
            train_end = train_start + pd.DateOffset(years=self.train_years) - pd.Timedelta(seconds=1)
            
            val_start = train_start + pd.DateOffset(years=self.train_years)
            val_end = val_start + pd.DateOffset(months=self.val_months) - pd.Timedelta(seconds=1)
            
            test_start = val_start + pd.DateOffset(months=self.val_months)
            test_end = test_start + pd.DateOffset(months=self.test_months) - pd.Timedelta(seconds=1)
            
            # Check if we have enough data for this window
            if test_end > data_end:
                logger.info(f"Stopping window generation: test_end ({test_end.date()}) > data_end ({data_end.date()})")
                break
            
            # Extract data for this window
            train_mask = (data[datetime_col] >= train_start) & (data[datetime_col] <= train_end)
            val_mask = (data[datetime_col] >= val_start) & (data[datetime_col] <= val_end)
            test_mask = (data[datetime_col] >= test_start) & (data[datetime_col] <= test_end)
            
            train_data = data[train_mask].copy()
            val_data = data[val_mask].copy()
            test_data = data[test_mask].copy()
            
            # Create window object
            window = TrainWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )
            
            windows.append(window)
            if window_id % 10 == 0:  # Log every 10 windows to avoid spam
                logger.info(f"Created {window}")
            
            # Move to next window
            current_train_start += pd.DateOffset(months=self.step_month)
            window_id += 1
        
        self.windows = windows
        logger.info(f"Generated {len(windows)} rolling windows")
        
        return windows
    
    def train_on_window(self,
                        model: BasePriceModel,
                        window: TrainWindow,
                        datetime_col: str = 'datetime',
                        price_col: str = 'price',
                        **kwargs) -> TrainedModelResult:
        """
        Train a model on a single window.
        
        This method ONLY trains the model - no validation is performed.
        
        Args:
            model: Model instance to train (will be deep copied)
            window: Training window with data
            datetime_col: Name of datetime column
            price_col: Name of price column
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            TrainedModelResult containing the trained model
        """
        import time
        
        if window.train_data is None or len(window.train_data) == 0:
            raise ValueError(f"Window {window.window_id} has no training data")
        
        # Deep copy model to avoid modifying the original
        model_copy = copy.deepcopy(model)
        model_name = model_copy.__class__.__name__
        
        logger.info(f"Training {model_name} on window {window.window_id} "
                    f"({window.train_size} samples)")
        
        start_time = time.time()
        
        # Train the model
        # Check if model supports validation data in fit (e.g. XGBoost)
        fit_kwargs = kwargs.copy()
        
        # If model is XGBoost (or supports eval_set), we can pass validation data
        # Note: We need to check the model type or signature, but for now we'll 
        # try to pass it if the model looks like it might accept it, or rely on 
        # the model implementation to handle extra kwargs.
        # However, standard sklearn interface doesn't take eval_set.
        # So we'll specifically check for XGBoostForecaster or pass it if validation_data is present.
        
        if window.val_data is not None and len(window.val_data) > 0:
            # We add this to kwargs. The model's fit method needs to handle it.
            # Since we updated XGBoostForecaster to split internally, we might want to 
            # update it to accept external validation data instead.
            # For now, we'll pass it as 'validation_data' and let the model handle it if updated.
            fit_kwargs['validation_data'] = window.val_data
        
        model_copy.fit(
            train_data=window.train_data,
            datetime_col=datetime_col,
            price_col=price_col,
            **fit_kwargs
        )
        
        training_time = time.time() - start_time
        
        logger.info(f"Completed training {model_name} on window {window.window_id} "
                    f"in {training_time:.2f}s")
        
        return TrainedModelResult(
            window=window,
            model=model_copy,
            model_name=model_name,
            training_time_seconds=training_time,
            metadata={
                'train_samples': window.train_size,
                'train_start': str(window.train_start.date()),
                'train_end': str(window.train_end.date()),
            }
        )
    
    def train_all_windows(self,
                          model: BasePriceModel,
                          datetime_col: str = 'datetime',
                          price_col: str = 'price',
                          **kwargs) -> List[TrainedModelResult]:
        """
        Train a model on all generated windows.
        
        Args:
            model: Model template to train (will be deep copied for each window)
            datetime_col: Name of datetime column
            price_col: Name of price column
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            List of TrainedModelResult for each window
        """
        if not self.windows:
            raise ValueError("No windows generated. Call generate_windows() first.")
        
        results = []
        model_name = model.__class__.__name__
        
        logger.info(f"Training {model_name} on {len(self.windows)} windows")
        
        total_windows = len(self.windows)
        for idx, window in enumerate(self.windows, 1):
            try:
                logger.info(f"Training window {idx}/{total_windows}: ID={window.window_id}, "
                            f"train=[{window.train_start.date()} to {window.train_end.date()}], "
                            f"test=[{window.test_start.date()} to {window.test_end.date()}]")
                result = self.train_on_window(
                    model=model,
                    window=window,
                    datetime_col=datetime_col,
                    price_col=price_col,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error training on window {window.window_id}: {str(e)}")
                raise
        
        total_time = sum(r.training_time_seconds for r in results)
        logger.info(f"Completed training {model_name} on all windows. "
                    f"Total time: {total_time:.2f}s")
        
        return results
    
    def train_multiple_models(self,
                              models: Dict[str, BasePriceModel],
                              datetime_col: str = 'datetime',
                              price_col: str = 'price',
                              **kwargs) -> Dict[str, List[TrainedModelResult]]:
        """
        Train multiple models on all windows.
        
        Args:
            models: Dictionary mapping model names to model instances
            datetime_col: Name of datetime column
            price_col: Name of price column
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            Dictionary mapping model names to lists of TrainedModelResult
        """
        if not self.windows:
            raise ValueError("No windows generated. Call generate_windows() first.")
        
        all_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training model: {model_name}")
            logger.info(f"{'='*50}")
            
            try:
                results = self.train_all_windows(
                    model=model,
                    datetime_col=datetime_col,
                    price_col=price_col,
                    **kwargs
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                all_results[model_name] = []
        
        return all_results
    
    def get_final_training_data(self,
                                data: pd.DataFrame,
                                datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        Get the final training data for production model.
        
        Uses the most recent train_years of data.
        
        Args:
            data: Full historical data
            datetime_col: Name of datetime column
            
        Returns:
            Training data for the final production model
        """
        data = data.copy()
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        data = data.sort_values(datetime_col)
        
        data_end = data[datetime_col].max()
        train_start = data_end - pd.DateOffset(years=self.train_years)
        
        final_train_data = data[data[datetime_col] >= train_start].copy()
        
        logger.info(f"Final training data: {train_start.date()} to {data_end.date()} "
                    f"({len(final_train_data)} samples)")
        
        return final_train_data
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of the trainer configuration and windows."""
        return {
            'train_years': self.train_years,
            'val_months': self.val_months,
            'test_months': self.test_months,
            'step_month': self.step_month,
            'region': self.region.value,
            'num_windows': len(self.windows),
            'windows': [
                {
                    'id': w.window_id,
                    'train_range': f"{w.train_start.date()} to {w.train_end.date()}",
                    'val_range': f"{w.val_start.date()} to {w.val_end.date()}",
                    'test_range': f"{w.test_start.date()} to {w.test_end.date()}",
                    'train_size': w.train_size,
                    'val_size': w.val_size,
                    'test_size': w.test_size,
                }
                for w in self.windows
            ]
        }
    
    @classmethod
    def create_for_model(
        cls,
        model_type: str,
        region: JapanRegion = JapanRegion.TOKYO,
        phase: str = 'phase2'
    ) -> 'RollingWindowTrainer':
        """
        Factory method to create trainer with model-specific configurations.
        
        Creates a RollingWindowTrainer with appropriate window settings based on
        the model type and execution phase. Different models have different optimal
        window configurations for training and validation.
        
        Phase 1 (structure search):
            - SARIMAX: 4yr train + 6mo val (for auto_arima search)
            - Prophet: 4yr train + 6mo val (for prior scale search)
            - XGBoost: 4yr train + 6mo val (for hyperparameter tuning)
        
        Phase 2 (rolling backtest):
            - SARIMAX: 3yr train + 3mo test (no separate val)
            - Prophet: 3yr train + 3mo test (no separate val)
            - XGBoost: 3yr train + 3mo test (needs val for early stopping)
        
        Args:
            model_type: Type of model ('sarimax', 'prophet', 'xgboost')
            region: Target Japanese electricity market region
            phase: Execution phase ('phase1' or 'phase2')
            
        Returns:
            RollingWindowTrainer configured for the specified model type
            
        Raises:
            ValueError: If model_type or phase is not recognized
        """
        model_type_lower = model_type.lower()
        phase_lower = phase.lower()
        
        if phase_lower not in ('phase1', 'phase2'):
            raise ValueError(f"Unknown phase: {phase}. Expected 'phase1' or 'phase2'")
        
        # Phase 1: Structure search configurations
        if phase_lower == 'phase1':
            if model_type_lower in ('sarimax', 'prophet', 'xgboost'):
                # All models use same window for structure search
                return cls(
                    train_years=4,
                    val_months=6,
                    test_months=0,  # No test in phase1
                    step_month=3,   # Coarser step for phase1
                    region=region
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}. "
                               f"Expected 'sarimax', 'prophet', or 'xgboost'")
        
        # Phase 2: Rolling backtest configurations
        if model_type_lower == 'sarimax':
            logger.info(f"Creating RollingWindowTrainer for SARIMAX (phase2): "
                        f"3yr train + 0mo val + 1mo test")
            return cls(
                train_years=3,
                val_months=0,  # SARIMAX doesn't need validation window
                test_months=1,
                step_month=1,  # 1-month step for higher resolution
                region=region
            )
        
        elif model_type_lower == 'prophet':
            logger.info(f"Creating RollingWindowTrainer for Prophet (phase2): "
                        f"3yr train + 0mo val + 1mo test")
            return cls(
                train_years=3,
                val_months=0,  # Prophet doesn't need validation window
                test_months=1,
                step_month=1,  # 1-month step for higher resolution
                region=region
            )
        
        elif model_type_lower == 'xgboost':
            logger.info(f"Creating RollingWindowTrainer for XGBoost (phase2): "
                        f"3yr (train + val) + 1mo test")
            return cls(
                train_years=3, # XGBoost needs validation for early stopping - split internally
                val_months=0,  
                test_months=1,
                step_month=1,  # 1-month step for higher resolution
                region=region
            )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}. "
                           f"Expected 'sarimax', 'prophet', or 'xgboost'")
