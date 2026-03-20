"""
XGBoost forecaster for electricity price prediction.

Implements gradient boosting model with hyperparameter tuning
for price forecasting.
"""
from datetime import datetime
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.base.model_interface import BasePriceModel, ModelConfig, ForecastResult
from src.data.preprocessors.feature_engineer import FeatureEngineer


class XGBoostForecaster(BasePriceModel):
    """
    XGBoost-based electricity price forecaster.
    
    Features:
    - Gradient boosting for non-linear relationships
    - Hyperparameter tuning via grid search or Bayesian optimization
    - Feature importance analysis
    - Probabilistic forecasts via quantile regression
    """
    
    def __init__(self,
                 n_estimators: int = 5000,
                 max_depth: int = 10,
                 learning_rate: float = 0.01,
                 config: Optional[ModelConfig] = None,
                 extra_lag_cols: Optional[List[str]] = None):
        """
        Initialize XGBoost forecaster.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            config: Model configuration
            extra_lag_cols: List of additional columns to create lag/rolling features for
        """
        default_config = ModelConfig(
            name="XGBoostForecaster",
            version="3.0.0",
            description="XGBoost for electricity price forecasting"
        )
        super().__init__(config or default_config)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.extra_lag_cols = extra_lag_cols
        
        self.feature_engineer = FeatureEngineer()
        self._model = None
        self._quantile_models: Dict[float, Any] = {}
        self._feature_names: List[str] = []
        self._feature_importance: Optional[pd.DataFrame] = None
    
    def fit(self,
            train_data: pd.DataFrame,
            datetime_col: str = 'datetime',
            price_col: str = 'price',
            fuel_prices: Optional[pd.DataFrame] = None,
            tune_hyperparams: bool = False,
            **kwargs) -> 'XGBoostForecaster':
        """
        Fit XGBoost model.
        
        Args:
            train_data: DataFrame with datetime and price
            datetime_col: Name of datetime column
            price_col: Name of price column
            fuel_prices: Optional fuel price data
            tune_hyperparams: Whether to tune hyperparameters
            
        Returns:
            Self for method chaining
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required. Install with: pip install xgboost")
        
        extra_lag_cols = kwargs.get('extra_lag_cols')
        if extra_lag_cols is not None:
            self.extra_lag_cols = extra_lag_cols
        
        # DEBUG
        print(f"DEBUG: fit called. train_data shape: {train_data.shape}")
        print(f"DEBUG: train_data columns: {train_data.columns.tolist()}")
        
        # Pre-calculate global statistics to avoid leakage and provide strong features
        self.feature_engineer.fit_statistics(
            train_data.rename(columns={price_col: 'price', datetime_col: 'datetime'}),
            target_col='price'
        )
        
        # Create features
        featured_df = self.feature_engineer.create_all_features(
            train_data.rename(columns={price_col: 'price', datetime_col: 'datetime'}),
            target_col='price',
            datetime_col='datetime',
            fuel_prices=fuel_prices,
            extra_lag_cols=self.extra_lag_cols
        )
        
        # Prepare training data
        exclude_cols = ['datetime', 'price', 'date', 'region']
        self._feature_names = [col for col in featured_df.columns if col not in exclude_cols]
        
        X = featured_df[self._feature_names].values
        y = featured_df['price'].values
        
        # Check for external validation data
        validation_data = kwargs.get('validation_data')
        X_val_ext = None
        y_val_ext = None
        
        if validation_data is not None:
            # Process validation data similarly to training data
            print(f"DEBUG: Using external validation data ({len(validation_data)} samples)")
            val_featured = self.feature_engineer.create_all_features(
                validation_data.rename(columns={price_col: 'price', datetime_col: 'datetime'}),
                target_col='price',
                datetime_col='datetime',
                fuel_prices=fuel_prices,
                extra_lag_cols=self.extra_lag_cols
            )
            # Use only features that exist in both train and validation sets
            common_features = [f for f in self._feature_names if f in val_featured.columns]
            if len(common_features) < len(self._feature_names):
                missing = set(self._feature_names) - set(common_features)
                print(f"DEBUG: Dropping {len(missing)} features not in validation: {missing}")
                self._feature_names = common_features
                X = featured_df[self._feature_names].values
            X_val_ext = val_featured[self._feature_names].values
            y_val_ext = val_featured['price'].values
        
        # Tune hyperparameters if requested
        if tune_hyperparams:
            best_params = self._tune_hyperparameters(X, y)
            self.n_estimators = best_params.get('n_estimators', self.n_estimators)
            self.max_depth = best_params.get('max_depth', self.max_depth)
            self.learning_rate = best_params.get('learning_rate', self.learning_rate)
        
        # Train model
        # Use parameters from _model_params if available, otherwise use defaults
        model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'early_stopping_rounds': 50,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with any additional parameters from _model_params
        if hasattr(self, '_model_params') and self._model_params:
            model_params.update(self._model_params)
        
        self._model = xgb.XGBRegressor(**model_params)
        
        # Determine validation set - use internal 85/15 split for early stopping
        # This is more appropriate for time series data
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        eval_set = [(X_train, y_train), (X_val, y_val)]
        print(f"XGBoost training: {len(X_train)} train samples, {len(X_val)} internal validation samples")
        
        # Fit with early stopping
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100  # Log every 100 rounds
        )

        # Log training results
        if hasattr(self._model, 'best_iteration'):
            print(f"XGBoost Best iteration: {self._model.best_iteration}")
            print(f"XGBoost Best validation MAE: {self._model.best_score}")

        # Calculate feature importance
        importance = self._model.feature_importances_
        self._feature_importance = pd.DataFrame({
            'feature': self._feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTOP 15 FEATURES (XGBoost):")
        print(self._feature_importance.head(15))
        print("\n")
        
        # Train quantile models for prediction intervals
        self._train_quantile_models(X, y)
        
        self._training_data = featured_df
        self._is_fitted = True
        
        # Store model parameters
        self._model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_features': len(self._feature_names),
        }
        
        return self
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Tune hyperparameters using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Best hyperparameters
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
        except ImportError:
            return {}
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        }
        
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_
    
    def _train_quantile_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train quantile regression models for prediction intervals.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        try:
            import xgboost as xgb
        except ImportError:
            return
        
        for quantile in [0.10, 0.90]:
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective='reg:quantileerror',
                quantile_alpha=quantile,
                random_state=42,
                n_jobs=-1
            )
            
            try:
                model.fit(X, y)
                self._quantile_models[quantile] = model
            except Exception:
                # Fallback: simple quantile estimation
                pass
    
    def predict(self,
                forecast_horizon: int,
                start_date: Optional[datetime] = None,
                historical_data: Optional[pd.DataFrame] = None,
                recursive: bool = True,
                **kwargs) -> ForecastResult:
        """
        Generate price forecasts.
        
        Args:
            forecast_horizon: Number of periods to forecast
            start_date: Start date for forecast
            historical_data: Historical data for feature creation
            recursive: Whether to use recursive forecasting (step-by-step update of lags)
            
        Returns:
            ForecastResult with forecasts
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if start_date is None:
            start_date = self._training_data['datetime'].max() + pd.Timedelta(minutes=30)
        
        if historical_data is None:
            historical_data = self._training_data
        
        # Generate forecast datetimes
        datetime_index = pd.date_range(
            start=start_date,
            periods=forecast_horizon,
            freq='30min'
        )
        
        # Truncate historical data to before forecast start
        hist_data = historical_data[historical_data['datetime'] < start_date].copy()
        
        if recursive:
            # Recursive forecasting: predict step-by-step and update lags
            point_forecast = self._predict_recursive(hist_data, datetime_index)
        else:
            # Direct forecasting: create features for all forecast periods at once
            # Note: this usually results in constant predictions for long horizons
            # as lag features will be NaN/zeroed.
            forecast_features = self.feature_engineer.create_forecast_features(
                hist_data,
                datetime_index,
                target_col='price',
                datetime_col='datetime',
                extra_lag_cols=self.extra_lag_cols
            )
            
            # Align forecast features with datetime_index
            forecast_features = forecast_features.set_index('datetime').reindex(datetime_index).reset_index()
            forecast_features = forecast_features.rename(columns={'index': 'datetime'})
            
            # Ensure features match training features
            for feat in self._feature_names:
                if feat not in forecast_features.columns:
                    forecast_features[feat] = 0
            
            X_forecast = forecast_features[self._feature_names].values
            X_forecast = np.nan_to_num(X_forecast, nan=0)
            point_forecast = self._model.predict(X_forecast)
        
        # Generate prediction intervals (non-recursive for simplicity)
        # Prediction intervals often use simpler heuristics for long horizons
        if self._quantile_models:
            # For quantile models, we use a single-pass prediction for efficiency
            # though recursive would be more accurate but much slower
            forecast_features_full = self.feature_engineer.create_forecast_features(
                hist_data,
                datetime_index,
                target_col='price',
                datetime_col='datetime',
                extra_lag_cols=self.extra_lag_cols
            )
            X_forecast_full = forecast_features_full.set_index('datetime').reindex(datetime_index).reset_index()
            X_forecast_full = X_forecast_full[self._feature_names].values
            X_forecast_full = np.nan_to_num(X_forecast_full, nan=0)
            
            lower_bound = self._quantile_models.get(0.10).predict(X_forecast_full)
            upper_bound = self._quantile_models.get(0.90).predict(X_forecast_full)
        else:
            # Simple interval estimation
            lower_bound = point_forecast * 0.8
            upper_bound = point_forecast * 1.2
        
        return ForecastResult(
            datetime=datetime_index,
            point_forecast=np.maximum(0, point_forecast),
            lower_bound=np.maximum(0, lower_bound),
            upper_bound=np.maximum(0, upper_bound),
            confidence_level=0.80,
            metadata={
                'model': 'XGBoost',
                'n_features': len(self._feature_names),
                'recursive': recursive
            }
        )

    def _predict_recursive(self, hist_data: pd.DataFrame, datetime_index: pd.DatetimeIndex) -> np.ndarray:
        """
        Helper for recursive multi-step forecasting.
        
        Args:
            hist_data: Historical price and exogenous data
            datetime_index: Future timestamps
            
        Returns:
            Array of point predictions
        """
        print(f"DEBUG: Starting recursive XGBoost forecasting for {len(datetime_index)} steps")
        
        predictions = []
        current_data = hist_data.copy()
        
        # Determine how many historical rows we need to keep for features
        max_lag = max(max(self.feature_engineer.lags) if self.feature_engineer.lags else 2016, 2016)
        
        # Check for future exogenous data in self._training_data or kwargs
        # (Assuming self._training_data might contain the full dataset including future exogenous)
        
        # Temporal features can be pre-calculated for the whole forecast range
        temp_features = self.feature_engineer.create_temporal_features(
            pd.DataFrame({'datetime': datetime_index}), 
            'datetime'
        )
        
        # Identify exogenous columns that should be updated from future data if available
        exog_cols = []
        if self.extra_lag_cols:
            exog_cols.extend([c for c in self.extra_lag_cols if c != 'price'])
        
        for i, dt in enumerate(datetime_index):
            # Only keep necessary history for performance
            if len(current_data) > max_lag + 100:
                current_data = current_data.iloc[-max_lag-50:]
                
            # Create next row
            # For exogenous columns, try to find the value for the current timestamp 'dt'
            # in hist_data (which might contain future values)
            next_row_dict = {'datetime': [dt], 'price': [np.nan]}
            
            for col in current_data.columns:
                if col in ['datetime', 'price']:
                    continue
                
                # Default to last known value
                val = current_data[col].iloc[-1]
                
                # Check if we have a future value for this exogenous column
                # Note: hist_data passed to predict() might contain future exog values
                if col in hist_data.columns:
                    future_vals = hist_data[hist_data['datetime'] == dt][col]
                    if not future_vals.empty:
                        val = future_vals.iloc[0]
                
                next_row_dict[col] = [val]
            
            next_row = pd.DataFrame(next_row_dict)
            
            # Combine history with this next row
            combined = pd.concat([current_data, next_row], ignore_index=True)
            
            # Apply feature generation
            featured = self.feature_engineer.create_lagged_features(combined, 'price')
            featured = self.feature_engineer.create_rolling_features(featured, 'price')
            featured = self.feature_engineer.create_volatility_features(featured, 'price')
            
            # Handle extra lag columns
            if self.extra_lag_cols:
                for col in self.extra_lag_cols:
                    if col in combined.columns and col != 'price':
                        featured = self.feature_engineer.create_lagged_features(featured, col)
                        featured = self.feature_engineer.create_rolling_features(featured, col)
            
            # Merge with pre-calculated temporal features
            row_features = featured.iloc[[-1]].copy()
            for col in temp_features.columns:
                if col != 'datetime':
                    row_features[col] = temp_features.iloc[i][col]
            
            # Ensure all feature columns exist and align with model expectation
            for feat in self._feature_names:
                if feat not in row_features.columns:
                    if feat in combined.columns:
                        row_features[feat] = combined[feat].iloc[-1]
                    else:
                        row_features[feat] = 0
            
            X_point = row_features[self._feature_names].values
            X_point = np.nan_to_num(X_point, nan=0)
            
            # Predict
            pred = self._model.predict(X_point)[0]
            
            # Add some variability if the model is too "conservative" (optional heuristic)
            # but better to let the model learn it from features
            
            predictions.append(pred)
            
            # Update history for next iteration
            next_row['price'] = [pred]
            current_data = pd.concat([current_data, next_row], ignore_index=True)
            
            if (i + 1) % 500 == 0:
                print(f"  Recursive forecast: {i+1}/{len(datetime_index)} steps complete")
                
        return np.array(predictions)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self._feature_importance is None:
            raise ValueError("Model must be fitted first")
        
        return self._feature_importance.head(top_n)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            **self._model_params
        }
    
    def set_params(self, **params) -> 'XGBoostForecaster':
        """Set model parameters."""
        if 'n_estimators' in params:
            self.n_estimators = params['n_estimators']
        if 'max_depth' in params:
            self.max_depth = params['max_depth']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        return self
    
    def calculate_shap_values(self,
                               X: Optional[np.ndarray] = None,
                               n_samples: int = 100) -> Optional[pd.DataFrame]:
        """
        Calculate SHAP values for model interpretability.
        
        Args:
            X: Feature matrix (uses training data if None)
            n_samples: Number of samples for SHAP calculation
            
        Returns:
            DataFrame with SHAP values or None if SHAP not available
        """
        try:
            import shap
        except ImportError:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        if X is None:
            X = self._training_data[self._feature_names].values[:n_samples]
        
        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)
        
        return pd.DataFrame(
            shap_values,
            columns=self._feature_names
        )
