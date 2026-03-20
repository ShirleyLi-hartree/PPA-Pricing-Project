"""
SARIMAX Model for electricity price prediction.

Implements Seasonal ARIMA with exogenous variables for time series forecasting.
Provides a solid statistical baseline with theoretical foundations.
"""
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models.base.model_interface import BasePriceModel, ModelConfig, ForecastResult
from src.utils.date_utils import JapanDateUtils

# Graceful import for pmdarima (optional dependency)
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError, Exception):
    pm = None  # type: ignore
    PMDARIMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SARIMAXModel(BasePriceModel):
    """
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
    model for electricity price forecasting.
    
    Features:
    - Captures trend, seasonality, and autocorrelation
    - Supports exogenous variables
    - Provides theoretical confidence intervals
    - Strong interpretability
    
    Model configuration:
    - SARIMA(p,d,q)(P,D,Q,s) where:
      - p: AR order, d: differencing order, q: MA order
      - P: seasonal AR, D: seasonal differencing, Q: seasonal MA
      - s: seasonal period (48 for daily in 30-min data)
    
    Note: SARIMAX can be computationally expensive for long series.
    For 30-minute data, consider daily aggregation or reduced history.
    """
    
    def __init__(self,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 48),
                 trend: str = 'c',
                 enforce_stationarity: bool = False,
                 enforce_invertibility: bool = False,
                 max_samples: int = 48 * 365,  # 1 year of 30-min data
                 config: Optional[ModelConfig] = None):
        """
        Initialize SARIMAX model.
        
        Args:
            order: (p, d, q) - ARIMA order
            seasonal_order: (P, D, Q, s) - Seasonal order
            trend: Trend parameter ('n', 'c', 't', 'ct')
            enforce_stationarity: Enforce stationarity
            enforce_invertibility: Enforce invertibility
            max_samples: Maximum samples to use (for computational efficiency)
            config: Model configuration
        """
        default_config = ModelConfig(
            name="SARIMAXModel",
            version="1.0.0",
            description="Seasonal ARIMA for electricity price forecasting"
        )
        super().__init__(config or default_config)
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.max_samples = max_samples
        
        self._model = None
        self._fitted_model = None
        self._train_end_date: Optional[datetime] = None
        self._last_values: Optional[np.ndarray] = None
    
    def fit(self,
            train_data: pd.DataFrame,
            datetime_col: str = 'datetime',
            price_col: str = 'price',
            exog: Optional[pd.DataFrame] = None,
            **kwargs) -> 'SARIMAXModel':
        """
        Fit SARIMAX model.
        
        Args:
            train_data: DataFrame with datetime and price
            datetime_col: Name of datetime column
            price_col: Name of price column
            exog: Optional exogenous variables DataFrame
            
        Returns:
            Self for method chaining
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")
        
        # Prepare data
        df = train_data[[datetime_col, price_col]].copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).set_index(datetime_col)
        
        # Limit samples for computational efficiency
        if len(df) > self.max_samples:
            logger.warning(f"Limiting training data from {len(df)} to {self.max_samples} samples")
            df = df.iloc[-self.max_samples:]
        
        y = df[price_col]
        
        logger.info(f"Fitting SARIMAX{self.order}x{self.seasonal_order} on {len(y)} samples")
        
        # Store last values for prediction initialization
        self._last_values = y.values[-max(self.seasonal_order[3], 48):]
        
        # Build and fit model
        try:
            self._model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            
            # Fit with warnings suppressed
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fitted_model = self._model.fit(disp=False, maxiter=100)
            
        except Exception as e:
            logger.warning(f"SARIMAX fitting failed with seasonal order, trying simpler model: {e}")
            # Fall back to simpler model
            self._model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=(0, 0, 0, 0),  # No seasonal component
                trend=self.trend,
            )
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fitted_model = self._model.fit(disp=False, maxiter=100)
        
        self._train_end_date = df.index.max()
        self._training_data = train_data
        self._is_fitted = True
        
        self._model_params = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend,
            'n_samples': len(y),
            'aic': self._fitted_model.aic if self._fitted_model else None,
            'bic': self._fitted_model.bic if self._fitted_model else None,
        }
        
        logger.info(f"SARIMAX fitted. AIC: {self._model_params['aic']:.2f}")
        
        return self
    
    def predict(self,
                forecast_horizon: int,
                start_date: Optional[datetime] = None,
                exog_future: Optional[pd.DataFrame] = None,
                **kwargs) -> ForecastResult:
        """
        Generate predictions using SARIMAX.
        
        Args:
            forecast_horizon: Number of periods to forecast
            start_date: Start date for forecast
            exog_future: Future exogenous variables (if model was fit with exog)
            
        Returns:
            ForecastResult with predictions and intervals
        """
        if not self._is_fitted or self._fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if start_date is None:
            start_date = self._train_end_date + pd.Timedelta(minutes=30)
        
        # Create future dates
        future_dates = pd.date_range(
            start=start_date,
            periods=forecast_horizon,
            freq='30min'
        )
        
        # Get forecast
        try:
            forecast = self._fitted_model.get_forecast(
                steps=forecast_horizon,
                exog=exog_future
            )
            
            point_forecast = forecast.predicted_mean.values
            conf_int = forecast.conf_int(alpha=0.10)  # 90% CI
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values
            
        except Exception as e:
            logger.warning(f"SARIMAX forecast failed, using naive forecast: {e}")
            # Fall back to naive forecast (last value repeated)
            if self._last_values is not None and len(self._last_values) > 0:
                point_forecast = np.full(forecast_horizon, self._last_values[-1])
            else:
                point_forecast = np.zeros(forecast_horizon)
            lower_bound = point_forecast * 0.9
            upper_bound = point_forecast * 1.1
        
        return ForecastResult(
            datetime=pd.DatetimeIndex(future_dates),
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=0.90,
            metadata={
                'model': 'SARIMAX',
                'order': self.order,
                'seasonal_order': self.seasonal_order,
            }
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self._model_params
    
    def set_params(self, **params) -> 'SARIMAXModel':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_diagnostics(self) -> Optional[Dict[str, Any]]:
        """
        Get model diagnostics.
        
        Returns:
            Dictionary with diagnostic information or None if not fitted
        """
        if not self._is_fitted or self._fitted_model is None:
            return None
        
        return {
            'aic': self._fitted_model.aic,
            'bic': self._fitted_model.bic,
            'llf': self._fitted_model.llf,  # Log-likelihood
            'params': dict(self._fitted_model.params),
        }
    
    def summary(self) -> str:
        """Get model summary."""
        if not self._is_fitted or self._fitted_model is None:
            return "Model not fitted."
        
        return str(self._fitted_model.summary())


class SimplifiedSARIMAX(BasePriceModel):
    """
    Simplified SARIMAX for faster training on high-frequency data.
    
    Uses daily aggregation internally for model fitting, then
    distributes predictions back to 30-minute granularity using
    historical intraday patterns.
    """
    
    def __init__(self,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),  # Weekly
                 config: Optional[ModelConfig] = None):
        """
        Initialize simplified SARIMAX.
        
        Args:
            order: ARIMA order for daily data
            seasonal_order: Seasonal order (ys=7 for weekly seasonalit)
            config: Model configuration
        """
        default_config = ModelConfig(
            name="SimplifiedSARIMAX",
            version="1.0.0",
            description="Simplified SARIMAX with daily aggregation"
        )
        super().__init__(config or default_config)
        
        self.order = order
        self.seasonal_order = seasonal_order
        
        self._daily_model = None
        self._fitted_model = None
        self._intraday_pattern: Optional[pd.Series] = None
        self._train_end_date: Optional[datetime] = None
        
        # --- Structure search state ---
        self._structure_fixed: bool = False
        self.selected_exog_cols: List[str] = []
        # Candidate exog cols (extend this list to add future variables)
        self._candidate_exog_cols: List[str] = [
            'is_holiday', 'is_golden_week', 'is_obon', 'is_new_year',
            'is_weekend', 'is_fiscal_year_end',
            'is_summer', 'is_winter',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        ]
        # Business-critical: keep even if p slightly > 0.1 (up to 0.20)
        self._business_critical_vars: List[str] = [
            'is_golden_week', 'is_obon', 'is_new_year', 'is_holiday'
        ]
    
    def _build_daily_exog(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Build daily exogenous variable DataFrame from date index.
        
        All features are deterministic functions of the date (calendar-based),
        so this works for both historical training and future prediction.
        To add new exogenous variables in the future, extend
        ``_candidate_exog_cols`` and add the computation here.
        
        Args:
            dates: DatetimeIndex at daily frequency
            
        Returns:
            DataFrame indexed by dates with candidate exogenous columns
        """
        date_utils = JapanDateUtils()
        records = []
        for dt in dates:
            d = dt.date() if hasattr(dt, 'date') else dt
            month = d.month
            dow = d.weekday()
            records.append({
                'is_holiday': int(date_utils.is_holiday(d)),
                'is_golden_week': int(date_utils.is_golden_week(d)),
                'is_obon': int(date_utils.is_obon(d)),
                'is_new_year': int(date_utils.is_new_year_period(d)),
                'is_weekend': int(dow >= 5),
                'is_fiscal_year_end': int(month == 3),
                'is_summer': int(month in (6, 7, 8)),
                'is_winter': int(month in (12, 1, 2)),
                'month_sin': np.sin(2 * np.pi * (month - 1) / 12),
                'month_cos': np.cos(2 * np.pi * (month - 1) / 12),
                'dow_sin': np.sin(2 * np.pi * dow / 7),
                'dow_cos': np.cos(2 * np.pi * dow / 7),
            })
        exog_df = pd.DataFrame(records, index=dates)
        return exog_df[[c for c in self._candidate_exog_cols if c in exog_df.columns]]
    
    def structure_search(
        self,
        train_data: pd.DataFrame,
        datetime_col: str = 'datetime',
        price_col: str = 'price',
        representative_years: int = 4,
        pvalue_threshold: float = 0.10,
        business_pvalue_threshold: float = 0.20,
        vif_threshold: float = 5.0,
    ) -> 'SimplifiedSARIMAX':
        """
        One-time structural search: find optimal ARIMA order and select
        significant exogenous variables.  Call ONCE before rolling-window
        training; subsequent ``fit()`` calls only re-estimate MLE parameters
        with the frozen structure.
        
        Steps:
          1. auto_arima on representative window -> optimal (p,d,q)(P,D,Q,s)
          2. Fit SARIMAX with all candidate exog -> p-value filtering
          3. VIF check -> remove multicollinear variables
          4. Final refit -> lock structure
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        logger.info("=" * 60)
        logger.info("SARIMAX STRUCTURE SEARCH (one-time)")
        logger.info("=" * 60)
        
        # --- Step 1: Prepare representative window (daily) ---
        df = train_data[[datetime_col, price_col]].copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col)
        
        cutoff = df[datetime_col].max() - pd.DateOffset(years=representative_years)
        df = df[df[datetime_col] >= cutoff]
        
        df['date'] = df[datetime_col].dt.date
        daily_df = df.groupby('date')[price_col].mean()
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.asfreq('D')
        daily_df = daily_df.ffill()
        
        logger.info(f"Representative window: {daily_df.index[0].date()} to "
                     f"{daily_df.index[-1].date()} ({len(daily_df)} days)")
        
        # --- Step 2: auto_arima -> find optimal order ---
        best_order = self.order
        best_seasonal_order = self.seasonal_order
        
        if PMDARIMA_AVAILABLE:
            logger.info("Running auto_arima for order selection (m=7, stepwise)...")
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    auto_model = pm.auto_arima(
                        daily_df.values,
                        start_p=0, max_p=3,
                        start_q=0, max_q=3,
                        start_P=0, max_P=2,
                        start_Q=0, max_Q=2,
                        d=None, D=None,
                        seasonal=True, m=7,
                        stepwise=True,
                        information_criterion='aic',
                        suppress_warnings=True,
                        error_action='ignore',
                        trace=False,
                    )
                best_order = auto_model.order
                best_seasonal_order = auto_model.seasonal_order
                logger.info(f"auto_arima selected: SARIMA{best_order}x{best_seasonal_order} "
                            f"AIC={auto_model.aic():.2f}")
            except Exception as e:
                logger.warning(f"auto_arima failed: {e}. Keeping default order.")
        else:
            logger.warning("pmdarima not installed. Using default order "
                           f"{best_order}x{best_seasonal_order}")
        
        # --- Step 3: Build candidate exog and fit full model ---
        all_exog = self._build_daily_exog(daily_df.index)
        candidate_cols = [c for c in self._candidate_exog_cols if c in all_exog.columns]
        
        if not candidate_cols:
            logger.warning("No candidate exog columns available.")
            self.order = best_order
            self.seasonal_order = best_seasonal_order
            self._structure_fixed = True
            return self
        
        logger.info(f"Fitting SARIMAX with {len(candidate_cols)} candidate exog variables...")
        
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_model = SARIMAX(
                    daily_df, exog=all_exog[candidate_cols],
                    order=best_order, seasonal_order=best_seasonal_order,
                    trend='c', enforce_stationarity=False, enforce_invertibility=False,
                )
                fitted_full = full_model.fit(disp=False, maxiter=200)
        except Exception as e:
            logger.warning(f"Full exog model fitting failed: {e}. Proceeding without exog.")
            self.order = best_order
            self.seasonal_order = best_seasonal_order
            self._structure_fixed = True
            return self
        
        # --- Step 4: P-value filtering ---
        pvalues = fitted_full.pvalues
        kept_cols: List[str] = []
        for col in candidate_cols:
            if col not in pvalues.index:
                continue
            pv = pvalues[col]
            is_biz = col in self._business_critical_vars
            threshold = business_pvalue_threshold if is_biz else pvalue_threshold
            if pv <= threshold:
                kept_cols.append(col)
                tag = " [business-critical]" if is_biz and pv > pvalue_threshold else ""
                logger.info(f"  KEEP  {col:25s}  p={pv:.4f}{tag}")
            else:
                logger.info(f"  DROP  {col:25s}  p={pv:.4f}")
        
        if not kept_cols:
            logger.info("No exog variables significant. Using ARIMA-only model.")
            self.order = best_order
            self.seasonal_order = best_seasonal_order
            self.selected_exog_cols = []
            self._structure_fixed = True
            return self
        
        # --- Step 5: VIF check (multicollinearity) ---
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            exog_kept = all_exog[kept_cols].copy()
            while len(exog_kept.columns) > 1:
                vif_data = pd.Series(
                    [variance_inflation_factor(exog_kept.values, i)
                     for i in range(exog_kept.shape[1])],
                    index=exog_kept.columns
                )
                max_vif = vif_data.max()
                if max_vif < vif_threshold:
                    break
                worst_col = vif_data.idxmax()
                logger.info(f"  VIF remove: {worst_col} (VIF={max_vif:.2f})")
                exog_kept = exog_kept.drop(columns=[worst_col])
                kept_cols = list(exog_kept.columns)
            
            if kept_cols:
                vif_final = pd.Series(
                    [variance_inflation_factor(exog_kept.values, i)
                     for i in range(exog_kept.shape[1])],
                    index=exog_kept.columns
                )
                for col in kept_cols:
                    logger.info(f"  VIF final: {col:25s}  VIF={vif_final[col]:.2f}")
        except Exception as e:
            logger.warning(f"VIF check failed: {e}. Keeping p-value filtered set.")
        
        # --- Step 6: Final refit with selected exog ---
        if kept_cols:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    final_model = SARIMAX(
                        daily_df, exog=all_exog[kept_cols],
                        order=best_order, seasonal_order=best_seasonal_order,
                        trend='c', enforce_stationarity=False, enforce_invertibility=False,
                    )
                    fitted_final = final_model.fit(disp=False, maxiter=200)
                
                logger.info(f"Final model AIC={fitted_final.aic:.2f}, BIC={fitted_final.bic:.2f}")
                for col in kept_cols:
                    if col in fitted_final.pvalues.index:
                        logger.info(f"  coeff {col:25s}  "
                                    f"est={fitted_final.params[col]:.4f}  "
                                    f"p={fitted_final.pvalues[col]:.4f}")
                
                # Ljung-Box residual diagnostics (informational only)
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    resid = fitted_final.resid.dropna()
                    lb = acorr_ljungbox(resid, lags=[7, 14], return_df=True)
                    for lag_idx, row in lb.iterrows():
                        logger.info(f"  Ljung-Box lag={lag_idx}: "
                                    f"stat={row['lb_stat']:.2f}, p={row['lb_pvalue']:.4f}")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Final refit failed: {e}.")
        
        # --- Step 7: Fix structure ---
        self.order = best_order
        self.seasonal_order = best_seasonal_order
        self.selected_exog_cols = kept_cols
        self._structure_fixed = True
        
        logger.info("-" * 60)
        logger.info(f"Structure fixed: SARIMA{self.order}x{self.seasonal_order}")
        logger.info(f"Selected exog ({len(self.selected_exog_cols)}): {self.selected_exog_cols}")
        logger.info("=" * 60)
        
        return self
    
    def fit(self,
            train_data: pd.DataFrame,
            datetime_col: str = 'datetime',
            price_col: str = 'price',
            **kwargs) -> 'SimplifiedSARIMAX':
        """
        Fit simplified SARIMAX model.
        
        Aggregates to daily data for SARIMAX, stores intraday pattern.
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels required")
        
        # Prepare data
        df = train_data[[datetime_col, price_col]].copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col)
        
        # Calculate intraday pattern (average by half-hour period)
        df['period'] = df[datetime_col].dt.hour * 2 + df[datetime_col].dt.minute // 30
        self._intraday_pattern = df.groupby('period')[price_col].mean()
        self._intraday_pattern = self._intraday_pattern / self._intraday_pattern.mean()
        
        # Aggregate to daily
        df['date'] = df[datetime_col].dt.date
        daily_df = df.groupby('date')[price_col].mean()
        
        # Ensure index is DatetimeIndex with frequency 'D' to avoid warnings
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.asfreq('D')
        
        logger.info(f"Fitting SimplifiedSARIMAX on {len(daily_df)} daily samples")
        
        # Build exogenous variables if structure has been fixed
        exog_daily = None
        if self._structure_fixed and self.selected_exog_cols:
            all_exog = self._build_daily_exog(daily_df.index)
            exog_daily = all_exog[self.selected_exog_cols]
        
        # Fit SARIMAX on daily data
        self._daily_model = SARIMAX(
            daily_df,
            exog=exog_daily,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend='c'
        )
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = self._daily_model.fit(disp=False, maxiter=100)
        
        self._train_end_date = df[datetime_col].max()
        self._training_data = train_data
        self._is_fitted = True
        
        self._model_params = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'n_daily_samples': len(daily_df),
        }
        
        return self
    
    def predict(self,
                forecast_horizon: int,
                start_date: Optional[datetime] = None,
                **kwargs) -> ForecastResult:
        """
        Generate 30-minute predictions by combining daily forecast with intraday pattern.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        if start_date is None:
            start_date = self._train_end_date + pd.Timedelta(minutes=30)
        
        # Create future dates
        future_dates = pd.date_range(
            start=start_date,
            periods=forecast_horizon,
            freq='30min'
        )
        
        # Calculate number of days needed
        n_days = (future_dates[-1].date() - future_dates[0].date()).days + 1
        
        # Build future exogenous variables if structure was fixed with exog
        exog_future = None
        if self._structure_fixed and self.selected_exog_cols:
            future_daily_dates = pd.date_range(
                start=future_dates[0].normalize(),
                periods=n_days, freq='D'
            )
            all_future_exog = self._build_daily_exog(future_daily_dates)
            exog_future = all_future_exog[self.selected_exog_cols]
        
        # Get daily forecast
        daily_forecast = self._fitted_model.get_forecast(steps=n_days, exog=exog_future)
        daily_pred = daily_forecast.predicted_mean.values
        daily_ci = daily_forecast.conf_int(alpha=0.10)
        
        # Expand to 30-minute using intraday pattern
        point_forecast = []
        lower_bound = []
        upper_bound = []
        
        for i, dt in enumerate(future_dates):
            day_idx = (dt.date() - future_dates[0].date()).days
            if day_idx >= len(daily_pred):
                day_idx = len(daily_pred) - 1
            
            period = dt.hour * 2 + dt.minute // 30
            pattern = self._intraday_pattern.get(period, 1.0)
            
            point_forecast.append(daily_pred[day_idx] * pattern)
            lower_bound.append(daily_ci.iloc[day_idx, 0] * pattern)
            upper_bound.append(daily_ci.iloc[day_idx, 1] * pattern)
        
        return ForecastResult(
            datetime=pd.DatetimeIndex(future_dates),
            point_forecast=np.array(point_forecast),
            lower_bound=np.array(lower_bound),
            upper_bound=np.array(upper_bound),
            confidence_level=0.90,
            metadata={'model': 'SimplifiedSARIMAX'}
        )
    
    def get_params(self) -> Dict[str, Any]:
        return self._model_params
    
    def set_params(self, **params) -> 'SimplifiedSARIMAX':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
