"""
Unified Validation Matrix - Independent validation framework.

This module handles model validation COMPLETELY SEPARATELY from training.
It receives trained models and computes standardized metrics for fair comparison.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.forecasting.training.rolling_trainer import TrainedModelResult
from src.models.base.metrics import PricingMetrics, PriceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Container for validation results of a single model on a single window.
    
    Attributes:
        model_name: Name of the validated model
        window_id: ID of the validation window
        metrics: Dictionary of metric name -> value
        scenario_metrics: Dictionary of scenario -> metrics
        granularity_metrics: Dictionary of granularity -> metrics
        predictions: Predicted values (optional, for debugging)
        actuals: Actual values (optional, for debugging)
        lower_bound: Lower bound of predictions (optional)
        upper_bound: Upper bound of predictions (optional)
        datetime_index: Datetime index for the validation period (optional)
    """
    model_name: str
    window_id: int
    metrics: Dict[str, float] = field(default_factory=dict)
    scenario_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    granularity_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    datetime_index: Optional[pd.DatetimeIndex] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {
            'model_name': self.model_name,
            'window_id': self.window_id,
            **self.metrics
        }
        
        # Add scenario metrics
        for scenario, metrics in self.scenario_metrics.items():
            for metric_name, value in metrics.items():
                result[f'{scenario}_{metric_name}'] = value
        
        return result


class UnifiedValidationMatrix:
    """
    Unified Validation Matrix - Independent validation framework.
    
    This class is COMPLETELY SEPARATE from training:
    - It receives already-trained models
    - It generates predictions on validation data
    - It computes standardized metrics
    - It enables fair comparison across models
    
    Metrics computed:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - sMAPE: Symmetric Mean Absolute Percentage Error
    - R2: Coefficient of Determination
    - MBE: Mean Bias Error
    
    Granularities:
    - Overall: All data
    - Hourly: By hour of day
    - Daily: By day of week
    - Monthly: By month
    - Quarterly: By quarter
    """
    
    # Standard metrics to compute
    METRICS = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'R2', 'MBE']
    
    # Time granularities for analysis
    GRANULARITIES = ['overall', 'hourly', 'daily', 'monthly', 'quarterly']
    
    def __init__(self,
                 peak_hour_start: int = 8,
                 peak_hour_end: int = 22):
        """
        Initialize validation matrix.
        
        Args:
            peak_hour_start: Start hour for peak period (inclusive)
            peak_hour_end: End hour for peak period (exclusive)
        """
        self.peak_hour_start = peak_hour_start
        self.peak_hour_end = peak_hour_end
        
        # Store all validation results
        self.results: List[ValidationResult] = []
    
    def _compute_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all standard metrics using PricingMetrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric name -> value
        """
        # Ensure arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return {m: np.nan for m in self.METRICS}
        
        if len(y_true) != len(y_pred):
            # Truncate to minimum length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        # Delegate to PricingMetrics
        return {
            'MAE': PricingMetrics.mae(y_true, y_pred),
            'RMSE': PricingMetrics.rmse(y_true, y_pred),
            'MAPE': PricingMetrics.mape(y_true, y_pred),
            'sMAPE': PricingMetrics.smape(y_true, y_pred),
            'R2': PricingMetrics.r_squared(y_true, y_pred),
            'MBE': PricingMetrics.mean_bias_error(y_true, y_pred),
        }
    
    def _get_scenario_mask(self,
                           datetime_index: pd.DatetimeIndex,
                           scenario: str) -> np.ndarray:
        """
        Get boolean mask for a supply scenario.
        
        Args:
            datetime_index: DatetimeIndex of the data
            scenario: Scenario name ('24_7', 'peak_only', 'off_peak')
            
        Returns:
            Boolean mask array
        """
        # Ensure we have a DatetimeIndex
        if not isinstance(datetime_index, pd.DatetimeIndex):
            datetime_index = pd.DatetimeIndex(datetime_index)
        
        hour = datetime_index.hour
        is_weekend = datetime_index.dayofweek >= 5
        
        if scenario == '24_7':
            return np.ones(len(datetime_index), dtype=bool)
        elif scenario == 'peak_only':
            # Peak: 8:00-22:00 (configurable)
            return np.array((hour >= self.peak_hour_start) & (hour < self.peak_hour_end))
        elif scenario == 'off_peak':
            # Off-peak: night hours (22:00-8:00) OR weekends
            is_night = (hour >= self.peak_hour_end) | (hour < self.peak_hour_start)
            return np.array(is_weekend | is_night)
        else:
            logger.warning(f"Unknown scenario: {scenario}, returning all True")
            return np.ones(len(datetime_index), dtype=bool)
    
    def validate_single(self,
                        trained_result: TrainedModelResult,
                        datetime_col: str = 'datetime',
                        price_col: str = 'price',
                        scenarios: List[str] = None) -> ValidationResult:
        """
        Validate a single trained model on its validation window.
        
        This method is completely separate from training - it only:
        1. Receives an already-trained model
        2. Generates predictions on validation data
        3. Computes metrics
        
        Args:
            trained_result: TrainedModelResult from training phase
            datetime_col: Name of datetime column
            price_col: Name of price column
            scenarios: List of scenarios to validate (default: all)
            
        Returns:
            ValidationResult with all metrics
        """
        scenarios = scenarios or ['24_7', 'peak_only', 'off_peak']
        
        model = trained_result.model
        window = trained_result.window
        model_name = trained_result.model_name
        
        # Use test_data for backtesting evaluation (formerly validation_data)
        if window.test_data is None or len(window.test_data) == 0:
            # Fallback for backward compatibility if test_data is not set but validation_data is
            # (though we removed validation_data field, so checking getattr for safety)
            if hasattr(window, 'validation_data') and window.validation_data is not None:
                val_data = window.validation_data
            else:
                raise ValueError(f"Window {window.window_id} has no test data for evaluation")
        else:
            val_data = window.test_data
        
        logger.info(f"Validating {model_name} on window {window.window_id} "
                    f"({len(val_data)} samples)")
        
        # Generate predictions
        try:
            forecast = model.predict(
                forecast_horizon=len(val_data),
                start_date=val_data[datetime_col].iloc[0],
                historical_data=window.train_data
            )
            y_pred = forecast.point_forecast
            lower_bound = forecast.lower_bound
            upper_bound = forecast.upper_bound
        except Exception as e:
            logger.error(f"Error predicting: {str(e)}")
            raise
        
        y_true = val_data[price_col].values
        datetime_index = pd.DatetimeIndex(val_data[datetime_col])
        
        # Compute overall metrics
        overall_metrics = self._compute_metrics(y_true, y_pred)
        
        # Compute scenario metrics
        scenario_metrics = {}
        for scenario in scenarios:
            mask = self._get_scenario_mask(datetime_index, scenario)
            if mask.sum() > 0:
                scenario_metrics[scenario] = self._compute_metrics(
                    y_true[mask], y_pred[mask]
                )
        
        # Compute granularity metrics
        granularity_metrics = self._compute_granularity_metrics(
            y_true, y_pred, datetime_index
        )
        
        result = ValidationResult(
            model_name=model_name,
            window_id=window.window_id,
            metrics=overall_metrics,
            scenario_metrics=scenario_metrics,
            granularity_metrics=granularity_metrics,
            predictions=y_pred,
            actuals=y_true,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            datetime_index=datetime_index
        )
        
        self.results.append(result)
        
        logger.info(f"Validation complete: MAE={overall_metrics['MAE']:.3f}, "
                    f"RMSE={overall_metrics['RMSE']:.3f}")
        
        return result
    
    def _compute_granularity_metrics(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      datetime_index: pd.DatetimeIndex) -> Dict[str, Dict[str, float]]:
        """Compute metrics at different time granularities."""
        # Ensure we have a DatetimeIndex
        if not isinstance(datetime_index, pd.DatetimeIndex):
            datetime_index = pd.DatetimeIndex(datetime_index)
        
        granularity_metrics = {}
        
        # Hourly (by hour of day)
        hourly_mae = {}
        for hour in range(24):
            mask = datetime_index.hour == hour
            if mask.sum() > 0:
                hourly_mae[str(hour)] = self._compute_metrics(
                    y_true[mask], y_pred[mask]
                )['MAE']
        granularity_metrics['hourly'] = hourly_mae
        
        # Daily (by day of week)
        daily_mae = {}
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for dow in range(7):
            mask = datetime_index.dayofweek == dow
            if mask.sum() > 0:
                daily_mae[day_names[dow]] = self._compute_metrics(
                    y_true[mask], y_pred[mask]
                )['MAE']
        granularity_metrics['daily'] = daily_mae
        
        # Monthly
        monthly_mae = {}
        for month in range(1, 13):
            mask = datetime_index.month == month
            if mask.sum() > 0:
                monthly_mae[str(month)] = self._compute_metrics(
                    y_true[mask], y_pred[mask]
                )['MAE']
        granularity_metrics['monthly'] = monthly_mae
        
        # Quarterly
        quarterly_mae = {}
        for quarter in range(1, 5):
            mask = datetime_index.quarter == quarter
            if mask.sum() > 0:
                quarterly_mae[f'Q{quarter}'] = self._compute_metrics(
                    y_true[mask], y_pred[mask]
                )['MAE']
        granularity_metrics['quarterly'] = quarterly_mae
        
        return granularity_metrics
    
    def validate_all(self,
                     trained_results: List[TrainedModelResult],
                     datetime_col: str = 'datetime',
                     price_col: str = 'price',
                     scenarios: List[str] = None) -> List[ValidationResult]:
        """
        Validate all trained models on their respective windows.
        
        Args:
            trained_results: List of TrainedModelResult from training phase
            datetime_col: Name of datetime column
            price_col: Name of price column
            scenarios: List of scenarios to validate
            
        Returns:
            List of ValidationResult
        """
        results = []
        
        for trained_result in trained_results:
            try:
                result = self.validate_single(
                    trained_result,
                    datetime_col=datetime_col,
                    price_col=price_col,
                    scenarios=scenarios
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error validating {trained_result.model_name} "
                             f"on window {trained_result.window.window_id}: {str(e)}")
        
        return results
    
    def validate_multiple_models(self,
                                  all_trained_results: Dict[str, List[TrainedModelResult]],
                                  datetime_col: str = 'datetime',
                                  price_col: str = 'price',
                                  scenarios: List[str] = None) -> Dict[str, List[ValidationResult]]:
        """
        Validate multiple models on all their windows.
        
        Args:
            all_trained_results: Dict mapping model name -> list of TrainedModelResult
            datetime_col: Name of datetime column
            price_col: Name of price column
            scenarios: List of scenarios to validate
            
        Returns:
            Dict mapping model name -> list of ValidationResult
        """
        all_results = {}
        
        for model_name, trained_results in all_trained_results.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Validating model: {model_name}")
            logger.info(f"{'='*50}")
            
            all_results[model_name] = self.validate_all(
                trained_results,
                datetime_col=datetime_col,
                price_col=price_col,
                scenarios=scenarios
            )
        
        return all_results
    
    def generate_comparison_matrix(self,
                                    all_results: Dict[str, List[ValidationResult]] = None) -> pd.DataFrame:
        """
        Generate comparison matrix across all models.
        
        Args:
            all_results: Dict of model results (uses stored results if None)
            
        Returns:
            DataFrame with model comparison
        """
        if all_results is None:
            # Group stored results by model
            all_results = {}
            for result in self.results:
                if result.model_name not in all_results:
                    all_results[result.model_name] = []
                all_results[result.model_name].append(result)
        
        if not all_results:
            return pd.DataFrame()
        
        # Aggregate metrics across windows
        summary_data = []
        
        for model_name, results in all_results.items():
            if not results:
                continue
            
            # Compute mean and std across windows
            metrics_df = pd.DataFrame([r.metrics for r in results])
            
            summary = {'Model': model_name, 'Windows': len(results)}
            
            for metric in self.METRICS:
                if metric in metrics_df.columns:
                    summary[f'{metric}_mean'] = metrics_df[metric].mean()
                    summary[f'{metric}_std'] = metrics_df[metric].std()
            
            # Add scenario metrics (mean across windows)
            for scenario in ['24_7', 'peak_only', 'off_peak']:
                scenario_maes = [
                    r.scenario_metrics.get(scenario, {}).get('MAE', np.nan)
                    for r in results
                ]
                summary[f'{scenario}_MAE'] = np.nanmean(scenario_maes)
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self,
                       all_results: Dict[str, List[ValidationResult]] = None,
                       metric: str = 'MAE') -> str:
        """
        Get the best model based on specified metric.
        
        Args:
            all_results: Dict of model results
            metric: Metric to use for comparison (lower is better for MAE, RMSE, MAPE)
            
        Returns:
            Name of the best model
        """
        comparison = self.generate_comparison_matrix(all_results)
        
        if comparison.empty:
            return None
        
        metric_col = f'{metric}_mean'
        if metric_col not in comparison.columns:
            logger.warning(f"Metric {metric} not found in results")
            return None
        
        # For MAE, RMSE, MAPE, MBE - lower is better
        # For R2 - higher is better
        if metric in ['R2']:
            best_idx = comparison[metric_col].idxmax()
        else:
            best_idx = comparison[metric_col].idxmin()
        
        return comparison.loc[best_idx, 'Model']
    
    def generate_detailed_report(self,
                                  all_results: Dict[str, List[ValidationResult]] = None) -> str:
        """
        Generate detailed validation report as formatted string.
        
        Args:
            all_results: Dict of model results
            
        Returns:
            Formatted report string
        """
        comparison = self.generate_comparison_matrix(all_results)
        
        if comparison.empty:
            return "No validation results available."
        
        lines = []
        lines.append("=" * 80)
        lines.append("UNIFIED VALIDATION MATRIX - MODEL COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall metrics
        lines.append("OVERALL METRICS (Mean ± Std across windows)")
        lines.append("-" * 60)
        lines.append(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'MAPE(%)':<15} {'R2':<10}")
        lines.append("-" * 60)
        
        for _, row in comparison.iterrows():
            mae_str = f"{row['MAE_mean']:.3f}±{row['MAE_std']:.3f}"
            rmse_str = f"{row['RMSE_mean']:.3f}±{row['RMSE_std']:.3f}"
            mape_str = f"{row['MAPE_mean']:.2f}±{row['MAPE_std']:.2f}"
            r2_str = f"{row['R2_mean']:.3f}"
            lines.append(f"{row['Model']:<20} {mae_str:<15} {rmse_str:<15} {mape_str:<15} {r2_str:<10}")
        
        lines.append("")
        lines.append("SCENARIO-SPECIFIC MAE")
        lines.append("-" * 60)
        lines.append(f"{'Model':<20} {'24/7':<15} {'Peak':<15} {'Off-Peak':<15}")
        lines.append("-" * 60)
        
        for _, row in comparison.iterrows():
            lines.append(f"{row['Model']:<20} {row['24_7_MAE']:<15.3f} {row['peak_only_MAE']:<15.3f} {row['off_peak_MAE']:<15.3f}")
        
        lines.append("")
        lines.append("=" * 80)
        
        best_model = self.get_best_model(all_results, 'MAE')
        if best_model:
            lines.append(f"BEST MODEL (by MAE): {best_model}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def calculate_tail_risk(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile: float = 0.95
    ) -> float:
        """
        Calculate tail risk using CVaR (Conditional Value at Risk).
        
        CVaR95 represents the expected value of errors in the worst 5% cases,
        providing a measure of tail risk in forecasting.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            quantile: Quantile threshold (default 0.95 for CVaR95)
            
        Returns:
            CVaR value (lower is better)
        """
        return PriceMetrics.cvar95(y_true, y_pred, quantile=quantile)
    
    def calculate_stability_score(
        self,
        results: List[ValidationResult],
        metric: str = 'MAE'
    ) -> float:
        """
        Calculate stability score based on metric variance across windows.
        
        Stability is measured as the inverse coefficient of variation (CV).
        A higher stability score indicates more consistent performance.
        
        Stability = 1 / (1 + CV) where CV = std / mean
        
        Args:
            results: List of validation results across windows
            metric: Metric to evaluate stability for
            
        Returns:
            Stability score between 0 and 1 (higher is better)
        """
        if not results:
            return 0.0
        
        metric_values = [r.metrics.get(metric, np.nan) for r in results]
        metric_values = [v for v in metric_values if not np.isnan(v)]
        
        if len(metric_values) < 2:
            return 1.0  # Can't compute stability with single value
        
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        
        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0
        
        cv = std_val / mean_val
        stability = 1.0 / (1.0 + cv)
        
        return stability
    
    def calculate_coverage(
        self,
        y_true: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        target_coverage: float = 0.90
    ) -> float:
        """
        Calculate prediction interval coverage rate.
        
        Coverage measures how often actual values fall within the predicted
        confidence intervals.
        
        Args:
            y_true: Actual values
            lower_bound: Lower confidence bounds
            upper_bound: Upper confidence bounds
            target_coverage: Expected coverage rate (default 0.90 for 90% CI)
            
        Returns:
            Actual coverage rate (ideally close to target_coverage)
        """
        y_true = np.asarray(y_true).flatten()
        lower_bound = np.asarray(lower_bound).flatten()
        upper_bound = np.asarray(upper_bound).flatten()
        
        if len(y_true) == 0:
            return 0.0
        
        # Count how many actuals fall within bounds
        within_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = np.mean(within_bounds)
        
        return coverage
    
    def calculate_parameter_stability(
        self,
        param_history: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate parameter stability across rolling windows.
        
        Uses L2 norm of parameter changes between consecutive windows
        to measure how stable model parameters are over time.
        
        Args:
            param_history: List of parameter dictionaries from each window
            
        Returns:
            Parameter stability score (lower changes = higher stability)
        """
        return PriceMetrics.parameter_stability_l2(param_history)
    
    def calculate_composite_score(
        self,
        results: List[ValidationResult],
        predictions_with_bounds: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
        param_history: Optional[List[Dict[str, Any]]] = None,
        tail_risk_weight: float = 0.4,
        stability_weight: float = 0.3,
        coverage_weight: float = 0.2,
        mae_weight: float = 0.1,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate composite score for model evaluation.
        
        CompositeScore = w1×TailRiskScore + w2×StabilityScore + w3×CoverageScore + w4×MAEScore
        
        Where each component is normalized to [0, 1] range if normalize=True,
        and higher composite score indicates better overall performance.
        
        Args:
            results: List of validation results across windows
            predictions_with_bounds: Optional list of (y_true, y_pred, lower, upper) tuples
            param_history: Optional list of parameter dictionaries for stability
            tail_risk_weight: Weight for tail risk component (default 0.4)
            stability_weight: Weight for stability component (default 0.3)
            coverage_weight: Weight for coverage component (default 0.2)
            mae_weight: Weight for MAE component (default 0.1)
            normalize: Whether to normalize components to [0, 1]
            
        Returns:
            Dictionary with composite score and component scores
        """
        # Validate weights sum to 1
        total_weight = tail_risk_weight + stability_weight + coverage_weight + mae_weight
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Composite score weights sum to {total_weight}, not 1.0. Normalizing.")
            tail_risk_weight /= total_weight
            stability_weight /= total_weight
            coverage_weight /= total_weight
            mae_weight /= total_weight
        
        scores = {}
        
        # 1. Calculate MAE score (lower MAE = higher score)
        mae_values = [r.metrics.get('MAE', np.nan) for r in results]
        mae_values = [v for v in mae_values if not np.isnan(v)]
        mean_mae = np.mean(mae_values) if mae_values else np.nan
        scores['mean_mae'] = mean_mae
        
        # 2. Calculate stability score
        stability = self.calculate_stability_score(results, metric='MAE')
        scores['stability'] = stability
        
        # 3. Calculate tail risk (CVaR95)
        cvar_values = []
        coverage_values = []
        
        if predictions_with_bounds:
            for y_true, y_pred, lower, upper in predictions_with_bounds:
                cvar = self.calculate_tail_risk(y_true, y_pred)
                cvar_values.append(cvar)
                
                if lower is not None and upper is not None:
                    cov = self.calculate_coverage(y_true, lower, upper)
                    coverage_values.append(cov)
        else:
            # Calculate from stored predictions/actuals in results
            for result in results:
                if result.actuals is not None and result.predictions is not None:
                    cvar = self.calculate_tail_risk(result.actuals, result.predictions)
                    cvar_values.append(cvar)
        
        mean_cvar = np.mean(cvar_values) if cvar_values else np.nan
        scores['mean_cvar95'] = mean_cvar
        
        # 4. Calculate coverage score
        mean_coverage = np.mean(coverage_values) if coverage_values else np.nan
        scores['mean_coverage'] = mean_coverage
        
        # 5. Calculate parameter stability if history provided
        if param_history and len(param_history) >= 2:
            param_stability = self.calculate_parameter_stability(param_history)
            scores['param_stability'] = param_stability
        else:
            scores['param_stability'] = np.nan
        
        # 6. Compute composite score
        # Convert raw metrics to scores (higher = better)
        if normalize:
            # MAE score: inverse, lower MAE is better
            # Use exponential decay for normalization
            mae_score = np.exp(-mean_mae / 10.0) if not np.isnan(mean_mae) else 0.0
            
            # Tail risk score: inverse, lower CVaR is better
            cvar_score = np.exp(-mean_cvar / 20.0) if not np.isnan(mean_cvar) else 0.0
            
            # Stability score: already in [0, 1], higher is better
            stability_score = stability
            
            # Coverage score: closer to target (0.9) is better
            if not np.isnan(mean_coverage):
                coverage_deviation = abs(mean_coverage - 0.9)
                coverage_score = max(0, 1 - coverage_deviation * 5)  # Penalize deviation
            else:
                coverage_score = 0.5  # Neutral if no coverage data
        else:
            mae_score = -mean_mae if not np.isnan(mean_mae) else 0.0
            cvar_score = -mean_cvar if not np.isnan(mean_cvar) else 0.0
            stability_score = stability
            coverage_score = mean_coverage if not np.isnan(mean_coverage) else 0.5
        
        scores['mae_score'] = mae_score
        scores['cvar_score'] = cvar_score
        scores['stability_score'] = stability_score
        scores['coverage_score'] = coverage_score
        
        # Calculate weighted composite
        composite = (
            tail_risk_weight * cvar_score +
            stability_weight * stability_score +
            coverage_weight * coverage_score +
            mae_weight * mae_score
        )
        
        scores['composite_score'] = composite
        scores['weights'] = {
            'tail_risk': tail_risk_weight,
            'stability': stability_weight,
            'coverage': coverage_weight,
            'mae': mae_weight
        }
        
        return scores
    
    def generate_composite_comparison(
        self,
        all_results: Dict[str, List[ValidationResult]],
        all_predictions_with_bounds: Optional[Dict[str, List[Tuple]]] = None,
        all_param_histories: Optional[Dict[str, List[Dict]]] = None
    ) -> pd.DataFrame:
        """
        Generate comparison matrix with composite scores for all models.
        
        Args:
            all_results: Dict mapping model name -> list of ValidationResult
            all_predictions_with_bounds: Optional dict of predictions with bounds per model
            all_param_histories: Optional dict of parameter histories per model
            
        Returns:
            DataFrame with composite score comparison
        """
        comparison_data = []
        
        for model_name, results in all_results.items():
            if not results:
                continue
            
            # Get predictions and params if available
            preds_bounds = None
            if all_predictions_with_bounds and model_name in all_predictions_with_bounds:
                preds_bounds = all_predictions_with_bounds[model_name]
            
            param_hist = None
            if all_param_histories and model_name in all_param_histories:
                param_hist = all_param_histories[model_name]
            
            # Calculate composite score
            scores = self.calculate_composite_score(
                results=results,
                predictions_with_bounds=preds_bounds,
                param_history=param_hist
            )
            
            comparison_data.append({
                'Model': model_name,
                'Windows': len(results),
                'CompositeScore': scores['composite_score'],
                'MAE_mean': scores['mean_mae'],
                'MAE_score': scores['mae_score'],
                'CVaR95_mean': scores['mean_cvar95'],
                'TailRisk_score': scores['cvar_score'],
                'Stability_score': scores['stability_score'],
                'Coverage_mean': scores['mean_coverage'],
                'Coverage_score': scores['coverage_score'],
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by composite score (descending)
        if not df.empty:
            df = df.sort_values('CompositeScore', ascending=False)
        
        return df
    
    def get_best_model_composite(
        self,
        all_results: Dict[str, List[ValidationResult]],
        all_predictions_with_bounds: Optional[Dict[str, List[Tuple]]] = None,
        all_param_histories: Optional[Dict[str, List[Dict]]] = None
    ) -> Tuple[str, float]:
        """
        Get the best model based on composite score.
        
        Args:
            all_results: Dict of validation results per model
            all_predictions_with_bounds: Optional predictions with bounds
            all_param_histories: Optional parameter histories
            
        Returns:
            Tuple of (best_model_name, composite_score)
        """
        comparison = self.generate_composite_comparison(
            all_results,
            all_predictions_with_bounds,
            all_param_histories
        )
        
        if comparison.empty:
            return None, 0.0
        
        best_row = comparison.iloc[0]
        return best_row['Model'], best_row['CompositeScore']
    
    def generate_monthly_comparison_data(
        self,
        all_validation_results: Dict[str, List[ValidationResult]],
        datetime_col: str = 'datetime',
        price_col: str = 'price'
    ) -> pd.DataFrame:
        """
        Generate monthly aggregated data for visualization comparing actual vs predictions.
        
        This method extracts actual values and predictions from validation results,
        aggregates them by month, and handles overlapping windows by averaging.
        
        Args:
            all_validation_results: Dict mapping model name -> list of ValidationResult
            datetime_col: Name of datetime column
            price_col: Name of price column
            
        Returns:
            DataFrame with monthly aggregated actuals and predictions
        """
        from collections import defaultdict
        
        logger.info("Generating monthly comparison data for visualization from validation results")
        
        # Collect actual values by month
        actuals_by_month = defaultdict(list)
        
        # Collect predictions by month for each model
        predictions_by_month = {
            model_name: defaultdict(lambda: {'point': [], 'lower': [], 'upper': []})
            for model_name in all_validation_results.keys()
        }
        
        # Process each model's validation results
        for model_name, validation_results in all_validation_results.items():
            logger.info(f"Processing {model_name} ({len(validation_results)} windows)")
            
            for result in validation_results:
                if result.datetime_index is None or result.predictions is None:
                    continue
                
                # Get datetime and convert to period
                dt_index = pd.DatetimeIndex(result.datetime_index)
                year_month = dt_index.to_period('M')
                
                # Store actual values (only once per month, not per model)
                if model_name == list(all_validation_results.keys())[0]:
                    df_actual = pd.DataFrame({
                        'year_month': year_month,
                        'actual': result.actuals
                    })
                    for month, group in df_actual.groupby('year_month'):
                        actuals_by_month[month].append(group['actual'].mean())
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'year_month': year_month,
                    'point': result.predictions
                })
                
                # Add bounds if available
                if result.lower_bound is not None:
                    pred_df['lower'] = result.lower_bound
                else:
                    pred_df['lower'] = np.nan
                
                if result.upper_bound is not None:
                    pred_df['upper'] = result.upper_bound
                else:
                    pred_df['upper'] = np.nan
                
                # Aggregate by month
                for month, group in pred_df.groupby('year_month'):
                    predictions_by_month[model_name][month]['point'].append(group['point'].mean())
                    predictions_by_month[model_name][month]['lower'].append(group['lower'].mean())
                    predictions_by_month[model_name][month]['upper'].append(group['upper'].mean())
        
        # Build the final dataframe
        all_months = set(actuals_by_month.keys())
        for model_preds in predictions_by_month.values():
            all_months.update(model_preds.keys())
        
        all_months = sorted(all_months)
        
        if not all_months:
            logger.warning("No monthly data collected from validation results")
            return pd.DataFrame()
        
        # Create result dataframe
        result_data = []
        
        for month in all_months:
            row = {'year_month': month}
            
            # Actual mean (average of overlapping windows)
            if month in actuals_by_month and actuals_by_month[month]:
                row['actual_mean'] = np.mean(actuals_by_month[month])
            else:
                row['actual_mean'] = np.nan
            
            # Model predictions
            for model_name in all_validation_results.keys():
                model_key = model_name.lower()
                month_preds = predictions_by_month[model_name].get(month, {'point': [], 'lower': [], 'upper': []})
                
                if month_preds['point']:
                    row[f'{model_key}_mean'] = np.mean(month_preds['point'])
                else:
                    row[f'{model_key}_mean'] = np.nan
                
                if month_preds['lower'] and not all(np.isnan(month_preds['lower'])):
                    row[f'{model_key}_lower'] = np.nanmean(month_preds['lower'])
                else:
                    row[f'{model_key}_lower'] = np.nan
                
                if month_preds['upper'] and not all(np.isnan(month_preds['upper'])):
                    row[f'{model_key}_upper'] = np.nanmean(month_preds['upper'])
                else:
                    row[f'{model_key}_upper'] = np.nan
            
            result_data.append(row)
        
        result_df = pd.DataFrame(result_data)
        
        logger.info(f"Generated monthly comparison data: {len(result_df)} months, "
                    f"{len(all_validation_results)} models")
        
        return result_df
