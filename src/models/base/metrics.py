"""
Pricing metrics and evaluation functions.

Provides comprehensive metrics for evaluating price forecasting models.
Includes standard metrics (MAE, RMSE, MAPE), interval metrics,
and advanced risk metrics (CVaR, worst week MAE, parameter stability).
"""
from typing import Dict, Optional, Union, List, Any
import numpy as np
import pandas as pd


class PricingMetrics:
    """
    Metrics for evaluating electricity price forecasts.
    
    Includes standard metrics (MAE, RMSE, MAPE) as well as
    electricity-market specific metrics (capture price error).
    """
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray,
             epsilon: float = 1e-10) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE value (as percentage)
        """
        return np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
    
    @staticmethod
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        More robust than MAPE when actual values are near zero.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            sMAPE value (as percentage)
        """
        numerator = np.abs(actual - predicted)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        
        # Handle zero denominator
        mask = denominator > 0
        result = np.zeros_like(actual, dtype=float)
        result[mask] = numerator[mask] / denominator[mask]
        
        return np.mean(result) * 100
    
    @staticmethod
    def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            MSE value
        """
        return np.mean((actual - predicted) ** 2)
    
    @staticmethod
    def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            R-squared value
        """
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def mean_bias_error(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Bias Error.
        
        Positive = model over-predicts on average
        Negative = model under-predicts on average
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            MBE value
        """
        return np.mean(predicted - actual)
    
    @staticmethod
    def max_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Maximum Absolute Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Max absolute error
        """
        return np.max(np.abs(actual - predicted))
    
    @staticmethod
    def capture_price_error(actual_prices: np.ndarray,
                            predicted_prices: np.ndarray,
                            generation_profile: np.ndarray) -> float:
        """
        Calculate capture price prediction error.
        
        Capture price = weighted average price where weights are generation.
        This metric measures how well the model predicts the price
        that a renewable generator would actually receive.
        
        Args:
            actual_prices: Actual spot prices
            predicted_prices: Predicted prices
            generation_profile: Generation profile (0-1 capacity factors)
            
        Returns:
            Absolute error in capture price (JPY/kWh)
        """
        # Calculate actual capture price
        actual_cp = np.sum(actual_prices * generation_profile) / np.sum(generation_profile)
        
        # Calculate predicted capture price
        predicted_cp = np.sum(predicted_prices * generation_profile) / np.sum(generation_profile)
        
        return np.abs(actual_cp - predicted_cp)
    
    @staticmethod
    def capture_price_ratio_error(actual_prices: np.ndarray,
                                   predicted_prices: np.ndarray,
                                   generation_profile: np.ndarray) -> float:
        """
        Calculate capture price ratio prediction error.
        
        Capture ratio = capture price / average price
        
        Args:
            actual_prices: Actual spot prices
            predicted_prices: Predicted prices
            generation_profile: Generation profile
            
        Returns:
            Absolute error in capture price ratio
        """
        # Actual capture ratio
        actual_cp = np.sum(actual_prices * generation_profile) / np.sum(generation_profile)
        actual_avg = np.mean(actual_prices)
        actual_ratio = actual_cp / actual_avg if actual_avg > 0 else 0
        
        # Predicted capture ratio
        predicted_cp = np.sum(predicted_prices * generation_profile) / np.sum(generation_profile)
        predicted_avg = np.mean(predicted_prices)
        predicted_ratio = predicted_cp / predicted_avg if predicted_avg > 0 else 0
        
        return np.abs(actual_ratio - predicted_ratio)
    
    @staticmethod
    def quantile_loss(actual: np.ndarray, 
                      predicted: np.ndarray,
                      quantile: float) -> float:
        """
        Calculate quantile (pinball) loss.
        
        Args:
            actual: Actual values
            predicted: Predicted quantile values
            quantile: Quantile level (e.g., 0.5 for median)
            
        Returns:
            Quantile loss value
        """
        errors = actual - predicted
        
        loss = np.where(
            errors >= 0,
            quantile * errors,
            (quantile - 1) * errors
        )
        
        return np.mean(loss)
    
    @staticmethod
    def coverage_probability(actual: np.ndarray,
                              lower_bound: np.ndarray,
                              upper_bound: np.ndarray) -> float:
        """
        Calculate prediction interval coverage probability.
        
        Measures what fraction of actual values fall within
        the predicted interval.
        
        Args:
            actual: Actual values
            lower_bound: Lower prediction interval bound
            upper_bound: Upper prediction interval bound
            
        Returns:
            Coverage probability (0-1)
        """
        in_interval = (actual >= lower_bound) & (actual <= upper_bound)
        return np.mean(in_interval)
    
    @staticmethod
    def interval_width(lower_bound: np.ndarray,
                       upper_bound: np.ndarray) -> float:
        """
        Calculate average prediction interval width.
        
        Args:
            lower_bound: Lower prediction interval bound
            upper_bound: Upper prediction interval bound
            
        Returns:
            Average interval width
        """
        return np.mean(upper_bound - lower_bound)
    
    @staticmethod
    def winkler_score(actual: np.ndarray,
                      lower_bound: np.ndarray,
                      upper_bound: np.ndarray,
                      alpha: float = 0.10) -> float:
        """
        Calculate Winkler score for prediction intervals.
        
        Combines interval width and coverage penalty.
        Lower is better.
        
        Args:
            actual: Actual values
            lower_bound: Lower bound
            upper_bound: Upper bound
            alpha: Significance level (1 - confidence)
            
        Returns:
            Winkler score
        """
        width = upper_bound - lower_bound
        
        below_lower = actual < lower_bound
        above_upper = actual > upper_bound
        
        penalty = np.zeros_like(actual, dtype=float)
        penalty[below_lower] = (2 / alpha) * (lower_bound[below_lower] - actual[below_lower])
        penalty[above_upper] = (2 / alpha) * (actual[above_upper] - upper_bound[above_upper])
        
        return np.mean(width + penalty)
    
    @staticmethod
    def price_spike_accuracy(actual: np.ndarray,
                              predicted: np.ndarray,
                              spike_threshold_percentile: float = 95) -> Dict[str, float]:
        """
        Calculate accuracy metrics specifically for price spikes.
        
        Price spikes are critical for risk management in electricity markets.
        
        Args:
            actual: Actual prices
            predicted: Predicted prices
            spike_threshold_percentile: Percentile above which prices are considered spikes
            
        Returns:
            Dictionary with precision, recall, and F1 for spike detection
        """
        # Determine spike threshold from actual data
        threshold = np.percentile(actual, spike_threshold_percentile)
        
        # Binary classification: spike or not
        actual_spikes = actual >= threshold
        predicted_spikes = predicted >= threshold
        
        # Calculate metrics
        true_positives = np.sum(actual_spikes & predicted_spikes)
        false_positives = np.sum(~actual_spikes & predicted_spikes)
        false_negatives = np.sum(actual_spikes & ~predicted_spikes)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'spike_precision': precision,
            'spike_recall': recall,
            'spike_f1': f1,
            'spike_threshold': threshold,
        }
    
    @classmethod
    def calculate_all_metrics(cls,
                              actual: np.ndarray,
                              predicted: np.ndarray,
                              generation_profile: Optional[np.ndarray] = None,
                              lower_bound: Optional[np.ndarray] = None,
                              upper_bound: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            generation_profile: Optional generation profile for capture price metrics
            lower_bound: Optional lower prediction interval
            upper_bound: Optional upper prediction interval
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'mae': cls.mae(actual, predicted),
            'rmse': cls.rmse(actual, predicted),
            'mape': cls.mape(actual, predicted),
            'smape': cls.smape(actual, predicted),
            'mse': cls.mse(actual, predicted),
            'r_squared': cls.r_squared(actual, predicted),
            'mean_bias_error': cls.mean_bias_error(actual, predicted),
            'max_absolute_error': cls.max_absolute_error(actual, predicted),
        }
        
        # Add capture price metrics if generation profile provided
        if generation_profile is not None:
            metrics['capture_price_error'] = cls.capture_price_error(
                actual, predicted, generation_profile
            )
            metrics['capture_price_ratio_error'] = cls.capture_price_ratio_error(
                actual, predicted, generation_profile
            )
        
        # Add interval metrics if bounds provided
        if lower_bound is not None and upper_bound is not None:
            metrics['coverage_probability'] = cls.coverage_probability(
                actual, lower_bound, upper_bound
            )
            metrics['interval_width'] = cls.interval_width(lower_bound, upper_bound)
            metrics['winkler_score'] = cls.winkler_score(
                actual, lower_bound, upper_bound
            )
        
        # Add spike metrics
        spike_metrics = cls.price_spike_accuracy(actual, predicted)
        metrics.update(spike_metrics)
        
        return metrics
    
    @staticmethod
    def compare_models(results: List[Dict[str, float]],
                       model_names: List[str],
                       primary_metric: str = 'mae') -> pd.DataFrame:
        """
        Compare multiple models based on metrics.
        
        Args:
            results: List of metric dictionaries from each model
            model_names: Names of models
            primary_metric: Metric to sort by
            
        Returns:
            DataFrame comparing models
        """
        df = pd.DataFrame(results)
        df['model'] = model_names
        df = df.set_index('model')
        
        # Sort by primary metric (lower is better for most metrics)
        df = df.sort_values(primary_metric)
        
        return df
    
    @staticmethod
    def calculate_by_period(actual: np.ndarray,
                            predicted: np.ndarray,
                            datetime_index: pd.DatetimeIndex,
                            period: str = 'hour') -> pd.DataFrame:
        """
        Calculate metrics grouped by time period.
        
        Useful for identifying when models perform well/poorly.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            datetime_index: Datetime index
            period: Grouping period ('hour', 'dayofweek', 'month')
            
        Returns:
            DataFrame with metrics by period
        """
        df = pd.DataFrame({
            'datetime': datetime_index,
            'actual': actual,
            'predicted': predicted,
            'error': np.abs(actual - predicted),
        })
        
        if period == 'hour':
            df['period'] = df['datetime'].dt.hour
        elif period == 'dayofweek':
            df['period'] = df['datetime'].dt.dayofweek
        elif period == 'month':
            df['period'] = df['datetime'].dt.month
        else:
            raise ValueError(f"Unknown period: {period}")
        
        # Group and calculate metrics
        grouped = df.groupby('period').agg({
            'error': ['mean', 'std', 'max'],
            'actual': 'mean',
        })
        
        grouped.columns = ['mae', 'error_std', 'max_error', 'mean_price']
        
        return grouped.reset_index()
    
    @staticmethod
    def forecast_bias_by_horizon(actual: np.ndarray,
                                  predicted: np.ndarray,
                                  periods_per_day: int = 48,
                                  horizons_days: List[int] = [7, 14, 30, 60, 90]) -> pd.DataFrame:
        """
        Calculate forecast accuracy metrics at different prediction horizons.
        
        This is critical for medium-term PPA pricing to understand how
        model accuracy degrades as forecast horizon increases.
        
        Args:
            actual: Actual price values
            predicted: Predicted price values
            periods_per_day: Number of periods per day (48 for 30-min data)
            horizons_days: List of forecast horizons in days to evaluate
            
        Returns:
            DataFrame with metrics (MAE, bias, RMSE) at each horizon
        """
        results = []
        
        for h_days in horizons_days:
            h_periods = h_days * periods_per_day
            
            if len(actual) >= h_periods:
                actual_h = actual[:h_periods]
                predicted_h = predicted[:h_periods]
                
                mae = np.mean(np.abs(actual_h - predicted_h))
                bias = np.mean(predicted_h - actual_h)
                rmse = np.sqrt(np.mean((actual_h - predicted_h) ** 2))
                
                # Also calculate metrics for just that horizon window
                # (i.e., day 7 only, not days 1-7)
                if h_days > horizons_days[0]:
                    prev_h_days = horizons_days[horizons_days.index(h_days) - 1]
                    prev_h_periods = prev_h_days * periods_per_day
                    actual_window = actual[prev_h_periods:h_periods]
                    predicted_window = predicted[prev_h_periods:h_periods]
                    mae_window = np.mean(np.abs(actual_window - predicted_window))
                else:
                    mae_window = mae
                
                results.append({
                    'horizon_days': h_days,
                    'horizon_periods': h_periods,
                    'mae_cumulative': mae,
                    'mae_window': mae_window,
                    'bias': bias,
                    'rmse': rmse,
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def monthly_contract_value_error(actual_prices: np.ndarray,
                                      predicted_prices: np.ndarray,
                                      datetime_index: pd.DatetimeIndex,
                                      contract_volume_mwh: float = 1.0) -> pd.DataFrame:
        """
        Calculate monthly contract value error for PPA pricing assessment.
        
        This metric directly measures the financial impact of prediction errors
        on PPA contract settlements.
        
        Args:
            actual_prices: Actual spot prices (JPY/kWh)
            predicted_prices: Predicted prices (JPY/kWh)
            datetime_index: Datetime index for grouping by month
            contract_volume_mwh: Monthly contract volume in MWh (default 1.0 for per-MWh calculation)
            
        Returns:
            DataFrame with monthly value errors
        """
        df = pd.DataFrame({
            'datetime': datetime_index,
            'actual': actual_prices,
            'predicted': predicted_prices,
        })
        df['month'] = df['datetime'].dt.to_period('M')
        
        monthly_results = []
        
        for month, group in df.groupby('month'):
            actual_avg = group['actual'].mean()
            predicted_avg = group['predicted'].mean()
            
            # Price error in JPY/kWh
            price_error = predicted_avg - actual_avg
            
            # Contract value error = price_error × volume × 1000 (kWh conversion)
            # Positive = overpredicted (buyer overpays with predicted price)
            # Negative = underpredicted (buyer underpays with predicted price)
            value_error_jpy = price_error * contract_volume_mwh * 1000
            
            # Percentage error relative to actual value
            actual_value = actual_avg * contract_volume_mwh * 1000
            pct_error = (price_error / actual_avg * 100) if actual_avg > 0 else 0
            
            monthly_results.append({
                'month': str(month),
                'actual_avg_price': actual_avg,
                'predicted_avg_price': predicted_avg,
                'price_error': price_error,
                'value_error_jpy': value_error_jpy,
                'pct_error': pct_error,
                'n_periods': len(group),
            })
        
        return pd.DataFrame(monthly_results)
    
    @staticmethod
    def calculate_ppa_risk_metrics(actual_prices: np.ndarray,
                                    predicted_prices: np.ndarray,
                                    generation_profile: Optional[np.ndarray] = None,
                                    confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate risk metrics relevant for PPA pricing decisions.
        
        Args:
            actual_prices: Actual spot prices
            predicted_prices: Predicted prices
            generation_profile: Optional generation profile (for capture price)
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            Dictionary with risk metrics
        """
        errors = predicted_prices - actual_prices
        
        # Value at Risk of prediction error
        var_percentile = (1 - confidence_level) * 100
        var_lower = np.percentile(errors, var_percentile)
        var_upper = np.percentile(errors, 100 - var_percentile)
        
        # Conditional VaR (Expected Shortfall)
        cvar_lower = errors[errors <= var_lower].mean() if len(errors[errors <= var_lower]) > 0 else var_lower
        cvar_upper = errors[errors >= var_upper].mean() if len(errors[errors >= var_upper]) > 0 else var_upper
        
        # Maximum underestimation (risk of setting PPA price too low)
        max_underestimate = np.min(errors)  # Most negative error
        
        # Maximum overestimation (risk of setting PPA price too high)
        max_overestimate = np.max(errors)  # Most positive error
        
        metrics = {
            'error_var_lower': var_lower,
            'error_var_upper': var_upper,
            'error_cvar_lower': cvar_lower,
            'error_cvar_upper': cvar_upper,
            'max_underestimate': max_underestimate,
            'max_overestimate': max_overestimate,
            'error_std': np.std(errors),
            'error_skewness': float(pd.Series(errors).skew()),
        }
        
        # Add capture price metrics if generation profile provided
        if generation_profile is not None and len(generation_profile) == len(actual_prices):
            total_gen = np.sum(generation_profile)
            if total_gen > 0:
                actual_capture = np.sum(actual_prices * generation_profile) / total_gen
                predicted_capture = np.sum(predicted_prices * generation_profile) / total_gen
                metrics['capture_price_actual'] = actual_capture
                metrics['capture_price_predicted'] = predicted_capture
                metrics['capture_price_error'] = predicted_capture - actual_capture
        
        return metrics
    
    # ==================== Phase 1/2 Advanced Metrics ====================
    
    @staticmethod
    def cvar95(
        actual: np.ndarray,
        predicted: np.ndarray,
        quantile: float = 0.95
    ) -> float:
        """
        Calculate CVaR95 (Conditional Value at Risk at 95% level).
        
        Also known as Expected Shortfall, this measures the expected absolute
        error in the worst (1-quantile)% of cases. Used as a tail risk indicator.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            quantile: Quantile threshold (default 0.95 for CVaR95)
            
        Returns:
            CVaR value (conditional mean of absolute errors beyond quantile)
        """
        abs_errors = np.abs(actual - predicted)
        var_threshold = np.percentile(abs_errors, quantile * 100)
        tail_errors = abs_errors[abs_errors >= var_threshold]
        
        if len(tail_errors) > 0:
            return float(np.mean(tail_errors))
        return float(var_threshold)
    
    @staticmethod
    def worst_week_mae(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       datetime_index: pd.DatetimeIndex) -> float:
        """
        Calculate the worst (maximum) weekly MAE.
        
        Groups data by ISO week and finds the week with highest MAE.
        Used to identify worst-case model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            datetime_index: DatetimeIndex for week grouping
            
        Returns:
            Maximum weekly MAE across all weeks
        """
        if not isinstance(datetime_index, pd.DatetimeIndex):
            datetime_index = pd.DatetimeIndex(datetime_index)
        
        # Create DataFrame with errors and week info
        df = pd.DataFrame({
            'error': np.abs(y_true - y_pred),
            'year_week': datetime_index.isocalendar().year.astype(str) + '_' + 
                        datetime_index.isocalendar().week.astype(str).str.zfill(2)
        })
        
        # Calculate MAE per week
        weekly_mae = df.groupby('year_week')['error'].mean()
        
        if len(weekly_mae) == 0:
            return float(np.mean(np.abs(y_true - y_pred)))
        
        return float(weekly_mae.max())
    
    @staticmethod
    def worst_month_mae(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        datetime_index: pd.DatetimeIndex) -> float:
        """
        Calculate the worst (maximum) monthly MAE.
        
        Groups data by year-month and finds the month with highest MAE.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            datetime_index: DatetimeIndex for month grouping
            
        Returns:
            Maximum monthly MAE across all months
        """
        if not isinstance(datetime_index, pd.DatetimeIndex):
            datetime_index = pd.DatetimeIndex(datetime_index)
        
        df = pd.DataFrame({
            'error': np.abs(y_true - y_pred),
            'year_month': datetime_index.to_period('M').astype(str)
        })
        
        monthly_mae = df.groupby('year_month')['error'].mean()
        
        if len(monthly_mae) == 0:
            return float(np.mean(np.abs(y_true - y_pred)))
        
        return float(monthly_mae.max())
    
    @staticmethod
    def parameter_stability_l2(params_history: List[Union[np.ndarray, Dict[str, Any]]]) -> float:
        """
        Calculate parameter stability using L2 norm of changes.
        
        Measures how much model parameters change between rolling windows.
        Lower values indicate more stable models.
        
        Args:
            params_history: List of parameter arrays or dicts from consecutive windows
            
        Returns:
            Mean L2 norm of parameter changes between consecutive windows
        """
        if len(params_history) < 2:
            return 0.0
        
        def params_to_array(params):
            """Convert parameters to numpy array for comparison."""
            if isinstance(params, dict):
                # Extract numeric values from dict
                values = []
                for v in params.values():
                    if isinstance(v, (int, float)):
                        values.append(v)
                    elif isinstance(v, (list, tuple)):
                        values.extend([x for x in v if isinstance(x, (int, float))])
                return np.array(values) if values else np.array([0.0])
            return np.asarray(params).flatten()
        
        changes = []
        for i in range(1, len(params_history)):
            prev_params = params_to_array(params_history[i - 1])
            curr_params = params_to_array(params_history[i])
            
            # Ensure same length
            min_len = min(len(prev_params), len(curr_params))
            if min_len > 0:
                change = np.linalg.norm(curr_params[:min_len] - prev_params[:min_len])
                changes.append(change)
        
        if len(changes) == 0:
            return 0.0
        
        return float(np.mean(changes))
    
    @staticmethod
    def feature_importance_drift(importance_history: List[pd.DataFrame],
                                 top_n: int = 10) -> float:
        """
        Calculate feature importance drift for tree-based models.
        
        Measures how much the top-N feature importances change between windows.
        Uses rank correlation to measure drift.
        
        Args:
            importance_history: List of feature importance DataFrames with 'feature' and 'importance' columns
            top_n: Number of top features to consider
            
        Returns:
            Mean rank correlation distance (0 = perfectly stable, 1 = completely different)
        """
        if len(importance_history) < 2:
            return 0.0
        
        correlations = []
        for i in range(1, len(importance_history)):
            prev_df = importance_history[i - 1].nlargest(top_n, 'importance')
            curr_df = importance_history[i].nlargest(top_n, 'importance')
            
            # Get common features
            prev_features = set(prev_df['feature'].tolist())
            curr_features = set(curr_df['feature'].tolist())
            common = prev_features.intersection(curr_features)
            
            if len(common) < 2:
                correlations.append(1.0)  # Completely different
                continue
            
            # Calculate Spearman correlation on common features
            prev_ranks = {f: i for i, f in enumerate(prev_df['feature'].tolist())}
            curr_ranks = {f: i for i, f in enumerate(curr_df['feature'].tolist())}
            
            prev_rank_vals = [prev_ranks[f] for f in common]
            curr_rank_vals = [curr_ranks[f] for f in common]
            
            # Spearman correlation
            from scipy.stats import spearmanr
            corr, _ = spearmanr(prev_rank_vals, curr_rank_vals)
            
            # Convert correlation to distance (0-1 scale)
            if np.isnan(corr):
                correlations.append(0.5)
            else:
                correlations.append((1 - corr) / 2)
        
        return float(np.mean(correlations))
    
    @staticmethod
    def calculate_loss_gs(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          datetime_index: pd.DatetimeIndex,
                          mae_weight: float = 0.5,
                          worst_week_weight: float = 0.3,
                          bias_weight: float = 0.2) -> float:
        """
        Calculate the Grid Search loss function (Loss_GS).
        
        Loss_GS = mae_weight * MAE + worst_week_weight * WorstWeekMAE + bias_weight * |Bias|
        
        This is the objective function for Phase 1 structure search.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            datetime_index: DatetimeIndex for week grouping
            mae_weight: Weight for MAE component (default 0.5)
            worst_week_weight: Weight for worst week MAE (default 0.3)
            bias_weight: Weight for absolute bias (default 0.2)
            
        Returns:
            Combined loss value
        """
        mae = PricingMetrics.mae(y_true, y_pred)
        worst_week = PricingMetrics.worst_week_mae(y_true, y_pred, datetime_index)
        bias = np.abs(PricingMetrics.mean_bias_error(y_true, y_pred))
        
        return mae_weight * mae + worst_week_weight * worst_week + bias_weight * bias


# Alias for backward compatibility and convenience
PriceMetrics = PricingMetrics
