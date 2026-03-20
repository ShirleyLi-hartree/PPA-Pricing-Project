"""
Forecast Command Module

Handles medium-term electricity price forecasting and model evaluation:
1. Load JEPX regional price data
2. Apply enhanced feature engineering
3. Train XGBoost and Prophet models
4. Run rolling window backtest
5. Generate evaluation report
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from src.models.ml_enhanced_v3.prophet_forecaster import ProphetForecaster

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import JapanRegion
from src.forecasting.core import load_jepx_data

logger = logging.getLogger(__name__)


def run_xgboost_backtest(data: pd.DataFrame, 
                        initial_train_days: int = 365 * 3,
                        forecast_days: int = 90,
                        step_days: int = 30) -> pd.DataFrame:
    """
    Run XGBoost backtest with enhanced features.
    
    Args:
        data: DataFrame with datetime and price columns
        initial_train_days: Initial training window in days
        forecast_days: Forecast horizon in days
        step_days: Step size in days
        
    Returns:
        DataFrame with backtest results
    """
    from src.models.ml_enhanced_v3.xgboost_forecaster import XGBoostForecaster
    from src.forecasting.evaluation.backtester import Backtester
    from src.data.preprocessors.feature_engineer import FeatureEngineer
    
    logger.info("Running XGBoost backtest...")
    
    # Initialize components
    feature_engineer = FeatureEngineer(
        lags=[1, 2, 48, 96, 336, 672, 1440, 2016],
        rolling_windows=[24, 48, 168, 720]
    )
    
    # Model parameters
    model_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
    }
    
    # Initialize backtester
    backtester = Backtester(
        model_class=XGBoostForecaster,
        model_params=model_params,
        feature_engineer=feature_engineer,
        initial_train_size=initial_train_days * 48,
        window_size=forecast_days * 48,
        step_size=step_days * 48,
        refit_frequency=3,  # Refit every 3 folds
        use_expanding_window=True
    )
    
    # Run backtest
    results = backtester.run(data)
    
    logger.info(f"XGBoost backtest completed: {len(results)} folds")
    
    return results


def run_prophet_forecast(data: pd.DataFrame,
                        forecast_days: int = 90) -> dict:
    """
    Run Prophet model for comparison.
    
    Args:
        data: DataFrame with datetime and price columns
        forecast_days: Forecast horizon in days
        
    Returns:
        Dictionary with forecast results and metrics
    """
    
    logger.info("Running Prophet forecast...")
    
    # Split data: use last forecast_days as test
    test_periods = forecast_days * 48
    train_data = data.iloc[:-test_periods].copy()
    test_data = data.iloc[-test_periods:].copy()
    
    # Initialize and fit Prophet
    prophet = ProphetForecaster(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    
    prophet.fit(train_data)
    
    # Generate forecast
    forecast = prophet.predict(
        forecast_horizon=len(test_data),
        start_date=test_data['datetime'].iloc[0]
    )
    
    # Calculate metrics
    from src.models.base.metrics import PricingMetrics
    
    actual = test_data['price'].values
    predicted = forecast.point_forecast
    
    metrics = PricingMetrics.calculate_all_metrics(
        actual=actual,
        predicted=predicted,
        lower_bound=forecast.lower_bound,
        upper_bound=forecast.upper_bound
    )
    
    logger.info(f"Prophet MAE: {metrics['mae']:.2f} JPY/kWh")
    
    return {
        'model': 'Prophet',
        'forecast': forecast,
        'metrics': metrics,
        'test_data': test_data
    }


def generate_report(xgb_results: pd.DataFrame, 
                   prophet_results: dict = None,
                   output_dir: Path = None) -> str:
    """
    Generate evaluation report.
    
    Args:
        xgb_results: XGBoost backtest results
        prophet_results: Prophet forecast results (optional)
        output_dir: Directory to save report (optional)
        
    Returns:
        Report as string
    """

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ELECTRICITY PRICE FORECASTING EVALUATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # XGBoost Results
    report_lines.append("## XGBoost Backtest Results")
    report_lines.append("-" * 40)
    report_lines.append(f"Number of folds: {len(xgb_results)}")
    report_lines.append(f"MAE (mean ± std): {xgb_results['mae'].mean():.2f} ± {xgb_results['mae'].std():.2f} JPY/kWh")
    report_lines.append(f"RMSE (mean ± std): {xgb_results['rmse'].mean():.2f} ± {xgb_results['rmse'].std():.2f} JPY/kWh")
    report_lines.append(f"R² (mean): {xgb_results['r2'].mean():.3f}")
    
    if 'mbe' in xgb_results.columns:
        report_lines.append(f"MBE (mean): {xgb_results['mbe'].mean():.2f} JPY/kWh")
    
    report_lines.append("")
    
    # Per-fold details
    report_lines.append("### Per-Fold Results")
    report_lines.append(xgb_results.to_string())
    report_lines.append("")
    
    # Prophet Results (if available)
    if prophet_results:
        report_lines.append("## Prophet Forecast Results")
        report_lines.append("-" * 40)
        metrics = prophet_results['metrics']
        report_lines.append(f"MAE: {metrics['mae']:.2f} JPY/kWh")
        report_lines.append(f"RMSE: {metrics['rmse']:.2f} JPY/kWh")
        report_lines.append(f"sMAPE: {metrics['smape']:.2f}%")
        report_lines.append(f"R²: {metrics['r_squared']:.3f}")
        report_lines.append(f"Coverage: {metrics.get('coverage_probability', 'N/A')}")
        report_lines.append("")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append("-" * 40)
    
    xgb_mae = xgb_results['mae'].mean()
    target_mae = 2.0  # Target: <2.0 JPY/kWh for 3-month forecast
    
    if xgb_mae < target_mae:
        report_lines.append(f"✓ XGBoost MAE ({xgb_mae:.2f}) meets target (<{target_mae} JPY/kWh)")
    else:
        report_lines.append(f"✗ XGBoost MAE ({xgb_mae:.2f}) exceeds target (<{target_mae} JPY/kWh)")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    # Save report if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    
    return report


def run_forecast_command(args):
    """Execute the forecast command."""
    logger.info("=" * 80)
    logger.info("FORECAST COMMAND")
    logger.info("=" * 80)
    logger.info(f"Region: {args.region}")
    logger.info(f"Training years: {args.years}")
    logger.info(f"Forecast horizon: {args.forecast_days} days")
    
    # Paths
    data_dir = PROJECT_ROOT / 'data'
    
    try:
        # Load data
        data = load_jepx_data(JapanRegion.TOKYO, str(data_dir))
        
        # Run XGBoost backtest
        xgb_results = run_xgboost_backtest(
            data,
            initial_train_days=365 * args.years,
            forecast_days=args.forecast_days,
            step_days=args.step_days
        )
        
        # Run Prophet (optional)
        prophet_results = None
        if not args.skip_prophet:
            try:
                prophet_results = run_prophet_forecast(data, args.forecast_days)
            except Exception as e:
                logger.warning(f"Prophet failed: {e}")
        
        # Generate report
        report = generate_report(
            xgb_results,
            prophet_results,
            output_dir=args.output_dir
        )
        
        print(report)
        
        logger.info("=" * 80)
        logger.info("FORECAST COMMAND COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in forecast command: {e}")
        import traceback
        traceback.print_exc()
        return 1