"""
PPA Pricing Command Module

Handles the complete PPA pricing workflow with lazy execution:
1. Check for external forecasts first
2. Load historical JEPX data if needed
3. Train multiple time series models using rolling windows
4. Validate models with unified metrics matrix
5. Calculate PPA prices for different supply scenarios
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import JapanRegion

# Core shared forecasting logic
from src.forecasting.core import (
    load_jepx_data,
    load_external_forecast,
    run_cross_validation_training,
    generate_final_forecast,
    create_standard_models
)

# PPA Pricing modules
from src.core.pricing.config import PPAPricingConfig
from src.core.pricing.ppa_pricing_engine import PPAPricingEngine
from src.core.pricing.sarimax_risk_pricing import SARIMAXRiskPricingEngine
from src.forecasting.evaluation.validation_matrix import UnifiedValidationMatrix

logger = logging.getLogger(__name__)


def _load_latest_strict_sarimax_oos(region: JapanRegion, results_dir: str) -> pd.DataFrame:
    """Load the latest strict monthly SARIMAX OOS history for risk calibration."""
    search_dir = Path(results_dir)
    if not any(search_dir.glob(f"strict_sarimax_monthly_oos_{region.value}_*.csv")):
        search_dir = PROJECT_ROOT / "results"

    files = sorted(search_dir.glob(f"strict_sarimax_monthly_oos_{region.value}_*.csv"))
    if not files:
        return pd.DataFrame()

    return pd.read_csv(files[-1], parse_dates=["month", "as_of_date", "train_start", "train_end_timestamp", "forecast_start", "forecast_end"])


def _load_sarimax_cached_params(param_dir: str) -> dict:
    """Load cached SARIMAX parameters or fall back to the project-wide default optimal file."""
    cached = _try_load_cached_params(["sarimax"], param_dir)
    if cached and "sarimax" in cached:
        return {"sarimax": {"params": cached["sarimax"].get("optimal_params", {})}}

    fallback = PROJECT_ROOT / "results" / "optimal_params" / "sarimax_optimal_20260228.yaml"
    if fallback.exists():
        with open(fallback, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return {"sarimax": {"params": payload.get("optimal_params", {})}}

    return {"sarimax": {"params": {}}}


def _generate_sarimax_risk_forecast(
    data: pd.DataFrame,
    config: PPAPricingConfig,
    output_dir: str,
) -> pd.DataFrame:
    """Generate the SARIMAX forecast used by the probabilistic pricing layer."""
    optimal_params = _load_sarimax_cached_params(output_dir + '/optimal_params')
    sarimax_model = _create_optimal_model("sarimax", optimal_params)
    return generate_price_forecast(
        data=data,
        model=sarimax_model,
        forecast_months=config.forecast_months,
    )


def _try_load_cached_params(model_types: list, param_dir: str) -> dict:
    """
    Try to load cached optimal parameters for given model types.
    
    Args:
        model_types: List of model types to load
        param_dir: Directory containing cached parameters
        
    Returns:
        Dictionary of {model_type: params_data} or None if any model missing
    """
    from pathlib import Path
    from src.forecasting.grid_search import load_optimal_params
    
    param_path = Path(param_dir)
    if not param_path.exists():
        logger.info(f"Parameter cache directory not found: {param_dir}")
        return None
    
    cached_params = {}
    
    for model_type in model_types:
        try:
            params_data = load_optimal_params(
                model_type=model_type,
                input_dir=param_dir
            )
            cached_params[model_type] = params_data
            logger.info(f"Loaded cached params for {model_type}")
        except FileNotFoundError:
            logger.info(f"No cached params found for {model_type}")
            return None  # Need all models, return None if any missing
        except Exception as e:
            logger.warning(f"Failed to load params for {model_type}: {e}")
            return None
    
    return cached_params if cached_params else None


def _create_optimal_model(model_name: str, optimal_params: dict):
    """
    Create a model instance with optimal parameters.
    
    Args:
        model_name: Name of the model (e.g., 'SARIMAX', 'Prophet', 'XGBoost')
        optimal_params: Dictionary containing optimal parameters
        
    Returns:
        Model instance configured with optimal parameters
    """
    model_name_lower = model_name.lower()
    
    if 'sarimax' in model_name_lower:
        from src.models.ml_enhanced_v3.sarimax_model import SimplifiedSARIMAX
        params = optimal_params.get('SARIMAX', optimal_params.get('sarimax', {}))
        order = params.get('params', {}).get('order', (1, 1, 1))
        seasonal_order = params.get('params', {}).get('seasonal_order', (1, 1, 1, 7))
        return SimplifiedSARIMAX(order=order, seasonal_order=seasonal_order)
    
    elif 'prophet' in model_name_lower:
        from src.models.ml_enhanced_v3.prophet_forecaster import ProphetForecaster
        params = optimal_params.get('Prophet', optimal_params.get('prophet', {}))
        prophet_params = params.get('params', {})
        return ProphetForecaster(
            changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.1),
            seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0)
        )
    
    elif 'xgboost' in model_name_lower:
        from src.models.ml_enhanced_v3.xgboost_forecaster import XGBoostForecaster
        params = optimal_params.get('XGBoost', optimal_params.get('xgboost', {}))
        xgb_params = params.get('params', {})
        
        # Create model with basic parameters
        model = XGBoostForecaster(
            n_estimators=100,
            max_depth=xgb_params.get('max_depth', 5),
            learning_rate=xgb_params.get('learning_rate', 0.1)
        )
        
        # Set additional parameters that are not part of the constructor
        if hasattr(model, '_model_params'):
            model._model_params.update({
                'subsample': xgb_params.get('subsample', 0.8),
                'colsample_bytree': xgb_params.get('colsample_bytree', 0.8)
            })
        
        return model
    
    else:
        # Fallback to standard model creation
        from src.forecasting.core import create_standard_models
        models = create_standard_models([model_name])
        return models.get(model_name)


def load_data(region: JapanRegion, data_dir: str = 'data') -> pd.DataFrame:
    """
    Load JEPX historical data using shared logic.
    
    Args:
        region: Target region
        data_dir: Data directory path
        
    Returns:
        DataFrame with historical price data
    """
    return load_jepx_data(region, data_dir)


def run_training_validation(
    data: pd.DataFrame,
    models: dict,
    config: PPAPricingConfig
) -> tuple:
    """
    Run rolling window training and validation using shared logic.
    
    Args:
        data: Historical data
        models: Dictionary of model instances
        config: PPA pricing configuration
        
    Returns:
        Tuple of (trained_results, validation_results, best_model_name)
    """
    return run_cross_validation_training(
        data=data,
        models=models
    )


def generate_price_forecast(
    data: pd.DataFrame,
    model: object,
    forecast_months: int = 3
) -> pd.DataFrame:
    """
    Generate price forecast from latest data.
    
    Args:
        data: Historical data (for training final model)
        model: Best model instance (or fresh instance to retrain)
        forecast_months: Number of months to forecast from latest data (default: 3)
        
    Returns:
        DataFrame with forecast
    """
    # Forecast from latest data for N months
    data_end = data['datetime'].max()
    forecast_start = data_end + pd.Timedelta(minutes=30)
    forecast_end = forecast_start + pd.DateOffset(months=forecast_months) - pd.Timedelta(minutes=30)
    
    return generate_final_forecast(
        data=data,
        model=model,
        start_date=forecast_start,
        end_date=forecast_end
    )


def calculate_ppa_prices(
    forecast_df: pd.DataFrame,
    config: PPAPricingConfig
) -> dict:
    """
    Calculate PPA prices for all scenarios.
    
    Args:
        forecast_df: Forecast DataFrame
        config: PPA pricing configuration
        
    Returns:
        Dictionary of pricing results
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: PPA PRICING")
    logger.info("=" * 80)
    
    # Initialize pricing engine
    pricing_engine = PPAPricingEngine(config=config)
    
    # Calculate prices for all scenarios
    pricing_results = pricing_engine.calculate_all_scenarios(
        forecast=forecast_df,
        datetime_col='datetime',
        price_col='forecast',
        lower_col='forecast_lower' if 'forecast_lower' in forecast_df.columns else None,
        upper_col='forecast_upper' if 'forecast_upper' in forecast_df.columns else None
    )
    
    # Print pricing report
    report = pricing_engine.generate_pricing_report_text(pricing_results)
    logger.info("\n" + report)
    
    return pricing_results


def save_results(
    output_dir: str,
    config: PPAPricingConfig,
    comparison_df: pd.DataFrame = None,
    pricing_results: dict = None,
    forecast_df: pd.DataFrame = None,
    sarimax_risk_result=None,
):
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    region = config.region.value
    
    if comparison_df is not None and not comparison_df.empty:
        comparison_path = output_path / f"model_comparison_{region}_{timestamp}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved model comparison to {comparison_path}")
    
    if pricing_results:
        pricing_engine = PPAPricingEngine(config=config)
        pricing_df = pricing_engine.generate_pricing_report(pricing_results)
        pricing_path = output_path / f"ppa_pricing_{region}_{timestamp}.csv"
        pricing_df.to_csv(pricing_path, index=False)
        logger.info(f"Saved pricing results to {pricing_path}")
    
    if forecast_df is not None:
        forecast_path = output_path / f"price_forecast_{region}_{timestamp}.csv"
        forecast_df.to_csv(forecast_path, index=False)
        logger.info(f"Saved forecast to {forecast_path}")

    if sarimax_risk_result is not None:
        monthly_risk_path = output_path / f"sarimax_quote_risk_{region}_{timestamp}.csv"
        contract_risk_path = output_path / f"sarimax_contract_risk_{region}_{timestamp}.csv"
        report_path = output_path / f"sarimax_risk_report_{region}_{timestamp}.txt"
        band_chart_path = output_path / f"sarimax_quote_bands_{region}_{timestamp}.png"
        pnl_chart_path = output_path / f"sarimax_pnl_distribution_{region}_{timestamp}.png"

        sarimax_risk_result.monthly_quotes.to_csv(monthly_risk_path, index=False)
        sarimax_risk_result.contract_summary.to_csv(contract_risk_path, index=False)
        SARIMAXRiskPricingEngine.write_text_report(
            sarimax_risk_result,
            report_path,
            region=region,
        )
        SARIMAXRiskPricingEngine.plot_monthly_quote_bands(
            sarimax_risk_result,
            band_chart_path,
            title=f"SARIMAX Quote Bands - {config.region.value.capitalize()}",
        )
        SARIMAXRiskPricingEngine.plot_contract_pnl_distribution(
            sarimax_risk_result,
            pnl_chart_path,
            title=f"SARIMAX Historical Error / PnL Distribution - {config.region.value.capitalize()}",
        )
        logger.info(f"Saved SARIMAX quote risk table to {monthly_risk_path}")
        logger.info(f"Saved SARIMAX contract risk summary to {contract_risk_path}")
        logger.info(f"Saved SARIMAX risk report to {report_path}")
        logger.info(f"Saved SARIMAX quote band chart to {band_chart_path}")
        logger.info(f"Saved SARIMAX PnL distribution chart to {pnl_chart_path}")


def run_pricing_command(args):
    """Execute the pricing command with integrated optimization pipeline."""
    logger.info("=" * 80)
    logger.info("PPA PRICING COMMAND")
    logger.info("=" * 80)
    logger.info(f"Region: {args.region}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Forecast horizon: {args.forecast_months} months from latest data")
    logger.info(f"Volume: {args.volume} MWh")
    
    # Check for external forecast first (lazy execution)
    if args.external_forecast_file:
        logger.info(f"Using external forecast file: {args.external_forecast_file}")
        try:
            forecast_df = load_external_forecast(args.external_forecast_file)
            logger.info("Successfully loaded external forecast")
        except Exception as e:
            logger.error(f"Failed to load external forecast: {e}")
            logger.info("Falling back to internal training...")
            forecast_df = None
    else:
        forecast_df = None
    
    # Create configuration
    region = JapanRegion(args.region)
    config = PPAPricingConfig(
        region=region,
        forecast_months=args.forecast_months,
        total_volume_mwh=args.volume,
        train_window_years=args.train_years,
    )
    data = None
    
    # Only load data and train models if no external forecast
    if forecast_df is None:
        # Load data
        data = load_data(region)
        
        best_model = None
        best_model_name = None
        comparison_df = pd.DataFrame()
        all_validation_results = None
        
        if not args.skip_validation:
            # Try to load cached optimal parameters first (unless force retrain)
            cached_params = None
            if not args.force_retrain:
                cached_params = _try_load_cached_params(
                    model_types=args.models,
                    param_dir=args.output_dir + '/optimal_params'
                )
            else:
                logger.info("Force retrain flag set, skipping cached parameters")
            
            if cached_params:
                # Reuse cached parameters (skip expensive grid search)
                logger.info("=" * 80)
                logger.info("REUSING CACHED OPTIMAL PARAMETERS")
                logger.info("=" * 80)
                logger.info("Found existing optimal parameters from previous run")
                logger.info("Skipping Phase 1 grid search (saving ~2 hours)...")
                
                for model_type, params_data in cached_params.items():
                    metadata = params_data.get('metadata', {})
                    logger.info(f"  {model_type.upper()}: Loss={metadata.get('loss_gs', 'N/A'):.3f}, "
                              f"SearchDate={metadata.get('search_date', 'N/A')}")
                
                # Run Phase 2 only with cached parameters
                try:
                    from src.forecasting.core import run_phase2_backtest
                    
                    logger.info("\nRunning Phase 2: Rolling backtest with cached parameters...")
                    
                    # Convert cached format to phase2 format
                    optimal_params_dict = {
                        k: {'params': v.get('optimal_params', {})} 
                        for k, v in cached_params.items()
                    }
                    
                    comparison_df, best_model_name, all_trained_results, all_validation_results = run_phase2_backtest(
                        data=data,
                        optimal_params=optimal_params_dict,
                        model_types=args.models,
                        region=region
                    )
                    
                    logger.info(f"Phase 2 complete. Best model: {best_model_name}")
                    
                    # Create best model with cached parameters
                    best_model = _create_optimal_model(
                        model_name=best_model_name,
                        optimal_params=optimal_params_dict
                    )
                    
                except Exception as e:
                    logger.error(f"Phase 2 with cached params failed: {e}")
                    logger.warning("Falling back to full optimization...")
                    import traceback
                    traceback.print_exc()
                    cached_params = None  # Force full optimization
            
            # Generate visualization if enabled and we have validation results (from cached params)
            if getattr(args, 'visualize', True) and all_validation_results and cached_params:
                try:
                    logger.info("\n" + "=" * 80)
                    logger.info("GENERATING VISUALIZATION FROM CACHED RESULTS")
                    logger.info("=" * 80)
                    
                    from src.forecasting.evaluation.validation_matrix import UnifiedValidationMatrix
                    from src.visualization.model_comparison import plot_monthly_comparison
                    
                    # Generate monthly comparison data
                    validator = UnifiedValidationMatrix()
                    monthly_df = validator.generate_monthly_comparison_data(
                        all_validation_results,
                        datetime_col='datetime',
                        price_col='price'
                    )
                    
                    if not monthly_df.empty:
                        # Save monthly data to CSV
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        monthly_csv_path = Path(args.output_dir) / f"monthly_comparison_{region.value}_{timestamp}.csv"
                        monthly_csv_path.parent.mkdir(parents=True, exist_ok=True)
                        monthly_df.to_csv(monthly_csv_path, index=False)
                        logger.info(f"Monthly comparison data saved to: {monthly_csv_path}")
                        
                        # Generate chart
                        chart_filename = f"model_comparison_{region.value}_{timestamp}.png"
                        chart_path = str(Path(args.output_dir) / chart_filename)
                        
                        plot_monthly_comparison(
                            monthly_df=monthly_df,
                            save_path=chart_path,
                            title=f"Historical vs Model Predictions - {region.value.capitalize()} (Monthly Average)"
                        )
                        logger.info(f"Comparison chart saved to: {chart_path}")
                    else:
                        logger.warning("No monthly data available for visualization")
                        
                except Exception as e:
                    logger.error(f"Visualization generation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            if not cached_params:
                # No cache available: run full optimization (Phase 1 + Phase 2)
                try:
                    from src.forecasting.core import run_full_pipeline
                    
                    logger.info("=" * 80)
                    logger.info("RUNNING FULL OPTIMIZATION PIPELINE")
                    logger.info("=" * 80)
                    logger.info("No cached parameters found, running complete optimization...")
                    logger.info("Phase 1: Grid search for optimal hyperparameters (may take 1-2 hours)")
                    logger.info("Phase 2: Rolling backtest with composite scoring")
                    
                    # Run full pipeline
                    optimal_params_result, comparison_df, best_model_name, chart_path, all_validation_results = run_full_pipeline(
                        data=data,
                        model_types=args.models,
                        output_dir=args.output_dir + '/optimal_params',
                        region=region,
                        visualize=getattr(args, 'visualize', True),
                        results_dir=args.output_dir
                    )
                    
                    logger.info(f"Optimization complete. Best model: {best_model_name}")
                    logger.info("Parameters saved for future reuse")
                    
                    # Create best model instance
                    best_model = _create_optimal_model(
                        model_name=best_model_name,
                        optimal_params=optimal_params_result
                    )
                    
                except Exception as e:
                    logger.error(f"Full optimization failed: {e}")
                    logger.warning("Falling back to simple training...")
                    import traceback
                    traceback.print_exc()
                    
                    # Fallback: simple training
                    models = create_standard_models(args.models)
                    trained_results, validation_results, best_model_name = run_training_validation(
                        data, models, config
                    )
                    validator = UnifiedValidationMatrix()
                    comparison_df = validator.generate_comparison_matrix(validation_results)
                    best_model = models.get(best_model_name, list(models.values())[0])
                    # Ensure all_validation_results is defined for downstream viz
                    all_validation_results = validation_results
        else:
            # Skip validation: use first model with defaults
            logger.info("Skipping validation, using first model with defaults")
            models = create_standard_models(args.models)
            best_model_name = list(models.keys())[0]
            best_model = models[best_model_name]
        
        # Generate forecast
        try:
            forecast_df = generate_price_forecast(
                data,
                best_model,
                forecast_months=config.forecast_months
            )
            logger.info(f"Forecast generated from {forecast_df['datetime'].min().date()} "
                       f"to {forecast_df['datetime'].max().date()}")
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            # Create simple forecast based on historical average
            data_end = data['datetime'].max()
            forecast_start = data_end + pd.Timedelta(minutes=30)
            forecast_periods = int(config.forecast_months * 30 * 48)  # Approx periods
            
            forecast_dates = pd.date_range(
                start=forecast_start,
                periods=forecast_periods,
                freq='30min'
            )
            forecast_df = pd.DataFrame({
                'datetime': forecast_dates,
                'forecast': data['price'].mean()
            })

    if data is None:
        data = load_data(region)

    # Calculate PPA prices
    pricing_results = calculate_ppa_prices(forecast_df, config)

    sarimax_risk_result = None
    try:
        strict_oos_df = _load_latest_strict_sarimax_oos(region, args.output_dir)
        if strict_oos_df.empty:
            logger.warning("No strict SARIMAX OOS history found. Skipping SARIMAX risk pricing outputs.")
        else:
            sarimax_forecast_df = _generate_sarimax_risk_forecast(
                data=data,
                config=config,
                output_dir=args.output_dir,
            )
            risk_engine = SARIMAXRiskPricingEngine(bootstrap_samples=10000, random_seed=42)
            sarimax_risk_result = risk_engine.build(
                forecast_df=sarimax_forecast_df,
                strict_oos_df=strict_oos_df,
            )
            contract = sarimax_risk_result.contract_summary.iloc[0]
            logger.info("=" * 80)
            logger.info("SARIMAX PROBABILISTIC PRICING")
            logger.info("=" * 80)
            logger.info(
                f"Point quote: {contract['quote_price_p50_jpy_kwh']:.3f} JPY/kWh | "
                f"Actual P5/P50/P95: {contract['simulated_actual_p5_jpy_kwh']:.3f} / "
                f"{contract['simulated_actual_p50_jpy_kwh']:.3f} / "
                f"{contract['simulated_actual_p95_jpy_kwh']:.3f}"
            )
            logger.info(
                f"Expected PnL: {contract['expected_pnl_jpy_kwh']:.3f} | "
                f"Prob(loss): {contract['probability_of_loss']:.1%} | "
                f"VaR95 loss: {contract['var_95_loss_jpy_kwh']:.3f} | "
                f"ES95 loss: {contract['es_95_loss_jpy_kwh']:.3f}"
            )
            logger.info(
                f"Quote ladder | Risk-neutral: {contract['risk_neutral_quote_jpy_kwh']:.3f} | "
                f"50% no-loss: {contract['quote_50pct_no_loss_jpy_kwh']:.3f} | "
                f"75% no-loss: {contract['quote_75pct_no_loss_jpy_kwh']:.3f} | "
                f"90% no-loss: {contract['quote_90pct_no_loss_jpy_kwh']:.3f}"
            )
    except Exception as e:
        logger.warning(f"SARIMAX risk pricing generation failed: {e}")

    # Save results
    save_results(
        args.output_dir,
        config,
        comparison_df=comparison_df if 'comparison_df' in locals() else None,
        pricing_results=pricing_results,
        forecast_df=forecast_df,
        sarimax_risk_result=sarimax_risk_result,
    )
    
    logger.info("=" * 80)
    logger.info("PRICING COMMAND COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return 0
