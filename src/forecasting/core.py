"""
Shared forecasting core logic for both pricing and forecast commands.

This module provides reusable components for:
- Data loading and preprocessing
- Model training with cross-validation
- Forecast generation
- Result validation and reporting
- Phase 1 structure search
- Phase 2 rolling backtest
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from config.settings import JapanRegion

logger = logging.getLogger(__name__)


def load_jepx_data(region: JapanRegion, data_dir: str = 'data',
                   years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Shared data loading logic for JEPX historical data using direct pandas approach.
    
    Args:
        region: Target region
        data_dir: Data directory path
        years: List of years to load (default: all available)
        
    Returns:
        DataFrame with historical price data
    """
    from pathlib import Path
    
    logger.info(f"Loading JEPX data for region: {region.value}")
    
    # Find available CSV files
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("spot_summary_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No spot_summary_*.csv files found in {data_dir}")
    
    logger.info(f"Found {len(csv_files)} data files")
    
    # Load and concatenate all files
    dfs = []
    for csv_file in csv_files:
        if years:
            year = csv_file.stem.split('_')[-1]
            if int(year) not in years:
                continue
        logger.info(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No data loaded")
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Create datetime column from date and time_code
    data['datetime'] = pd.to_datetime(data['date']) + pd.to_timedelta(
        (data['time_code'] - 1) * 30, unit='min'
    )
    
    # Select region-specific price column
    price_col = region.value.capitalize() if region.value.capitalize() in data.columns else 'system_price'
    data['price'] = data[price_col]
    
    # Sort by datetime
    data = data.sort_values('datetime').reset_index(drop=True)
    
    logger.info(f"Loaded {len(data)} records from {data['datetime'].min()} to {data['datetime'].max()}")
    
    return data[['datetime', 'price']].copy()


def load_external_forecast(file_path: str) -> pd.DataFrame:
    """
    Load pre-computed forecast from file.
    
    Args:
        file_path: Path to forecast CSV file
        
    Returns:
        DataFrame with forecast data
    """
    logger.info(f"Loading external forecast from: {file_path}")
    
    try:
        forecast_df = pd.read_csv(file_path)
        # Convert datetime column if present
        if 'datetime' in forecast_df.columns:
            forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
        
        logger.info(f"Loaded forecast with {len(forecast_df)} periods")
        return forecast_df
        
    except Exception as e:
        logger.error(f"Failed to load external forecast: {e}")
        raise


def create_standard_models(model_names: List[str]) -> Dict[str, Any]:
    """
    Create standard model instances for forecasting.
    
    Args:
        model_names: List of model names to create
        
    Returns:
        Dictionary mapping model names to instances
    """
    models = {}
    
    # XGBoost
    if 'xgboost' in model_names:
        try:
            from src.models.ml_enhanced_v3.xgboost_forecaster import XGBoostForecaster
            models['XGBoost'] = XGBoostForecaster(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1
            )
        except ImportError as e:
            logger.warning(f"XGBoost not available: {e}")
    
    # Prophet
    if 'prophet' in model_names:
        try:
            from src.models.ml_enhanced_v3.prophet_forecaster import ProphetForecaster
            models['Prophet'] = ProphetForecaster(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
        except ImportError as e:
            logger.warning(f"Prophet not available: {e}")
    
    # SARIMAX
    if 'sarimax' in model_names:
        try:
            from src.models.ml_enhanced_v3.sarimax_model import SimplifiedSARIMAX
            models['SARIMAX'] = SimplifiedSARIMAX(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7)
            )
        except ImportError as e:
            logger.warning(f"SARIMAX not available: {e}")
    
    # Historical Average
    if 'historical' in model_names:
        try:
            from src.models.statistics_v1.historical_average import HistoricalAverageModel
            models['HistoricalAverage'] = HistoricalAverageModel()
        except ImportError as e:
            logger.warning(f"Historical Average not available: {e}")
    
    logger.info(f"Created models: {list(models.keys())}")
    return models


def run_cross_validation_training(
    data: pd.DataFrame,
    models: Dict[str, Any],
    data_years: int = 5,
    step_month: int = 1  # 1-month step for higher resolution
) -> Tuple[Any, Any, str]:
    """
    Run cross-validation training and validation.
    
    Args:
        data: Historical data
        models: Dictionary of model instances
        train_window_years: Training window size in years
        predict_horizon_years: Prediction horizon in years
        step_years: Step size in years
        data_years: Years of data to use (from most recent), default 5 years
        step_month: Rolling window step size in months, default 3 months
        
    Returns:
        Tuple of (trained_results, validation_results, best_model_name)
    """
    from src.forecasting.training.rolling_trainer import RollingWindowTrainer
    from src.forecasting.evaluation.validation_matrix import UnifiedValidationMatrix
    
    logger.info("=" * 80)
    logger.info("CROSS-VALIDATION TRAINING")
    logger.info("=" * 80)
    
    # Filter to use only recent N years of data (consistent with Phase 1)
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values('datetime')
    
    data_end = data['datetime'].max()
    data_start_cutoff = data_end - pd.DateOffset(years=data_years)
    data = data[data['datetime'] >= data_start_cutoff].copy()
    
    logger.info(f"Using last {data_years} years of data: "
                f"{data['datetime'].min().date()} to {data['datetime'].max().date()} "
                f"({len(data)} samples)")
    
    # Initialize trainer with updated configuration
    # 3 years train, 6 months val, 3 months test, 4 weeks step
    trainer = RollingWindowTrainer(
        train_years=3,
        val_months=0,
        test_months=1,
        step_month=step_month,
        region=JapanRegion.TOKYO
    )
    
    # Generate windows
    windows = trainer.generate_windows(data, datetime_col='datetime')
    logger.info(f"Generated {len(windows)} rolling windows (step={step_month} months)")
    
    # One-time structure search for models that support it (e.g. SARIMAX)
    for model_name, model in models.items():
        if hasattr(model, 'structure_search') and not getattr(model, '_structure_fixed', False):
            logger.info(f"Running one-time structure search for {model_name}...")
            try:
                model.structure_search(data, datetime_col='datetime', price_col='price')
                logger.info(f"Structure fixed for {model_name}: order={model.order}, "
                            f"seasonal_order={model.seasonal_order}, "
                            f"exog={getattr(model, 'selected_exog_cols', [])}")
            except Exception as e:
                logger.warning(f"Structure search failed for {model_name}: {e}. Using defaults.")
    
    # Train all models on all windows
    all_trained_results = trainer.train_multiple_models(
        models=models,
        datetime_col='datetime',
        price_col='price'
    )
    
    logger.info("=" * 80)
    logger.info("MODEL VALIDATION")
    logger.info("=" * 80)
    
    # Initialize validation matrix
    validator = UnifiedValidationMatrix()
    
    # Validate all models
    all_validation_results = validator.validate_multiple_models(
        all_trained_results,
        datetime_col='datetime',
        price_col='price',
        scenarios=['24_7', 'peak_only', 'off_peak']
    )
    
    # Generate comparison report
    comparison_df = validator.generate_comparison_matrix(all_validation_results)
    report = validator.generate_detailed_report(all_validation_results)
    
    logger.info("\n" + report)
    
    # Get best model
    best_model_name = validator.get_best_model(all_validation_results, metric='MAE')
    logger.info(f"\nBest model (by MAE): {best_model_name}")
    
    return all_trained_results, all_validation_results, best_model_name


def generate_final_forecast(
    data: pd.DataFrame,
    model: Any,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Generate final forecast for specified period.
    
    Args:
        data: Historical data for training
        model: Trained model instance
        start_date: Forecast start date
        end_date: Forecast end date
        
    Returns:
        DataFrame with forecast results
    """
    logger.info("=" * 80)
    logger.info("FINAL FORECAST GENERATION")
    logger.info("=" * 80)
    
    # Use most recent data for final training
    train_years = 3  # Default 3 years
    cutoff_date = data['datetime'].max() - pd.DateOffset(years=train_years)
    final_train_data = data[data['datetime'] >= cutoff_date].copy()
    
    logger.info(f"Training final model on {len(final_train_data)} samples "
                f"({final_train_data['datetime'].min()} to {final_train_data['datetime'].max()})")
    
    # Fit model on final training data
    import copy
    final_model = copy.deepcopy(model)
    final_model.fit(
        train_data=final_train_data,
        datetime_col='datetime',
        price_col='price'
    )
    
    # Calculate number of periods (30-minute intervals)
    forecast_horizon = int((end_date - start_date).total_seconds() / 1800) + 1
    
    logger.info(f"Generating forecast for {start_date.date()} to {end_date.date()} "
                f"({forecast_horizon} periods)")
    
    # Generate forecast
    forecast_result = final_model.predict(
        forecast_horizon=forecast_horizon,
        start_date=start_date
    )
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'datetime': forecast_result.datetime,
        'forecast': forecast_result.point_forecast,
    })
    
    if forecast_result.lower_bound is not None:
        forecast_df['forecast_lower'] = forecast_result.lower_bound
    if forecast_result.upper_bound is not None:
        forecast_df['forecast_upper'] = forecast_result.upper_bound
    
    logger.info(f"Generated {len(forecast_df)} forecast periods")
    
    return forecast_df


def save_forecast_results(
    forecast_df: pd.DataFrame,
    output_dir: str,
    region: str,
    prefix: str = "forecast"
) -> str:
    """
    Save forecast results to file.
    
    Args:
        forecast_df: Forecast DataFrame
        output_dir: Output directory
        region: Region name
        prefix: File name prefix
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{region}_{timestamp}.csv"
    filepath = output_path / filename
    
    forecast_df.to_csv(filepath, index=False)
    logger.info(f"Saved forecast to {filepath}")
    
    return str(filepath)


def run_phase1_search(
    data: pd.DataFrame,
    model_types: List[str] = None,
    config: Optional[Any] = None,
    output_dir: str = 'config/optimal_params',
    region: JapanRegion = JapanRegion.TOKYO
) -> Dict[str, Dict[str, Any]]:
    """
    Run Phase 1 structure search to find optimal hyperparameters.
    
    Phase 1 uses a 5-year data window (4 years training + 6 months validation)
    to search for optimal model hyperparameters using grid search with
    Loss_GS objective function.
    
    Args:
        data: Historical price data with 'datetime' and 'price' columns
        model_types: List of model types to search ('sarimax', 'prophet', 'xgboost')
        config: Optional Phase1Config instance
        output_dir: Directory to save optimal parameters
        region: Target electricity market region
        
    Returns:
        Dictionary mapping model type -> optimal parameters
    """
    from src.forecasting.config import Phase1Config, GridSearchConfig
    from src.forecasting.grid_search import GridSearchEngine, save_optimal_params
    
    logger.info("=" * 80)
    logger.info("PHASE 1: STRUCTURE SEARCH")
    logger.info("=" * 80)
    
    # Default model types
    if model_types is None:
        model_types = ['sarimax', 'prophet', 'xgboost']
    
    # Use default config if not provided
    if config is None:
        config = Phase1Config()
    
    config.validate()
    
    # Prepare search data window
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values('datetime')
    
    data_end = data['datetime'].max()
    search_start = data_end - pd.DateOffset(years=config.search_years)
    search_data = data[data['datetime'] >= search_start].copy()
    
    logger.info(f"Phase 1 search window: {search_data['datetime'].min().date()} to "
                f"{search_data['datetime'].max().date()} ({len(search_data)} samples)")
    logger.info(f"Model types: {model_types}")
    
    # Get loss type from config
    loss_type = getattr(config, 'loss_type', 'rmse')
    logger.info(f"Loss function: {loss_type.upper()} (Primary)")
    
    # Initialize grid search engine
    gs_config = GridSearchConfig()
    engine = GridSearchEngine(
        phase1_config=config,
        grid_config=gs_config
    )
    
    # Run search for each model type
    all_optimal_params = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Searching optimal parameters for: {model_type.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Split search data into train and validation (4.5yr train + 6mo val)
            train_end = search_data['datetime'].max() - pd.DateOffset(months=6)
            train_data = search_data[search_data['datetime'] <= train_end].copy()
            val_data = search_data[search_data['datetime'] > train_end].copy()
            
            logger.info(f"Phase 1 train/val split: train={len(train_data)} samples, val={len(val_data)} samples")
            
            result = engine.search(
                model_type=model_type,
                train_data=train_data,
                val_data=val_data,
                datetime_col='datetime',
                price_col='price'
            )
            
            optimal_params = result.optimal_params
            all_optimal_params[model_type] = {
                'params': optimal_params,
                'best_loss': result.optimal_loss,
                'n_trials': result.search_metrics.get('total_trials', 0),
                'successful_trials': result.search_metrics.get('successful_trials', 0),
                'search_time_seconds': result.search_time_seconds,
            }
            
            logger.info(f"Best parameters for {model_type}: {optimal_params}")
            logger.info(f"Best Loss_GS: {result.optimal_loss:.4f}")
            
            # Save to YAML
            save_optimal_params(
                result=result,
                output_dir=output_dir
            )
            
        except Exception as e:
            logger.error(f"Phase 1 search failed for {model_type}: {e}")
            all_optimal_params[model_type] = {
                'params': {},
                'error': str(e)
            }
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1 COMPLETE - OPTIMAL PARAMETERS SUMMARY")
    logger.info("=" * 80)
    
    for model_type, result in all_optimal_params.items():
        if 'error' not in result:
            logger.info(f"{model_type.upper()}: {result['params']} (Loss_GS={result['best_loss']:.4f})")
        else:
            logger.info(f"{model_type.upper()}: FAILED - {result['error']}")
    
    return all_optimal_params


def run_phase2_backtest(
    data: pd.DataFrame,
    optimal_params: Dict[str, Dict[str, Any]],
    model_types: List[str] = None,
    config: Optional[Any] = None,
    region: JapanRegion = JapanRegion.TOKYO
) -> Tuple[pd.DataFrame, str, Dict[str, List], Dict[str, List]]:
    """
    Run Phase 2 rolling backtest with optimal parameters from Phase 1.
    
    Phase 2 performs rolling window training and evaluation using the optimal
    parameters found in Phase 1, generating a comprehensive comparison matrix.
    
    Window configurations by model:
    - SARIMAX: 3yr train + 1mo test
    - Prophet: 3yr train + 1mo test
    - XGBoost: 3yr train and validation split(85:15) + 1mo test
    
    Composite Score = 0.4×TailRisk + 0.3×Stability + 0.2×Coverage + 0.1×MAE
    
    Args:
        data: Historical price data with 'datetime' and 'price' columns
        optimal_params: Dictionary of optimal parameters from Phase 1
        model_types: List of model types to evaluate
        config: Optional Phase2Config instance
        region: Target electricity market region
        
    Returns:
        Tuple of (comparison_matrix_df, best_model_name, all_trained_results)
    """
    from src.forecasting.config import Phase2Config
    from src.forecasting.training.rolling_trainer import RollingWindowTrainer
    from src.forecasting.evaluation.validation_matrix import UnifiedValidationMatrix
    
    logger.info("=" * 80)
    logger.info("PHASE 2: ROLLING BACKTEST")
    logger.info("=" * 80)
    
    # Default model types
    if model_types is None:
        model_types = list(optimal_params.keys())
    
    # Use default config if not provided
    if config is None:
        config = Phase2Config()
    
    logger.info(f"Model types: {model_types}")
    logger.info(f"Composite score weights: TailRisk={config.tail_risk_weight}, "
                f"Stability={config.stability_weight}, Coverage={config.coverage_weight}, "
                f"Loss={getattr(config, 'loss_weight', 0.1)}")
    
    # Filter data to use only recent N years (consistent with Phase 1)
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values('datetime')
    data_end = data['datetime'].max()
    data_start_cutoff = data_end - pd.DateOffset(years=config.data_years)
    data = data[data['datetime'] >= data_start_cutoff].copy()
    logger.info(f"Using last {config.data_years} years of data: "
                f"{data['datetime'].min().date()} to {data['datetime'].max().date()} "
                f"({len(data)} samples)")
    
    # Create models with optimal parameters
    models = {}
    
    for model_type in model_types:
        if model_type not in optimal_params:
            logger.warning(f"No optimal params for {model_type}, skipping")
            continue
        
        params = optimal_params[model_type].get('params', {})
        
        if model_type == 'sarimax':
            try:
                from src.models.ml_enhanced_v3.sarimax_model import SimplifiedSARIMAX
                
                # Handle both formats: {'order': (p,d,q)} or {'order_p': p, 'order_d': d, 'order_q': q}
                if 'order' in params and isinstance(params['order'], (list, tuple)):
                    order = tuple(params['order'])
                else:
                    order = (
                        params.get('order_p', 1),
                        params.get('order_d', 1),
                        params.get('order_q', 1)
                    )
                
                if 'seasonal_order' in params and isinstance(params['seasonal_order'], (list, tuple)):
                    seasonal_order = tuple(params['seasonal_order'])
                else:
                    seasonal_order = (
                        params.get('seasonal_P', 1),
                        params.get('seasonal_D', 1),
                        params.get('seasonal_Q', 1),
                        params.get('seasonal_s', 7)
                    )
                
                models['SARIMAX'] = SimplifiedSARIMAX(
                    order=order,
                    seasonal_order=seasonal_order
                )
                # Mark structure as fixed to skip redundant search
                models['SARIMAX']._structure_fixed = True
                logger.info(f"Created SARIMAX with order={order}, seasonal_order={seasonal_order}")
            except ImportError as e:
                logger.warning(f"SARIMAX not available: {e}")
        
        elif model_type == 'prophet':
            try:
                from src.models.ml_enhanced_v3.prophet_forecaster import ProphetForecaster
                models['Prophet'] = ProphetForecaster(
                    changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                    seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                    holidays_prior_scale=params.get('holidays_prior_scale', 10.0)
                )
                models['Prophet']._structure_fixed = True
                logger.info(f"Created Prophet with params: {params}")
            except ImportError as e:
                logger.warning(f"Prophet not available: {e}")
        
        elif model_type == 'xgboost':
            try:
                from src.models.ml_enhanced_v3.xgboost_forecaster import XGBoostForecaster
                model = XGBoostForecaster(
                    n_estimators=params.get('n_estimators', 1000),
                    max_depth=params.get('max_depth', 5),
                    learning_rate=params.get('learning_rate', 0.1)
                )
                
                # Set additional parameters that are not part of the constructor
                if hasattr(model, '_model_params'):
                    model._model_params.update({
                        'subsample': params.get('subsample', 0.8),
                        'colsample_bytree': params.get('colsample_bytree', 0.8),
                        'early_stopping_rounds': params.get('early_stopping_rounds', 50)
                    })
                
                models['XGBoost'] = model
                logger.info(f"Created XGBoost with params: {params}")
            except ImportError as e:
                logger.warning(f"XGBoost not available: {e}")

                
    
    if not models:
        raise ValueError("No models could be created with the provided parameters")
    
    # Train and validate each model with appropriate window configuration
    all_trained_results = {}
    all_validation_results = {}
    all_predictions_with_bounds = {}
    all_param_histories = {}
    
    for model_name, model in models.items():
        model_type_lower = model_name.lower()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")
        
        # Create model-specific trainer
        trainer = RollingWindowTrainer.create_for_model(
            model_type=model_type_lower,
            region=region,
            phase='phase2'
        )
        
        # Generate windows
        windows = trainer.generate_windows(data, datetime_col='datetime')
        logger.info(f"Generated {len(windows)} rolling windows for {model_name}")
        
        # Train on all windows
        trained_results = trainer.train_all_windows(
            model=model,
            datetime_col='datetime',
            price_col='price'
        )
        all_trained_results[model_name] = trained_results
        
        # Collect parameter history for stability calculation
        param_history = []
        for result in trained_results:
            if hasattr(result.model, 'get_params'):
                param_history.append(result.model.get_params())
        all_param_histories[model_name] = param_history
    
    # Unified validation across all models
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING UNIFIED VALIDATION")
    logger.info("=" * 60)
    
    validator = UnifiedValidationMatrix()
    
    for model_name, trained_results in all_trained_results.items():
        logger.info(f"Validating {model_name}...")
        
        validation_results = validator.validate_all(
            trained_results,
            datetime_col='datetime',
            price_col='price',
            scenarios=['24_7', 'peak_only', 'off_peak']
        )
        all_validation_results[model_name] = validation_results
        
        # Collect predictions with bounds for composite scoring
        predictions_list = []
        for result in validation_results:
            if result.actuals is not None and result.predictions is not None:
                # Note: bounds would come from forecast result if available
                predictions_list.append((
                    result.actuals,
                    result.predictions,
                    None,  # lower bound
                    None   # upper bound
                ))
        all_predictions_with_bounds[model_name] = predictions_list
    
    # Generate comparison matrices
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING COMPARISON MATRIX")
    logger.info("=" * 80)
    
    # Standard comparison
    standard_comparison = validator.generate_comparison_matrix(all_validation_results)
    
    # Composite score comparison
    composite_comparison = validator.generate_composite_comparison(
        all_validation_results,
        all_predictions_with_bounds,
        all_param_histories
    )
    
    # Get best model
    best_model, best_score = validator.get_best_model_composite(
        all_validation_results,
        all_predictions_with_bounds,
        all_param_histories
    )
    
    # Print detailed report
    report = validator.generate_detailed_report(all_validation_results)
    logger.info("\n" + report)
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPOSITE SCORE COMPARISON")
    logger.info("=" * 80)
    logger.info("\n" + composite_comparison.to_string())
    
    logger.info("\n" + "=" * 80)
    logger.info(f"BEST MODEL (by Composite Score): {best_model} (Score: {best_score:.4f})")
    logger.info("=" * 80)
    
    return composite_comparison, best_model, all_trained_results, all_validation_results


def run_full_pipeline(
    data: pd.DataFrame,
    model_types: List[str] = None,
    phase1_config: Optional[Any] = None,
    phase2_config: Optional[Any] = None,
    output_dir: str = 'config/optimal_params',
    region: JapanRegion = JapanRegion.TOKYO,
    visualize: bool = True,
    results_dir: str = 'results'
) -> Tuple[Dict[str, Any], pd.DataFrame, str, Optional[str], Dict[str, List]]:
    """
    Run the complete Phase1 -> Phase2 pipeline with optional visualization.
    
    This is the main entry point for running the full structure search and
    rolling backtest workflow.
    
    Args:
        data: Historical price data
        model_types: List of model types to evaluate
        phase1_config: Optional Phase1Config instance
        phase2_config: Optional Phase2Config instance
        output_dir: Directory to save optimal parameters
        region: Target electricity market region
        visualize: Whether to generate comparison chart (default True)
        results_dir: Directory to save visualization output
        
    Returns:
        Tuple of (optimal_params, comparison_matrix, best_model_name, chart_path, all_validation_results)
        chart_path is None if visualize=False
    """
    logger.info("=" * 80)
    logger.info("FULL PIPELINE: PHASE 1 + PHASE 2")
    logger.info("=" * 80)
    
    # Phase 1: Structure Search
    optimal_params = run_phase1_search(
        data=data,
        model_types=model_types,
        config=phase1_config,
        output_dir=output_dir,
        region=region
    )
    
    # Phase 2: Rolling Backtest
    comparison_matrix, best_model, all_trained_results, all_validation_results = run_phase2_backtest(
        data=data,
        optimal_params=optimal_params,
        model_types=model_types,
        config=phase2_config,
        region=region
    )
    
    
    # Visualization
    chart_path = None
    if visualize and all_validation_results:
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING VISUALIZATION")
        logger.info("=" * 80)
        
        try:
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
                monthly_csv_path = Path(results_dir) / f"monthly_comparison_{region.value}_{timestamp}.csv"
                monthly_csv_path.parent.mkdir(parents=True, exist_ok=True)
                monthly_df.to_csv(monthly_csv_path, index=False)
                logger.info(f"Monthly comparison data saved to: {monthly_csv_path}")
                
                # Generate chart
                chart_filename = f"model_comparison_{region.value}_{timestamp}.png"
                chart_path = str(Path(results_dir) / chart_filename)
                
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
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Best model: {best_model}")
    if chart_path:
        logger.info(f"Visualization: {chart_path}")
    logger.info("=" * 80)
    
    return optimal_params, comparison_matrix, best_model, chart_path, all_validation_results