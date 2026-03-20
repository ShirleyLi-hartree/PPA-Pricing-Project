"""
Analysis Command Module.

Handles:
1. Fundamental analysis using standardized LiveSheet inputs
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import yaml

from config.settings import JapanRegion
from src.data.loaders.live_sheet_loader import LiveSheetExtractor
from src.data.loaders.live_sheet_loader import StandardizedFundamentalData
from src.models.fundamental_v2.calibration import FundamentalPriceCalibrator
from src.models.fundamental_v2.monthly_fundamental import MonthlyFundamentalEngine
from src.visualization.fundamental_analysis import (
    plot_calibration_bridge,
    plot_fundamental_comparison,
    plot_fundamental_drivers,
    plot_oos_model_vs_fundamental,
    plot_strict_sarimax_forward_actual,
)
from src.forecasting.core import load_jepx_data

logger = logging.getLogger(__name__)


def _format_fundamental_report(
    region: JapanRegion,
    monthly_prices: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
    chart_paths: Optional[dict] = None,
    pricing_source: Optional[dict] = None,
    calibration_metrics: Optional[dict] = None,
    oos_metrics: Optional[dict] = None,
) -> str:
    """Generate a concise text report for the fundamental run."""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("FUNDAMENTAL ANALYSIS REPORT")
    report_lines.append(f"Region: {region.value}")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    if monthly_prices.empty:
        report_lines.append("No monthly prices were generated.")
        return "\n".join(report_lines)

    latest_window = monthly_prices.head(6).copy()
    report_lines.append("## Monthly Clearing Prices (first 6 months)")
    report_lines.append("-" * 40)
    for _, row in latest_window.iterrows():
        report_lines.append(
            f"{row['month'].strftime('%Y-%m')}: {row['clearing_price_jpy_kwh']:.3f} JPY/kWh "
            f"(Demand {row['demand_mw']:.0f} MW, Marginal {row['marginal_technology']})"
        )

    report_lines.append("")
    report_lines.append("## Summary Statistics")
    report_lines.append("-" * 40)
    if "calibrated_price_jpy_kwh" in monthly_prices.columns:
        report_lines.append(f"Average calibrated price: {monthly_prices['calibrated_price_jpy_kwh'].mean():.3f} JPY/kWh")
        report_lines.append(f"Min calibrated price: {monthly_prices['calibrated_price_jpy_kwh'].min():.3f} JPY/kWh")
        report_lines.append(f"Max calibrated price: {monthly_prices['calibrated_price_jpy_kwh'].max():.3f} JPY/kWh")
        report_lines.append(f"Average raw clearing price: {monthly_prices['clearing_price_jpy_kwh'].mean():.3f} JPY/kWh")
    else:
        report_lines.append(f"Average clearing price: {monthly_prices['clearing_price_jpy_kwh'].mean():.3f} JPY/kWh")
        report_lines.append(f"Min clearing price: {monthly_prices['clearing_price_jpy_kwh'].min():.3f} JPY/kWh")
        report_lines.append(f"Max clearing price: {monthly_prices['clearing_price_jpy_kwh'].max():.3f} JPY/kWh")
    report_lines.append(f"Average demand: {monthly_prices['demand_mw'].mean():.0f} MW")
    report_lines.append("")

    if calibration_metrics:
        report_lines.append("## Calibration Quality")
        report_lines.append("-" * 40)
        report_lines.append(f"Training months: {int(calibration_metrics['training_months'])}")
        report_lines.append(f"Raw MAE: {calibration_metrics['raw_mae']:.3f} JPY/kWh")
        report_lines.append(f"Calibrated MAE: {calibration_metrics['calibrated_mae']:.3f} JPY/kWh")
        report_lines.append(f"Raw RMSE: {calibration_metrics['raw_rmse']:.3f} JPY/kWh")
        report_lines.append(f"Calibrated RMSE: {calibration_metrics['calibrated_rmse']:.3f} JPY/kWh")
        report_lines.append("")

    if comparison_df is not None and not comparison_df.empty:
        overlap = comparison_df.dropna(subset=['fundamental_price_jpy_kwh', 'pricing_price_jpy_kwh'])
        report_lines.append("## Pricing Comparison")
        report_lines.append("-" * 40)
        if pricing_source:
            report_lines.append(
                f"Pricing source: {pricing_source.get('model', 'Unknown')} "
                f"from {pricing_source.get('forecast_file', 'N/A')}"
            )
        if overlap.empty:
            report_lines.append("No overlapping months found with pricing forecast outputs.")
        else:
            mean_diff = (overlap["pricing_price_jpy_kwh"] - overlap["fundamental_price_jpy_kwh"]).mean()
            report_lines.append(f"Overlapping months: {len(overlap)}")
            report_lines.append(f"Average pricing - fundamental spread: {mean_diff:.3f} JPY/kWh")
            for _, row in overlap.head(6).iterrows():
                report_lines.append(
                    f"{row['month'].strftime('%Y-%m')}: fundamental={row['fundamental_price_jpy_kwh']:.3f}, "
                    f"pricing={row['pricing_price_jpy_kwh']:.3f}"
                )
        report_lines.append("")

    if oos_metrics:
        report_lines.append("## OOS Comparison vs Actual Spot")
        report_lines.append("-" * 40)
        report_lines.append(
            "Calibration rationale: raw merit-order prices represent a structural SRMC-based fair value, "
            "while actual monthly spot embeds scarcity, balancing, and market premia. "
            "The calibration is therefore applied to the residual premium above raw price, not to the full price level."
        )
        report_lines.append(f"Rolling scheme: expanding window with minimum {int(oos_metrics['min_train_months'])} training months")
        report_lines.append(f"OOS window start: {oos_metrics['window_start']}")
        report_lines.append(f"OOS window end: {oos_metrics['window_end']}")
        report_lines.append(f"OOS months: {int(oos_metrics['months'])}")
        report_lines.append(f"SARIMAX MAE: {oos_metrics['sarimax_mae']:.3f} JPY/kWh")
        report_lines.append(f"Fundamental MAE: {oos_metrics['fundamental_mae']:.3f} JPY/kWh")
        report_lines.append(f"SARIMAX RMSE: {oos_metrics['sarimax_rmse']:.3f} JPY/kWh")
        report_lines.append(f"Fundamental RMSE: {oos_metrics['fundamental_rmse']:.3f} JPY/kWh")
        report_lines.append(f"Average SARIMAX - Actual spread: {oos_metrics['sarimax_mean_error']:.3f} JPY/kWh")
        report_lines.append(f"Average Fundamental - Actual spread: {oos_metrics['fundamental_mean_error']:.3f} JPY/kWh")
        report_lines.append("")

    if chart_paths:
        report_lines.append("## Visualization Output")
        report_lines.append("-" * 40)
        for label, path in chart_paths.items():
            report_lines.append(f"{label}: {path}")
        report_lines.append("")

    report_lines.append("=" * 60)
    return "\n".join(report_lines)


def _load_latest_pricing_forecast(region: JapanRegion, results_dir: Path) -> pd.DataFrame:
    """Load and monthly-average the best available pricing forecast output for a region."""
    search_dir = results_dir if any(results_dir.glob(f"model_comparison_{region.value}_*.csv")) else PROJECT_ROOT / "results"
    comparison_files = sorted(search_dir.glob(f"model_comparison_{region.value}_*.csv"))
    if not comparison_files:
        return pd.DataFrame()

    best_entry = None
    preferred_entries = []
    fallback_entries = []
    for comparison_file in comparison_files:
        try:
            comparison_df = pd.read_csv(comparison_file)
        except Exception:
            continue
        if comparison_df.empty or "CompositeScore" not in comparison_df.columns:
            continue
        row = comparison_df.iloc[0]
        try:
            score = float(row["CompositeScore"])
        except Exception:
            continue
        timestamp = comparison_file.stem.replace(f"model_comparison_{region.value}_", "")
        forecast_file = search_dir / f"price_forecast_{region.value}_{timestamp}.csv"
        if not forecast_file.exists():
            continue
        candidate = {
            "score": score,
            "model": str(row.get("Model", "Unknown")),
            "forecast_file": forecast_file.name,
            "forecast_path": forecast_file,
            "comparison_file": comparison_file.name,
            "timestamp": timestamp,
        }
        if candidate["model"].strip().lower() == "sarimax":
            preferred_entries.append(candidate)
        else:
            fallback_entries.append(candidate)

    candidate_pool = preferred_entries if preferred_entries else fallback_entries
    for candidate in candidate_pool:
        if best_entry is None or candidate["score"] > best_entry["score"]:
            best_entry = candidate

    if best_entry is None:
        return pd.DataFrame()

    forecast_df = pd.read_csv(best_entry["forecast_path"])
    if forecast_df.empty or "datetime" not in forecast_df.columns or "forecast" not in forecast_df.columns:
        return pd.DataFrame()

    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
    forecast_df["month"] = forecast_df["datetime"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        forecast_df.groupby("month", as_index=False)["forecast"]
        .mean()
        .rename(columns={"forecast": "pricing_price_jpy_kwh"})
    )
    monthly["pricing_forecast_file"] = best_entry["forecast_file"]
    monthly["pricing_model"] = best_entry["model"]
    monthly["pricing_composite_score"] = best_entry["score"]
    return monthly


def _load_model_oos_monthly_comparison(region: JapanRegion, model_name: str, results_dir: Path) -> pd.DataFrame:
    """Load monthly OOS comparison data for a specific model from monthly comparison files."""
    search_dir = results_dir if any(results_dir.glob(f"monthly_comparison_{region.value}_*.csv")) else PROJECT_ROOT / "results"
    files = sorted(search_dir.glob(f"monthly_comparison_{region.value}_*.csv"))
    if not files:
        return pd.DataFrame()

    target_prefix = model_name.strip().lower()
    best_df = None
    best_mtime = None

    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception:
            continue
        model_col = f"{target_prefix}_mean"
        if df.empty or "year_month" not in df.columns or "actual_mean" not in df.columns or model_col not in df.columns:
            continue
        mtime = file.stat().st_mtime
        if best_df is None or mtime > best_mtime:
            tmp = df.copy()
            tmp["month"] = pd.to_datetime(tmp["year_month"].astype(str)).dt.to_period("M").dt.to_timestamp()
            tmp = tmp.rename(
                columns={
                    "actual_mean": "actual_spot_price_jpy_kwh",
                    model_col: "sarimax_oos_price_jpy_kwh",
                }
            )
            tmp["oos_file"] = file.name
            best_df = tmp[["month", "actual_spot_price_jpy_kwh", "sarimax_oos_price_jpy_kwh", "oos_file"]]
            best_mtime = mtime

    return best_df if best_df is not None else pd.DataFrame()


def _load_actual_monthly_spot(region: JapanRegion) -> pd.DataFrame:
    """Load historical spot prices and aggregate to month for reference."""
    try:
        data = load_jepx_data(region)
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    data["month"] = data["datetime"].dt.to_period("M").dt.to_timestamp()
    return (
        data.groupby("month", as_index=False)["price"]
        .mean()
        .rename(columns={"price": "actual_spot_price_jpy_kwh"})
    )


def _load_forward_history(region: JapanRegion, load_shape: str = "baseload") -> pd.DataFrame:
    """Load standardized historical forward curve history if available."""
    if region != JapanRegion.TOKYO or load_shape != "baseload":
        return pd.DataFrame()

    path = PROJECT_ROOT / "data" / "processed" / "forward" / "mosaic_tokyo_baseload_monthly.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["observation_date", "delivery_month"])
    if df.empty:
        return pd.DataFrame()

    return df.sort_values(["observation_date", "delivery_month"]).reset_index(drop=True)


def _load_sarimax_optimal_params() -> dict:
    """Load saved optimal SARIMAX parameters."""
    path = PROJECT_ROOT / "results" / "optimal_params" / "sarimax_optimal_20260228.yaml"
    if not path.exists():
        return {"order": (0, 1, 0), "seasonal_order": (0, 1, 1, 7)}

    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    params = payload.get("optimal_params", {})
    order = tuple(params.get("order", [0, 1, 0]))
    seasonal_order = tuple(params.get("seasonal_order", [0, 1, 1, 7]))
    return {"order": order, "seasonal_order": seasonal_order}


def _generate_strict_sarimax_monthly_oos(
    region: JapanRegion,
    start_month: str = "2024-01",
    end_month: str = "2026-01",
) -> pd.DataFrame:
    """Generate strict monthly OOS SARIMAX forecasts with a unique as-of date per delivery month."""
    from src.models.ml_enhanced_v3.sarimax_model import SimplifiedSARIMAX

    params = _load_sarimax_optimal_params()
    data = load_jepx_data(region)
    if data.empty:
        return pd.DataFrame()

    data = data.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values("datetime")

    months = pd.period_range(start=start_month, end=end_month, freq="M")
    rows = []
    for month_period in months:
        month_start = month_period.to_timestamp()
        month_end = (month_start + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=23, minutes=30)
        train_end = month_start - pd.Timedelta(minutes=30)
        train_start = month_start - pd.DateOffset(years=3)

        train_data = data[(data["datetime"] >= train_start) & (data["datetime"] <= train_end)].copy()
        actual_data = data[(data["datetime"] >= month_start) & (data["datetime"] <= month_end)].copy()
        if train_data.empty or actual_data.empty:
            continue

        model = SimplifiedSARIMAX(
            order=params["order"],
            seasonal_order=params["seasonal_order"],
        )
        model._structure_fixed = True
        model.selected_exog_cols = []
        model.fit(train_data=train_data, datetime_col="datetime", price_col="price")

        forecast_horizon = len(actual_data)
        forecast = model.predict(forecast_horizon=forecast_horizon, start_date=month_start)
        rows.append(
            {
                "month": month_start,
                "as_of_date": train_end.normalize(),
                "sarimax_strict_oos_price_jpy_kwh": float(np.mean(forecast.point_forecast)),
                "actual_spot_price_jpy_kwh": float(actual_data["price"].mean()),
                "train_start": train_data["datetime"].min(),
                "train_end_timestamp": train_data["datetime"].max(),
                "forecast_start": month_start,
                "forecast_end": month_end,
            }
        )

    return pd.DataFrame(rows)


def _match_forward_curve_to_strict_sarimax(
    strict_oos_df: pd.DataFrame,
    forward_history: pd.DataFrame,
) -> pd.DataFrame:
    """Match strict monthly SARIMAX OOS forecasts to market forwards on the same or latest prior observation date."""
    if strict_oos_df.empty or forward_history.empty:
        return pd.DataFrame()

    candidates = strict_oos_df.merge(
        forward_history,
        left_on="month",
        right_on="delivery_month",
        how="left",
    )
    candidates = candidates[candidates["observation_date"] <= candidates["as_of_date"]].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(["month", "observation_date"])
    matched = candidates.groupby("month", as_index=False).tail(1).copy()
    matched = matched.rename(columns={"forward_price_jpy_kwh": "market_forward_price_jpy_kwh"})
    matched["sarimax_vs_forward_error_jpy_kwh"] = (
        matched["sarimax_strict_oos_price_jpy_kwh"] - matched["market_forward_price_jpy_kwh"]
    )
    return matched[
        [
            "month",
            "as_of_date",
            "observation_date",
            "contract",
            "rank",
            "market_forward_price_jpy_kwh",
            "sarimax_strict_oos_price_jpy_kwh",
            "actual_spot_price_jpy_kwh",
            "sarimax_vs_forward_error_jpy_kwh",
            "source_alias",
        ]
    ].rename(columns={"observation_date": "market_observation_date"})


def _compute_oos_metrics(comparison_df: pd.DataFrame) -> dict:
    """Compute OOS error metrics for SARIMAX and calibrated fundamental vs actual spot."""
    if comparison_df.empty:
        return {}

    required_cols = {
        "month",
        "actual_spot_price_jpy_kwh",
        "sarimax_oos_price_jpy_kwh",
        "calibrated_price_jpy_kwh",
    }
    if not required_cols.issubset(comparison_df.columns):
        return {}

    df = comparison_df.dropna(
        subset=["actual_spot_price_jpy_kwh", "sarimax_oos_price_jpy_kwh", "calibrated_price_jpy_kwh"]
    ).copy()
    if df.empty:
        return {}

    sarimax_err = df["sarimax_oos_price_jpy_kwh"] - df["actual_spot_price_jpy_kwh"]
    fundamental_err = df["calibrated_price_jpy_kwh"] - df["actual_spot_price_jpy_kwh"]

    return {
        "window_start": df["month"].min().strftime("%Y-%m"),
        "window_end": df["month"].max().strftime("%Y-%m"),
        "months": len(df),
        "min_train_months": float(df["train_size"].min()) if "train_size" in df.columns else np.nan,
        "sarimax_mae": sarimax_err.abs().mean(),
        "fundamental_mae": fundamental_err.abs().mean(),
        "sarimax_rmse": float(np.sqrt((sarimax_err ** 2).mean())),
        "fundamental_rmse": float(np.sqrt((fundamental_err ** 2).mean())),
        "sarimax_mean_error": sarimax_err.mean(),
        "fundamental_mean_error": fundamental_err.mean(),
    }


def run_fundamental_analysis(region: JapanRegion, output_dir: str, standardized_data_dir: str):
    """
    Run fundamental analysis using standardized LiveSheet inputs and plant-level clearing logic.
    
    Args:
        region: Target region
        output_dir: Output directory
    """
    logger.info("Running fundamental analysis...")

    live_sheet_file = PROJECT_ROOT / "data" / "2026-02-09 - Japan Merit Order LiveSheet - 1.2.xlsx"
    contract_calendar_file = PROJECT_ROOT / "data" / "elec_unit.csv"
    extractor = LiveSheetExtractor(live_sheet_file, contract_calendar_file)
    standardized_dir = Path(standardized_data_dir)
    standardized_dir.mkdir(parents=True, exist_ok=True)

    try:
        standardized = extractor.extract_standardized_data(region)
        standardized_files = extractor.save_standardized_data(
            standardized,
            output_dir=standardized_data_dir,
            region=region,
        )
    except Exception as exc:
        logger.warning(f"LiveSheet extraction failed, falling back to cached standardized data: {exc}")
        plant_master_file = standardized_dir / "plant_master.csv"
        monthly_demand_file = standardized_dir / "monthly_demand.csv"
        dashboard_inputs_file = standardized_dir / "dashboard_inputs.csv"
        monthly_inputs_file = standardized_dir / f"monthly_market_inputs_{region.value}.csv"
        contract_calendar_file = standardized_dir / "contract_calendar.csv"

        required_files = [
            plant_master_file,
            monthly_demand_file,
            dashboard_inputs_file,
            monthly_inputs_file,
        ]
        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"LiveSheet could not be read and cached standardized files are missing: {missing_files}"
            ) from exc

        standardized = StandardizedFundamentalData(
            plant_master=pd.read_csv(plant_master_file, parse_dates=["start_date", "end_date"]),
            monthly_demand=pd.read_csv(monthly_demand_file, parse_dates=["month"]),
            dashboard_inputs=pd.read_csv(dashboard_inputs_file, parse_dates=["month"]),
            monthly_market_inputs=pd.read_csv(monthly_inputs_file, parse_dates=["month"]),
            contract_calendar=(
                pd.read_csv(contract_calendar_file, parse_dates=[
                    "contract_month_start",
                    "baseload_first_trading_day",
                    "baseload_last_trading_day",
                    "baseload_final_settlement_day",
                    "peakload_first_trading_day",
                    "peakload_last_trading_day",
                    "peakload_final_settlement_day",
                    "weekly_period_start",
                    "weekly_period_end",
                ])
                if contract_calendar_file.exists() else pd.DataFrame()
            ),
        )
        standardized_files = {
            "plant_master": plant_master_file,
            "monthly_demand": monthly_demand_file,
            "dashboard_inputs": dashboard_inputs_file,
            "monthly_market_inputs": monthly_inputs_file,
            "contract_calendar": contract_calendar_file,
        }

    engine = MonthlyFundamentalEngine(region=region)
    run_result = engine.run(
        plant_master=standardized.plant_master,
        monthly_market_inputs=standardized.monthly_market_inputs,
    )

    monthly_prices = run_result.monthly_prices.copy()
    monthly_stacks = run_result.monthly_stacks.copy()
    actual_monthly = _load_actual_monthly_spot(region)
    calibrator = FundamentalPriceCalibrator(regularization=1.0)
    calibration_result = calibrator.calibrate(monthly_prices, actual_monthly)
    rolling_result = calibrator.rolling_oos(monthly_prices, actual_monthly, min_train_months=3)
    calibrated_monthly = calibration_result.calibrated_prices.copy()
    monthly_prices = monthly_prices.merge(
        calibrated_monthly[["month", "raw_fundamental_price_jpy_kwh", "predicted_premium_jpy_kwh", "calibrated_price_jpy_kwh"]],
        on="month",
        how="left",
    )
    historical_monthly_prices = monthly_prices.copy()
    pricing_monthly = _load_latest_pricing_forecast(region, Path(output_dir))
    sarimax_oos_monthly = _load_model_oos_monthly_comparison(region, "sarimax", Path(output_dir))

    pricing_source = None
    if not pricing_monthly.empty:
        forecast_months = pricing_monthly["month"].sort_values().unique()
        monthly_prices = monthly_prices[monthly_prices["month"].isin(forecast_months)].copy()
        monthly_stacks = monthly_stacks[monthly_stacks["month"].isin(forecast_months)].copy()
        pricing_source = {
            "model": pricing_monthly["pricing_model"].iloc[0],
            "forecast_file": pricing_monthly["pricing_forecast_file"].iloc[0],
            "composite_score": float(pricing_monthly["pricing_composite_score"].iloc[0]),
        }

    comparison_df = monthly_prices[
        ["month", "raw_fundamental_price_jpy_kwh", "calibrated_price_jpy_kwh", "clearing_price_jpy_kwh"]
    ].copy()
    comparison_df["fundamental_price_jpy_kwh"] = comparison_df["calibrated_price_jpy_kwh"].fillna(
        comparison_df["clearing_price_jpy_kwh"]
    )
    if not actual_monthly.empty:
        comparison_df = comparison_df.merge(actual_monthly, on="month", how="left")
    if not pricing_monthly.empty:
        comparison_df = comparison_df.merge(pricing_monthly, on="month", how="left")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"fundamental_analysis_{region.value}_{timestamp}.txt"
    monthly_price_file = output_path / f"fundamental_monthly_prices_{region.value}_{timestamp}.csv"
    monthly_stack_file = output_path / f"fundamental_monthly_stack_{region.value}_{timestamp}.csv"
    comparison_file = output_path / f"fundamental_vs_pricing_{region.value}_{timestamp}.csv"
    comparison_chart_file = output_path / f"fundamental_comparison_{region.value}_{timestamp}.png"
    drivers_chart_file = output_path / f"fundamental_drivers_{region.value}_{timestamp}.png"
    calibration_chart_file = output_path / f"fundamental_calibration_bridge_{region.value}_{timestamp}.png"
    oos_comparison_file = output_path / f"fundamental_vs_sarimax_oos_{region.value}_{timestamp}.csv"
    oos_chart_file = output_path / f"fundamental_vs_sarimax_oos_{region.value}_{timestamp}.png"
    strict_sarimax_oos_file = output_path / f"strict_sarimax_monthly_oos_{region.value}_{timestamp}.csv"
    strict_forward_validation_file = output_path / f"strict_forward_validation_{region.value}_{timestamp}.csv"
    strict_forward_validation_chart_file = output_path / f"strict_forward_validation_{region.value}_{timestamp}.png"

    oos_comparison_df = rolling_result.oos_predictions.copy()
    if not oos_comparison_df.empty:
        oos_comparison_df = oos_comparison_df.rename(
            columns={"rolling_calibrated_price_jpy_kwh": "calibrated_price_jpy_kwh"}
        )
        oos_comparison_df["clearing_price_jpy_kwh"] = oos_comparison_df["raw_fundamental_price_jpy_kwh"]
    if not sarimax_oos_monthly.empty and not oos_comparison_df.empty:
        oos_comparison_df = oos_comparison_df.merge(
            sarimax_oos_monthly[["month", "sarimax_oos_price_jpy_kwh", "oos_file"]],
            on="month",
            how="inner",
        )
    else:
        oos_comparison_df = pd.DataFrame()
    oos_metrics = _compute_oos_metrics(oos_comparison_df)
    forward_history = _load_forward_history(region, load_shape="baseload")
    strict_sarimax_oos_df = _generate_strict_sarimax_monthly_oos(region) if not forward_history.empty else pd.DataFrame()
    strict_forward_validation_df = _match_forward_curve_to_strict_sarimax(
        strict_sarimax_oos_df,
        forward_history,
    )

    chart_paths = {}
    try:
        chart_paths["comparison_chart"] = plot_fundamental_comparison(
            comparison_df,
            str(comparison_chart_file),
            title=f"Fundamental vs Actual vs Pricing - {region.value.capitalize()}",
        )
    except Exception as exc:
        logger.warning(f"Failed to generate fundamental comparison chart: {exc}")

    try:
        chart_paths["drivers_chart"] = plot_fundamental_drivers(
            monthly_prices,
            str(drivers_chart_file),
            title=f"Fundamental Drivers - {region.value.capitalize()}",
        )
    except Exception as exc:
        logger.warning(f"Failed to generate fundamental drivers chart: {exc}")

    try:
        chart_paths["calibration_bridge_chart"] = plot_calibration_bridge(
            comparison_df,
            str(calibration_chart_file),
            title=f"Raw to Calibrated Fundamental - {region.value.capitalize()}",
        )
    except Exception as exc:
        logger.warning(f"Failed to generate calibration bridge chart: {exc}")

    if not oos_comparison_df.empty:
        try:
            chart_paths["sarimax_oos_chart"] = plot_oos_model_vs_fundamental(
                oos_comparison_df,
                str(oos_chart_file),
                title=f"SARIMAX OOS vs Fundamental vs Actual - {region.value.capitalize()}",
            )
        except Exception as exc:
            logger.warning(f"Failed to generate SARIMAX OOS comparison chart: {exc}")

    if not strict_forward_validation_df.empty:
        try:
            chart_paths["strict_forward_validation_chart"] = plot_strict_sarimax_forward_actual(
                strict_forward_validation_df,
                str(strict_forward_validation_chart_file),
                title=f"Strict Monthly OOS: SARIMAX vs Market Forward vs Actual - {region.value.capitalize()}",
            )
        except Exception as exc:
            logger.warning(f"Failed to generate strict forward validation chart: {exc}")

    report = _format_fundamental_report(
        region,
        monthly_prices,
        comparison_df,
        chart_paths,
        pricing_source,
        calibration_result.metrics,
        oos_metrics,
    )
    print(report)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    monthly_prices.to_csv(monthly_price_file, index=False)
    monthly_stacks.to_csv(monthly_stack_file, index=False)
    comparison_df.to_csv(comparison_file, index=False)
    if not oos_comparison_df.empty:
        oos_comparison_df.to_csv(oos_comparison_file, index=False)
    if not strict_sarimax_oos_df.empty:
        strict_sarimax_oos_df.to_csv(strict_sarimax_oos_file, index=False)
    if not strict_forward_validation_df.empty:
        strict_forward_validation_df.to_csv(strict_forward_validation_file, index=False)
    calibration_result.coefficients.to_csv(
        output_path / f"fundamental_calibration_coefficients_{region.value}_{timestamp}.csv",
        index=False,
    )

    logger.info(f"Fundamental analysis report saved to {report_file}")
    logger.info(f"Monthly prices saved to {monthly_price_file}")
    logger.info(f"Monthly stack saved to {monthly_stack_file}")
    logger.info(f"Comparison output saved to {comparison_file}")
    if not oos_comparison_df.empty:
        logger.info(f"SARIMAX OOS comparison output saved to {oos_comparison_file}")
    if not strict_sarimax_oos_df.empty:
        logger.info(f"Strict SARIMAX monthly OOS saved to {strict_sarimax_oos_file}")
    if not strict_forward_validation_df.empty:
        logger.info(f"Strict forward validation output saved to {strict_forward_validation_file}")
    for label, path in chart_paths.items():
        logger.info(f"{label} saved to {path}")
    logger.info(f"Standardized data saved under {standardized_data_dir}")
    for name, file_path in standardized_files.items():
        logger.info(f"  {name}: {file_path}")


def run_analysis_command(args):
    """Execute the analysis command."""
    logger.info("=" * 80)
    logger.info("ANALYSIS COMMAND")
    logger.info("=" * 80)
    logger.info("Analysis Type: fundamental")
    logger.info(f"Region: {args.region}")
    
    region = JapanRegion(args.region)
    
    try:
        run_fundamental_analysis(
            region,
            args.output_dir,
            getattr(args, 'standardized_data_dir', 'data/processed/fundamental')
        )
            
        logger.info("=" * 80)
        logger.info("ANALYSIS COMMAND COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in analysis command: {e}")
        import traceback
        traceback.print_exc()
        return 1
