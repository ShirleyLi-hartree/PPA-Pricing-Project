# Project Guide

## 1. Repository Purpose

This repository supports a full PPA pricing workflow for the Japanese power market. It combines forecasting, structural pricing, market validation, and pricing-risk analysis in one codebase.

The design goal is to make the workflow:
- reproducible
- explainable
- maintainable
- reusable across future model improvements

## 2. Top-Level Structure

```text
config/      configuration files
data/        raw input data
docs/        documentation and final reports
results/     local run outputs
src/         application and pricing logic
```

## 3. Source Structure

### 3.1 CLI Layer

Entry point:
- `src/cli/main.py`

Commands:
- `pricing`
- `forecast`
- `analysis`

### 3.2 Pricing Layer

Key files:
- `src/cli/commands/pricing.py`
- `src/core/pricing/ppa_pricing_engine.py`
- `src/core/pricing/sarimax_risk_pricing.py`

Responsibilities:
- train or reuse forecast outputs
- compute PPA point pricing
- generate SARIMAX-based quote ranges and contract risk metrics

### 3.3 Forecasting Layer

Key files:
- `src/forecasting/core.py`
- `src/forecasting/grid_search.py`
- `src/forecasting/training/rolling_trainer.py`
- `src/forecasting/evaluation/backtester.py`

Responsibilities:
- rolling backtest execution
- model comparison
- price forecast generation

### 3.4 Model Layer

Statistical and ML models:
- `src/models/ml_enhanced_v3/sarimax_model.py`
- `src/models/ml_enhanced_v3/prophet_model.py`
- `src/models/ml_enhanced_v3/xgboost_model.py`
- `src/models/statistics_v1/historical_average.py`

Fundamental models:
- `src/models/fundamental_v2/monthly_fundamental.py`
- `src/models/fundamental_v2/calibration.py`

### 3.5 Data Loaders

Key files:
- `src/data/loaders/live_sheet_loader.py`
- `src/data/loaders/mosaic_forward_loader.py`

Responsibilities:
- standardize LiveSheet-based fundamental inputs
- standardize Mosaic forward history

### 3.6 Visualization Layer

Key files:
- `src/visualization/fundamental_analysis.py`
- `src/visualization/model_comparison.py`

Responsibilities:
- model comparison charts
- fundamental vs forward vs actual validation charts
- quote band and PnL distribution charts

## 4. Key Input Files

### 4.1 LiveSheet Inputs

Files:
- `data/2026-02-09 - Japan Merit Order LiveSheet - 1.2.xlsx`
- `data/elec_unit.csv`

Use:
- monthly fundamental input construction

### 4.2 Forward Inputs

Standardized output:
- `data/processed/forward/mosaic_tokyo_baseload_monthly.csv`

Source alias:
- `JAPAN-BASE-POWER-MONTH-TOKYO-FOBM`

### 4.3 Important Result Files

Reusable result files:
- `results/model_comparison_tokyo_20260305_111746.csv`
- `results/price_forecast_tokyo_20260305_111746.csv`
- `results/fundamental_analysis_tokyo_20260320_110335.txt`
- `results/strict_forward_validation_tokyo_20260320_110335.csv`
- `results/sarimax_risk_report_tokyo_20260320_142836.txt`

## 5. How to Run

### 5.1 Forecast

```bash
python -m src.cli.main forecast --region tokyo --forecast-days 90
```

Purpose:
- run rolling backtests
- generate `model_comparison`, `monthly_comparison`, and `price_forecast`

### 5.2 Fundamental Analysis

```bash
python -m src.cli.main analysis --region tokyo --output-dir results --standardized-data-dir data/processed/fundamental
```

Purpose:
- standardize LiveSheet data
- generate fundamental reports, charts, and forward validation outputs

### 5.3 Pricing

Standard run:

```bash
python -m src.cli.main pricing --region tokyo --models sarimax historical --forecast-months 3 --volume 2 --output-dir results
```

Reuse an existing forecast:

```bash
python -m src.cli.main pricing --region tokyo --external-forecast-file results\\price_forecast_tokyo_20260305_111746.csv --forecast-months 3 --volume 2 --output-dir results --skip-validation
```

Purpose:
- output point pricing
- output PPA pricing results
- output SARIMAX risk ranges and contract-level PnL analysis

## 6. Recommended Reading Order

To understand the repository quickly, read in this order:

1. `README.md`
2. `docs/PROJECT_REPORT.md`
3. `src/cli/commands/pricing.py`
4. `src/core/pricing/sarimax_risk_pricing.py`
5. `src/models/fundamental_v2/monthly_fundamental.py`
6. `src/models/fundamental_v2/calibration.py`
7. `src/data/loaders/live_sheet_loader.py`

## 7. Methodological Scope

What is already covered:
- Tokyo monthly baseload pricing workflow
- unified validation across SARIMAX, Fundamental, Forward, and Actual
- SARIMAX-based pricing-risk outputs

What is not yet expanded:
- full regional replication
- historical snapshot fundamental inputs
- seasonal or regime-based residual segmentation
- integration with production databases or downstream trading systems

## 8. Maintenance Principles

### 8.1 Code Principles

- keep `pricing` as the business-facing pricing workflow
- keep `analysis` as the validation and explanation workflow
- any new model must be accompanied by OOS validation and comparison against actual / forward
- avoid reintroducing isolated modules that are not connected to the main workflow

### 8.2 Result Management

- `results/` should remain a local output directory rather than a long-term versioned artifact store
- if specific outputs need to be preserved, reference them in `docs/` or move them to a dedicated archive path

### 8.3 Future Enhancement Principles

- prioritize residual segmentation before adding model complexity
- preserve strict alignment across forecast, forward, and actual whenever possible
- any pricing uplift must have a clear historical and economic interpretation

## 9. Recommended Git Contents

Recommended for repository publication:
- `src/`
- `config/`
- `requirements.txt`
- `README.md`
- `docs/PROJECT_REPORT.md`
- `docs/PROJECT_GUIDE.md`

Recommended to exclude:
- `.venv/`
- `results/`
- `data/processed/`
- `__pycache__/`
