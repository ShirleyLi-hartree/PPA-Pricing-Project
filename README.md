# PPA Pricing Framework

This repository contains a practical PPA pricing framework for the Japanese power market. The objective is not to claim perfect price forecasting, but to provide a complete, explainable, and reusable methodology that connects data, market assumptions, model outputs, validation, and pricing risk into one workflow.

The current framework integrates four lines of work:
- Statistical forecasting, with `SARIMAX` as the primary forecasting model
- Fundamental pricing, based on Japan Merit Order LiveSheet inputs and plant-level monthly merit-order clearing
- Market validation, using Tokyo monthly baseload forward history aligned with strict monthly OOS forecasts
- Risk pricing, using strict OOS forecast errors to generate pricing ranges, contract-level PnL, VaR / ES, and no-loss quote ladders

Supporting documentation:
- [Project Report](docs/PROJECT_REPORT.md)
- [Project Guide](docs/PROJECT_GUIDE.md)

## Current Conclusions

The latest Tokyo conclusions are:
- `SARIMAX` is the most suitable primary statistical pricing model in the current framework
- `Market Forward` is closer to realized `actual spot` than strict monthly OOS `SARIMAX`, and should be treated as a market anchor
- `Fundamental` raw prices provide a structural anchor; residual-calibrated fundamental prices provide a comparable structural fair value
- Final pricing should not stop at a single point estimate; the framework now produces empirical pricing ranges, seller-side risk metrics, and no-loss quote levels

## Environment Setup

Recommended Python version: `3.10+`

```bash
pip install -r requirements.txt
```

If `jpholiday` is not available locally, the code falls back gracefully and does not block the main workflow.

## Repository Structure

```text
src/
  cli/
    commands/
      pricing.py
      forecast.py
      analysis.py
  core/
    pricing/
      ppa_pricing_engine.py
      sarimax_risk_pricing.py
  data/
    loaders/
      live_sheet_loader.py
      mosaic_forward_loader.py
  forecasting/
    core.py
    grid_search.py
    training/
    evaluation/
  models/
    ml_enhanced_v3/
      sarimax_model.py
      prophet_model.py
      xgboost_model.py
    fundamental_v2/
      monthly_fundamental.py
      calibration.py
  visualization/
    fundamental_analysis.py
    model_comparison.py
data/
results/
docs/
```

## Data Inputs

The framework currently relies on three core data groups:

1. JEPX historical spot data
- Used for SARIMAX / Prophet / XGBoost training and OOS validation

2. Japan Merit Order LiveSheet
- Used to extract `plant master`, `monthly demand`, `availability`, and `generation cost`
- Used to build the monthly fundamental merit-order clearing logic

3. Mosaic monthly forward curve
- Alias: `JAPAN-BASE-POWER-MONTH-TOKYO-FOBM`
- Used for strict monthly OOS market-forward validation

## Main Commands

### 1. Forecast

```bash
python -m src.cli.main forecast --region tokyo --forecast-days 90
```

Use this command to:
- run rolling backtests across forecasting models
- generate `monthly_comparison`, `model_comparison`, and `price_forecast`

### 2. Fundamental Analysis

```bash
python -m src.cli.main analysis --region tokyo --output-dir results --standardized-data-dir data/processed/fundamental
```

Use this command to:
- standardize LiveSheet inputs
- generate raw / calibrated monthly fundamental prices
- produce validation charts and reports against SARIMAX, market forward, and actual spot

### 3. Pricing

Standard run:

```bash
python -m src.cli.main pricing --region tokyo --models sarimax historical --forecast-months 3 --volume 2 --output-dir results
```

If reusing an existing forecast file:

```bash
python -m src.cli.main pricing --region tokyo --external-forecast-file results\\price_forecast_tokyo_20260305_111746.csv --forecast-months 3 --volume 2 --output-dir results --skip-validation
```

Use this command to:
- generate PPA point pricing
- output model comparison results
- produce SARIMAX risk ranges, contract bootstrap risk metrics, and quote ladders

## Key Outputs

Important output files include:

Forecasting:
- `results/model_comparison_tokyo_20260305_111746.csv`
- `results/price_forecast_tokyo_20260305_111746.csv`

Fundamental:
- `results/fundamental_analysis_tokyo_20260320_110335.txt`
- `results/fundamental_vs_sarimax_oos_tokyo_20260320_110335.png`
- `results/strict_forward_validation_tokyo_20260320_110335.png`

SARIMAX risk pricing:
- `results/sarimax_risk_report_tokyo_20260320_142836.txt`
- `results/sarimax_contract_risk_tokyo_20260320_142836.csv`
- `results/sarimax_quote_bands_tokyo_20260320_142836.png`
- `results/sarimax_pnl_distribution_tokyo_20260320_142836.png`

## Latest Tokyo Snapshot

Latest Tokyo metrics:
- Fundamental raw MAE: `6.893 JPY/kWh`
- Fundamental calibrated MAE: `0.732 JPY/kWh`
- SARIMAX OOS MAE vs actual: `0.902 JPY/kWh`
- Fundamental OOS MAE vs actual: `1.041 JPY/kWh`
- Market Forward MAE vs actual: `0.820 JPY/kWh`
- SARIMAX strict OOS MAE vs actual: `1.864 JPY/kWh`

Three-month SARIMAX contract pricing metrics:
- Point quote / fair value: `13.127 JPY/kWh`
- Risk-neutral quote: `14.068 JPY/kWh`
- 90% no-loss quote: `16.224 JPY/kWh`
- Expected PnL at point quote: `-0.951 JPY/kWh`
- 95% VaR loss: `2.802 JPY/kWh`
- 95% ES loss: `3.170 JPY/kWh`

## Recommended Usage

The current business interpretation is:
- `Market Forward` as market anchor and sanity check
- `Fundamental` as structural explanation layer
- `SARIMAX` as the primary pricing engine
- strict OOS error history as the basis for risk-neutral and no-loss quote construction

## Notes

- This repository is intended as a releaseable pricing framework and a reusable internal knowledge base
- The framework is complete enough for management reporting and further deployment in a shared git repository
- Future work should focus on seasonal / regime segmentation, historical snapshot fundamental construction, and pricing policy refinement
