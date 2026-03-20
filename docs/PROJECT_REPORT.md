# Project Report

## 1. Objective

This project delivers a complete and explainable PPA pricing framework for the Japanese power market. The purpose is not to claim perfect price prediction, but to establish a robust methodology that connects data, assumptions, models, validation, risk analysis, and pricing decisions in one coherent process.

The framework is designed to be:
- transparent in data usage and assumptions
- explainable at model and pricing level
- verifiable against both realized spot prices and market forward prices
- reusable for internal knowledge transfer and future enhancement

## 2. Data Architecture

### 2.1 JEPX Historical Spot Data

Purpose:
- train and backtest `SARIMAX`, `Prophet`, `XGBoost`, and `Historical Average`
- aggregate realized spot prices into monthly actuals for validation and pricing-risk statistics

### 2.2 Japan Merit Order LiveSheet

Core source files:
- `data/2026-02-09 - Japan Merit Order LiveSheet - 1.2.xlsx`
- `data/elec_unit.csv`

Standardized tables:
- `data/processed/fundamental/plant_master.csv`
- `data/processed/fundamental/monthly_demand.csv`
- `data/processed/fundamental/dashboard_inputs.csv`
- `data/processed/fundamental/monthly_market_inputs_tokyo.csv`
- `data/processed/fundamental/contract_calendar.csv`

Purpose:
- `plant_master` defines unit capacity, technology, efficiency, start date, and end date
- `monthly_demand` provides monthly Tokyo / East demand assumptions
- `dashboard_inputs` provides technology-level availability and generation cost
- `contract_calendar` provides Baseload / Peakload calendar mapping

### 2.3 Mosaic Forward Curve

Historical source:
- alias: `JAPAN-BASE-POWER-MONTH-TOKYO-FOBM`

Standardized table:
- `data/processed/forward/mosaic_tokyo_baseload_monthly.csv`

Purpose:
- provide monthly Tokyo baseload forward observations
- validate model-implied monthly fair values against market-observed forward levels

## 3. Assumptions and Methodology

### 3.1 Statistical Forecasting Layer

After model comparison, `SARIMAX` was selected as the primary statistical pricing model. The selection is based on overall balance rather than a single metric: stability, interpretability, and usability for monthly pricing matter more than purely maximizing one backtest score.

Relevant outputs:
- `results/model_comparison_tokyo_20260305_111746.csv`
- `results/price_forecast_tokyo_20260305_111746.csv`

Current Tokyo `SARIMAX` metrics:
- Composite Score: `0.655`
- MAE mean: `2.787 JPY/kWh`
- Stability score: `0.775`

### 3.2 Fundamental Pricing Layer

The monthly fundamental model works as follows:

1. select active plants from `plant_master`
2. compute `effective_capacity` using technology-level availability
3. build plant-level `SRMC` using generation cost and plant efficiency
4. sort by `SRMC` to form a monthly merit order
5. identify the marginal unit that satisfies demand
6. use the marginal unit `SRMC` as raw monthly clearing price

Economic interpretation of raw fundamental price:
- it is a structural short-run marginal cost anchor
- it reflects plant stack ordering and demand positioning
- it does not fully capture scarcity, balancing premia, or trading premia embedded in observed spot prices

For that reason, the project does not calibrate the full price level directly. Instead, it calibrates the residual premium:

```text
premium = actual spot - raw fundamental
```

This preserves structural interpretation while allowing the model to align with observed monthly market levels.

### 3.3 Market Validation Layer

A forecasting model should not only be checked against realized spot prices. For PPA pricing, it should also be checked against market-observed forward levels.

The project therefore uses a strict monthly OOS definition:
- for each delivery month `M`
- use the final historical point of the previous month as `as_of_date`
- fit SARIMAX using fixed optimal parameters
- forecast the full month `M`
- compare that result with the same `as_of_date` forward contract `YYYYMM`

This ensures exact alignment across:
- forecast date
- market observation date
- delivery month

### 3.4 Risk Pricing Layer

The final pricing layer uses a simple but defensible logic:

1. the point quote is the monthly `SARIMAX` forecast
2. pricing ranges are built from the empirical strict OOS error distribution
3. contract risk is estimated via bootstrap resampling of historical monthly forecast errors

Error is defined as:

```text
error = forecast - actual
```

From a seller perspective:

```text
PnL = quote - actual
```

If `quote = forecast`, the historical OOS error distribution is already the empirical pricing-risk distribution.

## 4. Validation Results

### 4.1 Fundamental vs SARIMAX vs Actual Spot

Report:
- `results/fundamental_analysis_tokyo_20260320_110335.txt`

Key metrics:
- Raw MAE: `6.893 JPY/kWh`
- Calibrated MAE: `0.732 JPY/kWh`
- Raw RMSE: `6.969 JPY/kWh`
- Calibrated RMSE: `0.905 JPY/kWh`

Interpretation:
- raw fundamental prices are systematically too low, but they provide a valid structural anchor
- residual calibration brings the price level back into a comparable range without losing structural meaning

Rolling OOS comparison:
- window: `2024-07` to `2026-01`
- months: `19`
- SARIMAX MAE: `0.902`
- Fundamental MAE: `1.041`
- SARIMAX RMSE: `1.266`
- Fundamental RMSE: `1.267`

Conclusion:
- calibrated fundamental is now close to SARIMAX on OOS realized-spot accuracy
- SARIMAX still remains the better primary pricing model in the current release

### 4.2 Strict Monthly OOS: SARIMAX vs Forward vs Actual

Relevant outputs:
- `results/strict_forward_validation_tokyo_20260320_110335.csv`
- `results/strict_forward_validation_tokyo_20260320_110335.png`

Sample window:
- `2024-02` to `2026-01`
- `24` months

Error metrics:
- Market Forward vs Actual
  - MAE: `0.820`
  - RMSE: `0.965`
- SARIMAX Strict OOS vs Actual
  - MAE: `1.864`
  - RMSE: `2.203`
- SARIMAX vs Forward
  - MAE: `1.730`

Conclusion:
- market forward is historically closer to realized monthly spot than strict monthly OOS SARIMAX
- forward should therefore be treated as a market anchor
- SARIMAX still adds value as an independent model-based fair-value view rather than as a pure market proxy

## 5. Risk Analysis and Pricing Outputs

### 5.1 Historical Strict OOS Error Statistics

Report:
- `results/sarimax_risk_report_tokyo_20260320_142836.txt`

Historical sample:
- `25` monthly strict OOS observations

Statistics:
- Mean error: `-0.942 JPY/kWh`
- MAE: `1.855 JPY/kWh`
- RMSE: `2.184 JPY/kWh`
- Probability of loss: `72.0%`
- Worst historical loss: `-4.839 JPY/kWh`
- Best historical gain: `2.711 JPY/kWh`

Quantiles:
- P05: `-3.470`
- P10: `-3.097`
- P25: `-2.670`
- P50: `-1.042`
- P75: `0.190`
- P90: `1.785`
- P95: `2.388`

Interpretation:
- the negative mean error shows that a seller quoting directly at the SARIMAX point forecast has historically tended to underquote on average
- the point forecast is therefore best interpreted as fair value, not as the final commercial offer

### 5.2 Three-Month Contract Pricing: 2026-02 to 2026-04

Relevant outputs:
- `results/sarimax_contract_risk_tokyo_20260320_142836.csv`
- `results/sarimax_quote_bands_tokyo_20260320_142836.png`
- `results/sarimax_pnl_distribution_tokyo_20260320_142836.png`

Point quote:
- Point quote / fair value: `13.127 JPY/kWh`

Risk-adjusted quote ladder:
- Risk-neutral quote: `14.068 JPY/kWh`
- 50% no-loss quote: `14.168 JPY/kWh`
- 75% no-loss quote: `15.796 JPY/kWh`
- 90% no-loss quote: `16.224 JPY/kWh`
- 95% no-loss quote: `16.596 JPY/kWh`

Bootstrapped actual delivery distribution:
- Actual P5: `12.144 JPY/kWh`
- Actual P50: `14.108 JPY/kWh`
- Actual P95: `15.928 JPY/kWh`

Contract PnL metrics:
- Expected PnL at point quote: `-0.951 JPY/kWh`
- Probability of loss: `79.3%`
- 95% VaR loss: `2.802 JPY/kWh`
- 95% Expected Shortfall loss: `3.170 JPY/kWh`
- Worst simulated loss: `-4.839 JPY/kWh`
- Best simulated gain: `2.650 JPY/kWh`

Interpretation:
- `13.127` should be treated as neutral fair value
- quoting directly at point value leaves the seller exposed to underpricing risk on average
- if the objective is `expected PnL >= 0`, the quote should be at least `14.068`
- if the objective is to increase no-loss probability, a higher no-loss quantile quote should be used

## 6. Business Decision Framework

The recommended business interpretation is:

1. `Forward` as market anchor
- use it to observe tradable market level
- use it as the most realistic external benchmark

2. `Fundamental` as structural explanation layer
- raw price explains marginal technology and SRMC structure
- calibrated price provides structural fair value
- it should not yet be used as the standalone commercial pricing engine

3. `SARIMAX` as the primary pricing engine
- use it for point forecast generation and quote ladder construction
- its current advantage is that the full chain from forecast to contract risk is already connected and explainable

4. Pricing should not stop at one number
- at minimum, pricing output should include:
  - fair value
  - risk-neutral quote
  - conservative no-loss quote

## 7. Release Positioning

The current repository is suitable for:
- management reporting
- internal knowledge-base publication
- reuse and extension in a shared company git repository
- future enhancement into a more granular pricing framework

It should be positioned as a complete and releaseable pricing framework, while still leaving room for future enhancement in:
- seasonal / regime-based residual segmentation
- historical snapshot fundamental construction
- pricing policy refinement
- expansion beyond Tokyo
