"""
SARIMAX-based probabilistic pricing for monthly PPA quotes.

This module converts a point forecast into an explainable pricing range by
using the empirical distribution of strict monthly OOS forecast errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SARIMAXRiskResult:
    monthly_quotes: pd.DataFrame
    contract_summary: pd.DataFrame
    error_history: pd.DataFrame
    error_metrics: Dict[str, float]
    simulated_actual_distribution: np.ndarray
    simulated_pnl_distribution: np.ndarray


class SARIMAXRiskPricingEngine:
    """Build monthly quote ranges and contract PnL risk from SARIMAX OOS errors."""

    def __init__(self, bootstrap_samples: int = 10000, random_seed: int = 42):
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed

    @staticmethod
    def _prepare_error_history(strict_oos_df: pd.DataFrame) -> pd.DataFrame:
        if strict_oos_df.empty:
            raise ValueError("strict_oos_df is empty")

        df = strict_oos_df.copy()
        df["month"] = pd.to_datetime(df["month"])
        required_cols = {"sarimax_strict_oos_price_jpy_kwh", "actual_spot_price_jpy_kwh"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"strict_oos_df missing required columns: {sorted(missing)}")

        df = df.dropna(subset=["sarimax_strict_oos_price_jpy_kwh", "actual_spot_price_jpy_kwh"]).copy()
        if df.empty:
            raise ValueError("strict_oos_df has no valid OOS rows after dropping NaNs")

        df["error_jpy_kwh"] = (
            df["sarimax_strict_oos_price_jpy_kwh"] - df["actual_spot_price_jpy_kwh"]
        )
        return df.sort_values("month").reset_index(drop=True)

    @staticmethod
    def _prepare_monthly_quotes(forecast_df: pd.DataFrame) -> pd.DataFrame:
        if forecast_df.empty:
            raise ValueError("forecast_df is empty")

        df = forecast_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["month"] = df["datetime"].dt.to_period("M").dt.to_timestamp()

        monthly = (
            df.groupby("month", as_index=False)
            .agg(
                quote_price_jpy_kwh=("forecast", "mean"),
                delivery_periods=("forecast", "size"),
                delivery_start=("datetime", "min"),
                delivery_end=("datetime", "max"),
            )
            .sort_values("month")
            .reset_index(drop=True)
        )
        monthly["weight"] = monthly["delivery_periods"] / monthly["delivery_periods"].sum()
        return monthly

    @staticmethod
    def _error_metrics(error_history: pd.DataFrame) -> Dict[str, float]:
        errors = error_history["error_jpy_kwh"].to_numpy(dtype=float)
        q05, q10, q25, q50, q75, q90, q95 = np.quantile(
            errors,
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        )
        left_tail = errors[errors <= q05]

        return {
            "sample_size": float(len(errors)),
            "mean_error": float(errors.mean()),
            "median_error": float(q50),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "std_error": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
            "probability_of_loss": float(np.mean(errors < 0)),
            "best_historical_gain": float(errors.max()),
            "worst_historical_loss": float(errors.min()),
            "q05_error": float(q05),
            "q10_error": float(q10),
            "q25_error": float(q25),
            "q50_error": float(q50),
            "q75_error": float(q75),
            "q90_error": float(q90),
            "q95_error": float(q95),
            "es95_error": float(left_tail.mean()) if len(left_tail) else float(q05),
        }

    def _build_monthly_bands(
        self,
        monthly_quotes: pd.DataFrame,
        error_metrics: Dict[str, float],
    ) -> pd.DataFrame:
        df = monthly_quotes.copy()

        # actual ~= quote - error ; report interval around quote using empirical OOS quantiles
        df["lower_90_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q90_error"]
        df["upper_90_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q10_error"]
        df["lower_95_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q95_error"]
        df["upper_95_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q05_error"]

        df["expected_pnl_jpy_kwh"] = error_metrics["mean_error"]
        df["probability_of_loss"] = error_metrics["probability_of_loss"]
        df["var_95_loss_jpy_kwh"] = max(0.0, -error_metrics["q05_error"])
        df["es_95_loss_jpy_kwh"] = max(0.0, -error_metrics["es95_error"])
        df["worst_historical_loss_jpy_kwh"] = error_metrics["worst_historical_loss"]
        df["best_historical_gain_jpy_kwh"] = error_metrics["best_historical_gain"]
        df["risk_neutral_quote_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["mean_error"]
        df["quote_50pct_no_loss_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q50_error"]
        df["quote_75pct_no_loss_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q25_error"]
        df["quote_90pct_no_loss_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q10_error"]
        df["quote_95pct_no_loss_jpy_kwh"] = df["quote_price_jpy_kwh"] - error_metrics["q05_error"]
        return df

    def _simulate_contract_distribution(
        self,
        monthly_quotes: pd.DataFrame,
        error_history: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        errors = error_history["error_jpy_kwh"].to_numpy(dtype=float)
        rng = np.random.default_rng(self.random_seed)
        metrics = self._error_metrics(error_history)

        weights = monthly_quotes["weight"].to_numpy(dtype=float)
        quote_prices = monthly_quotes["quote_price_jpy_kwh"].to_numpy(dtype=float)
        contract_quote = float(np.dot(weights, quote_prices))

        simulated_errors = rng.choice(
            errors,
            size=(self.bootstrap_samples, len(monthly_quotes)),
            replace=True,
        )
        contract_error = np.dot(simulated_errors, weights)
        simulated_actual = contract_quote - contract_error
        simulated_pnl = contract_quote - simulated_actual

        q05, q50, q95 = np.quantile(simulated_actual, [0.05, 0.50, 0.95])
        pnl_q05 = np.quantile(simulated_pnl, 0.05)
        pnl_left_tail = simulated_pnl[simulated_pnl <= pnl_q05]

        summary = pd.DataFrame(
            [
                {
                    "contract_start": monthly_quotes["month"].min(),
                    "contract_end": monthly_quotes["month"].max(),
                    "delivery_months": int(len(monthly_quotes)),
                    "quote_price_p50_jpy_kwh": contract_quote,
                    "risk_neutral_quote_jpy_kwh": float(contract_quote - metrics["mean_error"]),
                    "quote_50pct_no_loss_jpy_kwh": float(contract_quote - metrics["q50_error"]),
                    "quote_75pct_no_loss_jpy_kwh": float(contract_quote - metrics["q25_error"]),
                    "quote_90pct_no_loss_jpy_kwh": float(contract_quote - metrics["q10_error"]),
                    "quote_95pct_no_loss_jpy_kwh": float(contract_quote - metrics["q05_error"]),
                    "simulated_actual_p5_jpy_kwh": float(q05),
                    "simulated_actual_p50_jpy_kwh": float(q50),
                    "simulated_actual_p95_jpy_kwh": float(q95),
                    "expected_pnl_jpy_kwh": float(simulated_pnl.mean()),
                    "probability_of_loss": float(np.mean(simulated_pnl < 0)),
                    "var_95_loss_jpy_kwh": float(max(0.0, -pnl_q05)),
                    "es_95_loss_jpy_kwh": float(max(0.0, -pnl_left_tail.mean())) if len(pnl_left_tail) else float(max(0.0, -pnl_q05)),
                    "worst_simulated_loss_jpy_kwh": float(simulated_pnl.min()),
                    "best_simulated_gain_jpy_kwh": float(simulated_pnl.max()),
                }
            ]
        )
        return summary, simulated_actual, simulated_pnl

    def build(
        self,
        forecast_df: pd.DataFrame,
        strict_oos_df: pd.DataFrame,
    ) -> SARIMAXRiskResult:
        error_history = self._prepare_error_history(strict_oos_df)
        monthly_quotes = self._prepare_monthly_quotes(forecast_df)
        metrics = self._error_metrics(error_history)
        monthly_output = self._build_monthly_bands(monthly_quotes, metrics)
        contract_summary, simulated_actual, simulated_pnl = self._simulate_contract_distribution(monthly_quotes, error_history)
        return SARIMAXRiskResult(
            monthly_quotes=monthly_output,
            contract_summary=contract_summary,
            error_history=error_history,
            error_metrics=metrics,
            simulated_actual_distribution=simulated_actual,
            simulated_pnl_distribution=simulated_pnl,
        )

    @staticmethod
    def write_text_report(
        result: SARIMAXRiskResult,
        output_path: str | Path,
        region: str,
    ) -> Path:
        metrics = result.error_metrics
        contract = result.contract_summary.iloc[0]

        lines = [
            "=" * 80,
            f"SARIMAX RISK PRICING REPORT ({region.upper()})",
            "=" * 80,
            "",
            "Methodology:",
            "- Point quote uses the SARIMAX monthly forecast directly.",
            "- Range metrics use the empirical distribution of strict monthly OOS errors.",
            "- Error is defined as forecast minus realized monthly actual spot.",
            "- Contract risk is estimated by bootstrap resampling monthly OOS errors.",
            "",
            "Historical OOS Error Statistics:",
            f"- Sample size: {int(metrics['sample_size'])} months",
            f"- Mean error: {metrics['mean_error']:.3f} JPY/kWh",
            f"- MAE: {metrics['mae']:.3f} JPY/kWh",
            f"- RMSE: {metrics['rmse']:.3f} JPY/kWh",
            f"- Probability of loss: {metrics['probability_of_loss']:.1%}",
            f"- Worst historical loss: {metrics['worst_historical_loss']:.3f} JPY/kWh",
            f"- Best historical gain: {metrics['best_historical_gain']:.3f} JPY/kWh",
            "",
            "Empirical Error Quantiles:",
            f"- P05 error: {metrics['q05_error']:.3f}",
            f"- P10 error: {metrics['q10_error']:.3f}",
            f"- P25 error: {metrics['q25_error']:.3f}",
            f"- P50 error: {metrics['q50_error']:.3f}",
            f"- P75 error: {metrics['q75_error']:.3f}",
            f"- P90 error: {metrics['q90_error']:.3f}",
            f"- P95 error: {metrics['q95_error']:.3f}",
            "",
            "Risk-Adjusted Quote Ladder:",
            f"- Point quote / fair value: {contract['quote_price_p50_jpy_kwh']:.3f} JPY/kWh",
            f"- Risk-neutral quote (Expected PnL >= 0): {contract['risk_neutral_quote_jpy_kwh']:.3f} JPY/kWh",
            f"- 50% no-loss quote: {contract['quote_50pct_no_loss_jpy_kwh']:.3f} JPY/kWh",
            f"- 75% no-loss quote: {contract['quote_75pct_no_loss_jpy_kwh']:.3f} JPY/kWh",
            f"- 90% no-loss quote: {contract['quote_90pct_no_loss_jpy_kwh']:.3f} JPY/kWh",
            f"- 95% no-loss quote: {contract['quote_95pct_no_loss_jpy_kwh']:.3f} JPY/kWh",
            "",
            "Contract-Level Risk Summary:",
            f"- Delivery window: {pd.Timestamp(contract['contract_start']).strftime('%Y-%m')} to "
            f"{pd.Timestamp(contract['contract_end']).strftime('%Y-%m')}",
            f"- Point quote (P50): {contract['quote_price_p50_jpy_kwh']:.3f} JPY/kWh",
            f"- Simulated actual P5/P50/P95: "
            f"{contract['simulated_actual_p5_jpy_kwh']:.3f} / "
            f"{contract['simulated_actual_p50_jpy_kwh']:.3f} / "
            f"{contract['simulated_actual_p95_jpy_kwh']:.3f} JPY/kWh",
            f"- Expected PnL: {contract['expected_pnl_jpy_kwh']:.3f} JPY/kWh",
            f"- Probability of loss: {contract['probability_of_loss']:.1%}",
            f"- 95% VaR loss: {contract['var_95_loss_jpy_kwh']:.3f} JPY/kWh",
            f"- 95% Expected Shortfall loss: {contract['es_95_loss_jpy_kwh']:.3f} JPY/kWh",
            f"- Worst simulated loss: {contract['worst_simulated_loss_jpy_kwh']:.3f} JPY/kWh",
            f"- Best simulated gain: {contract['best_simulated_gain_jpy_kwh']:.3f} JPY/kWh",
            "=" * 80,
        ]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        return output

    @staticmethod
    def plot_monthly_quote_bands(
        result: SARIMAXRiskResult,
        output_path: str | Path,
        title: str,
    ) -> Path:
        df = result.monthly_quotes.copy()
        df["month"] = pd.to_datetime(df["month"])

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(df["month"], df["quote_price_jpy_kwh"], color="#103d60", linewidth=2.6, marker="o", label="SARIMAX Quote")
        ax.fill_between(df["month"], df["lower_95_jpy_kwh"], df["upper_95_jpy_kwh"], color="#bfdbfe", alpha=0.35, label="95% empirical band")
        ax.fill_between(df["month"], df["lower_90_jpy_kwh"], df["upper_90_jpy_kwh"], color="#60a5fa", alpha=0.35, label="90% empirical band")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel("Price (JPY/kWh)")
        ax.set_xlabel("Delivery Month")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", framealpha=0.95)
        fig.autofmt_xdate()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output

    @staticmethod
    def plot_contract_pnl_distribution(
        result: SARIMAXRiskResult,
        output_path: str | Path,
        title: str,
    ) -> Path:
        contract = result.contract_summary.iloc[0]
        simulated_pnl = result.simulated_pnl_distribution
        mean_pnl = float(simulated_pnl.mean())
        var95 = float(np.quantile(simulated_pnl, 0.05))

        fig, ax = plt.subplots(figsize=(13, 7))
        ax.hist(
            simulated_pnl,
            bins=40,
            color="#93c5fd",
            edgecolor="#1d4ed8",
            alpha=0.8,
        )
        ax.axvline(mean_pnl, color="#0f766e", linewidth=2.2, label=f"Mean error/PnL {mean_pnl:.2f}")
        ax.axvline(var95, color="#b91c1c", linewidth=2.2, linestyle="--", label=f"95% VaR threshold {var95:.2f}")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Contract seller PnL (JPY/kWh)")
        ax.set_ylabel("Frequency")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(loc="upper left", framealpha=0.95)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output
