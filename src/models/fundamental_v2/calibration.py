"""
Calibration layer for monthly fundamental prices.

This keeps the raw merit-order logic intact and fits a transparent premium model
from structural state variables to observed monthly spot prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    calibrated_prices: pd.DataFrame
    metrics: Dict[str, float]
    coefficients: pd.DataFrame


@dataclass
class RollingCalibrationResult:
    oos_predictions: pd.DataFrame
    metrics: Dict[str, float]


class FundamentalPriceCalibrator:
    """Linear ridge-style premium calibrator for raw monthly fundamental prices."""

    FEATURE_COLUMNS = [
        "intercept",
        "demand_gw",
        "tightness",
        "gas_dummy",
        "coal_dummy",
        "oil_dummy",
        "other_dummy",
    ]

    def __init__(self, regularization: float = 1.0):
        self.regularization = regularization

    def calibrate(
        self,
        monthly_prices: pd.DataFrame,
        actual_monthly: pd.DataFrame,
    ) -> CalibrationResult:
        if monthly_prices.empty:
            raise ValueError("monthly_prices is empty")

        merged = monthly_prices.merge(actual_monthly, on="month", how="left")
        feature_df = self._build_features(merged)

        train_mask = feature_df["actual_spot_price_jpy_kwh"].notna()
        if train_mask.sum() < 6:
            feature_df["calibrated_price_jpy_kwh"] = feature_df["raw_fundamental_price_jpy_kwh"]
            feature_df["predicted_premium_jpy_kwh"] = 0.0
            coefficients = pd.DataFrame(
                [{"feature": feature, "coefficient": 0.0} for feature in self.FEATURE_COLUMNS]
            )
            return CalibrationResult(
                calibrated_prices=feature_df,
                metrics={},
                coefficients=coefficients,
            )

        X = feature_df.loc[train_mask, self.FEATURE_COLUMNS].to_numpy(dtype=float)
        y = feature_df.loc[train_mask, "premium_jpy_kwh"].to_numpy(dtype=float)

        ridge = self.regularization * np.eye(X.shape[1])
        ridge[0, 0] = 0.0
        beta = np.linalg.solve(X.T @ X + ridge, X.T @ y)

        feature_df["predicted_premium_jpy_kwh"] = feature_df[self.FEATURE_COLUMNS].to_numpy(dtype=float) @ beta
        feature_df["calibrated_price_jpy_kwh"] = np.maximum(
            feature_df["raw_fundamental_price_jpy_kwh"] + feature_df["predicted_premium_jpy_kwh"],
            0.01,
        )

        in_sample_pred = feature_df.loc[train_mask, "calibrated_price_jpy_kwh"].to_numpy(dtype=float)
        raw_pred = feature_df.loc[train_mask, "raw_fundamental_price_jpy_kwh"].to_numpy(dtype=float)
        actual = feature_df.loc[train_mask, "actual_spot_price_jpy_kwh"].to_numpy(dtype=float)

        metrics = {
            "training_months": float(train_mask.sum()),
            "raw_mae": float(np.mean(np.abs(actual - raw_pred))),
            "calibrated_mae": float(np.mean(np.abs(actual - in_sample_pred))),
            "raw_rmse": float(np.sqrt(np.mean((actual - raw_pred) ** 2))),
            "calibrated_rmse": float(np.sqrt(np.mean((actual - in_sample_pred) ** 2))),
        }

        coefficients = pd.DataFrame(
            {
                "feature": self.FEATURE_COLUMNS,
                "coefficient": beta,
            }
        )

        return CalibrationResult(
            calibrated_prices=feature_df,
            metrics=metrics,
            coefficients=coefficients,
        )

    def rolling_oos(
        self,
        monthly_prices: pd.DataFrame,
        actual_monthly: pd.DataFrame,
        min_train_months: int = 12,
    ) -> RollingCalibrationResult:
        """Run expanding-window OOS premium calibration month by month."""
        if monthly_prices.empty:
            raise ValueError("monthly_prices is empty")

        merged = monthly_prices.merge(actual_monthly, on="month", how="left")
        feature_df = self._build_features(merged)
        hist = (
            feature_df[feature_df["actual_spot_price_jpy_kwh"].notna()]
            .sort_values("month")
            .reset_index(drop=True)
        )
        if len(hist) <= min_train_months:
            return RollingCalibrationResult(oos_predictions=pd.DataFrame(), metrics={})

        rows = []
        for idx in range(min_train_months, len(hist)):
            train = hist.iloc[:idx].copy()
            test = hist.iloc[[idx]].copy()

            X_train = train[self.FEATURE_COLUMNS].to_numpy(dtype=float)
            y_train = train["premium_jpy_kwh"].to_numpy(dtype=float)
            ridge = self.regularization * np.eye(X_train.shape[1])
            ridge[0, 0] = 0.0
            beta = np.linalg.solve(X_train.T @ X_train + ridge, X_train.T @ y_train)

            X_test = test[self.FEATURE_COLUMNS].to_numpy(dtype=float)
            predicted_premium = float((X_test @ beta)[0])
            raw_price = float(test["raw_fundamental_price_jpy_kwh"].iloc[0])
            actual_price = float(test["actual_spot_price_jpy_kwh"].iloc[0])
            rolling_price = max(raw_price + predicted_premium, 0.01)

            rows.append(
                {
                    "month": test["month"].iloc[0],
                    "raw_fundamental_price_jpy_kwh": raw_price,
                    "predicted_premium_jpy_kwh": predicted_premium,
                    "premium_actual_jpy_kwh": float(test["premium_jpy_kwh"].iloc[0]),
                    "rolling_calibrated_price_jpy_kwh": rolling_price,
                    "actual_spot_price_jpy_kwh": actual_price,
                    "fundamental_error_jpy_kwh": rolling_price - actual_price,
                    "train_window_start": train["month"].min(),
                    "train_window_end": train["month"].max(),
                    "train_size": len(train),
                }
            )

        oos_predictions = pd.DataFrame(rows)
        if oos_predictions.empty:
            return RollingCalibrationResult(oos_predictions=oos_predictions, metrics={})

        raw_err = oos_predictions["raw_fundamental_price_jpy_kwh"] - oos_predictions["actual_spot_price_jpy_kwh"]
        calibrated_err = oos_predictions["fundamental_error_jpy_kwh"]
        metrics = {
            "oos_months": float(len(oos_predictions)),
            "min_train_months": float(min_train_months),
            "raw_oos_mae": float(np.mean(np.abs(raw_err))),
            "rolling_calibrated_oos_mae": float(np.mean(np.abs(calibrated_err))),
            "raw_oos_rmse": float(np.sqrt(np.mean(raw_err ** 2))),
            "rolling_calibrated_oos_rmse": float(np.sqrt(np.mean(calibrated_err ** 2))),
        }
        return RollingCalibrationResult(oos_predictions=oos_predictions, metrics=metrics)

    def _build_features(self, merged: pd.DataFrame) -> pd.DataFrame:
        df = merged.copy()
        df["raw_fundamental_price_jpy_kwh"] = df["clearing_price_jpy_kwh"]
        df["intercept"] = 1.0
        df["demand_gw"] = df["demand_mw"].astype(float) / 1000.0
        df["tightness"] = df["demand_mw"].astype(float) / df["available_capacity_mw"].replace(0, np.nan).astype(float)
        df["tightness"] = df["tightness"].fillna(0.0)
        df["premium_jpy_kwh"] = df["actual_spot_price_jpy_kwh"] - df["raw_fundamental_price_jpy_kwh"]

        marginal = df["marginal_technology"].fillna("other").str.lower()
        df["gas_dummy"] = (marginal == "gas").astype(float)
        df["coal_dummy"] = (marginal == "coal").astype(float)
        df["oil_dummy"] = (marginal == "oil").astype(float)
        df["other_dummy"] = (~marginal.isin(["gas", "coal", "oil"])).astype(float)

        return df
