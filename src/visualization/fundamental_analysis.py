"""
Visualization helpers for the fundamental analysis workflow.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_fundamental_comparison(
    comparison_df: pd.DataFrame,
    save_path: str,
    title: str = "Fundamental vs Actual vs Pricing",
) -> str:
    """Plot monthly fundamental, actual spot and pricing forecast levels."""
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")

    df = comparison_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        df["month"],
        df["fundamental_price_jpy_kwh"],
        color="#103d60",
        linewidth=2.4,
        label="Calibrated Fundamental",
    )

    if "raw_fundamental_price_jpy_kwh" in df.columns and df["raw_fundamental_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["raw_fundamental_price_jpy_kwh"],
            color="#6b7280",
            linewidth=1.5,
            linestyle=":",
            label="Raw Fundamental",
        )

    if "actual_spot_price_jpy_kwh" in df.columns and df["actual_spot_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["actual_spot_price_jpy_kwh"],
            color="#d97706",
            linewidth=2.0,
            linestyle="--",
            label="Actual Spot",
        )

    if "pricing_price_jpy_kwh" in df.columns and df["pricing_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["pricing_price_jpy_kwh"],
            color="#2f855a",
            linewidth=2.0,
            linestyle="-.",
            label="Pricing Forecast",
        )

        overlap = df.dropna(subset=["fundamental_price_jpy_kwh", "pricing_price_jpy_kwh"])
        if not overlap.empty:
            ax.fill_between(
                overlap["month"],
                overlap["fundamental_price_jpy_kwh"],
                overlap["pricing_price_jpy_kwh"],
                color="#74c69d",
                alpha=0.18,
                label="Forecast Spread",
            )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Price (JPY/kWh)", fontsize=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output)


def plot_fundamental_drivers(
    monthly_prices: pd.DataFrame,
    save_path: str,
    title: str = "Fundamental Demand and Clearing Price",
) -> str:
    """Plot demand and clearing price together to show the fundamental driver."""
    if monthly_prices.empty:
        raise ValueError("monthly_prices is empty")

    df = monthly_prices.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()

    ax1.bar(
        df["month"],
        df["demand_mw"],
        width=20,
        color="#cfe8ff",
        edgecolor="#7aa6d8",
        label="Demand (MW)",
    )

    ax2.plot(
        df["month"],
        df["clearing_price_jpy_kwh"],
        color="#8b1e3f",
        linewidth=1.8,
        linestyle="--",
        marker="o",
        markersize=3.5,
        label="Raw Clearing Price",
    )

    if "calibrated_price_jpy_kwh" in df.columns and df["calibrated_price_jpy_kwh"].notna().any():
        ax2.plot(
            df["month"],
            df["calibrated_price_jpy_kwh"],
            color="#8b1e3f",
            linewidth=2.6,
            label="Calibrated Price",
        )

    ax1.set_title(title, fontsize=16, fontweight="bold")
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Demand (MW)", fontsize=12, color="#355c7d")
    ax2.set_ylabel("Price (JPY/kWh)", fontsize=12, color="#8b1e3f")
    ax1.grid(True, axis="y", alpha=0.2)

    if "reserve_margin" in df.columns:
        low_margin = df.nsmallest(min(3, len(df)), "reserve_margin")
        for _, row in low_margin.iterrows():
            ax2.annotate(
                row["month"].strftime("%Y-%m"),
                (row["month"], row["clearing_price_jpy_kwh"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color="#8b1e3f",
            )

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", framealpha=0.95)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output)


def plot_calibration_bridge(
    comparison_df: pd.DataFrame,
    save_path: str,
    title: str = "Raw Fundamental to Calibrated Price",
) -> str:
    """Plot raw fundamental, calibrated fundamental, and actual spot together."""
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")

    df = comparison_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    fig, ax = plt.subplots(figsize=(17, 7.5))

    if "raw_fundamental_price_jpy_kwh" in df.columns and df["raw_fundamental_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["raw_fundamental_price_jpy_kwh"],
            color="#6b7280",
            linewidth=2.0,
            linestyle=":",
            marker="o",
            markersize=3,
            label="Raw Fundamental",
        )

    if "fundamental_price_jpy_kwh" in df.columns and df["fundamental_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["fundamental_price_jpy_kwh"],
            color="#103d60",
            linewidth=2.8,
            marker="o",
            markersize=3,
            label="Calibrated Fundamental",
        )

    if "actual_spot_price_jpy_kwh" in df.columns and df["actual_spot_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["actual_spot_price_jpy_kwh"],
            color="#d97706",
            linewidth=2.2,
            linestyle="--",
            label="Actual Spot",
        )

        overlap = df.dropna(subset=["raw_fundamental_price_jpy_kwh", "fundamental_price_jpy_kwh", "actual_spot_price_jpy_kwh"])
        if not overlap.empty:
            ax.fill_between(
                overlap["month"],
                overlap["raw_fundamental_price_jpy_kwh"],
                overlap["fundamental_price_jpy_kwh"],
                color="#93c5fd",
                alpha=0.18,
                label="Calibration Lift",
            )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Price (JPY/kWh)", fontsize=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)

    month_count = max(len(df), 1)
    interval = 1 if month_count <= 12 else 2 if month_count <= 24 else 3
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output)


def plot_oos_model_vs_fundamental(
    comparison_df: pd.DataFrame,
    save_path: str,
    title: str = "SARIMAX OOS vs Fundamental vs Actual",
) -> str:
    """Plot actual spot, SARIMAX OOS, raw fundamental, and calibrated fundamental over the same window."""
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")

    df = comparison_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    fig, (ax, ax_spread) = plt.subplots(
        2,
        1,
        figsize=(18, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    if "actual_spot_price_jpy_kwh" in df.columns and df["actual_spot_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["actual_spot_price_jpy_kwh"],
            color="#d97706",
            linewidth=2.5,
            label="Actual Spot",
        )

    if "sarimax_oos_price_jpy_kwh" in df.columns and df["sarimax_oos_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["sarimax_oos_price_jpy_kwh"],
            color="#1f77b4",
            linewidth=2.2,
            linestyle="--",
            label="SARIMAX OOS",
        )

    if "raw_fundamental_price_jpy_kwh" in df.columns and df["raw_fundamental_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["raw_fundamental_price_jpy_kwh"],
            color="#6b7280",
            linewidth=1.8,
            linestyle=":",
            label="Raw Fundamental",
        )

    if "calibrated_price_jpy_kwh" in df.columns and df["calibrated_price_jpy_kwh"].notna().any():
        ax.plot(
            df["month"],
            df["calibrated_price_jpy_kwh"],
            color="#103d60",
            linewidth=2.6,
            label="Calibrated Fundamental",
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Price (JPY/kWh)", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)

    if {"sarimax_oos_price_jpy_kwh", "actual_spot_price_jpy_kwh"}.issubset(df.columns):
        model_spread = df["sarimax_oos_price_jpy_kwh"] - df["actual_spot_price_jpy_kwh"]
        ax_spread.bar(
            df["month"],
            model_spread,
            width=20,
            color="#93c5fd",
            alpha=0.7,
            label="SARIMAX - Actual",
        )

    if {"calibrated_price_jpy_kwh", "actual_spot_price_jpy_kwh"}.issubset(df.columns):
        fundamental_spread = df["calibrated_price_jpy_kwh"] - df["actual_spot_price_jpy_kwh"]
        ax_spread.plot(
            df["month"],
            fundamental_spread,
            color="#8b1e3f",
            linewidth=2.0,
            marker="o",
            markersize=3,
            label="Fundamental - Actual",
        )

    ax_spread.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax_spread.set_ylabel("Error / Spread", fontsize=11)
    ax_spread.grid(True, axis="y", alpha=0.2)
    ax_spread.legend(loc="upper left", framealpha=0.95)

    month_count = max(len(df), 1)
    interval = 1 if month_count <= 12 else 2 if month_count <= 24 else 3
    ax_spread.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_spread.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax_spread.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output)


def plot_strict_sarimax_forward_actual(
    comparison_df: pd.DataFrame,
    save_path: str,
    title: str = "Strict Monthly OOS: SARIMAX vs Market Forward vs Actual",
) -> str:
    """Plot strict monthly OOS SARIMAX, market forward, and actual spot with error panel."""
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")

    df = comparison_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(17, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.plot(
        df["month"],
        df["actual_spot_price_jpy_kwh"],
        color="#d97706",
        linewidth=2.8,
        marker="o",
        markersize=4,
        label="Actual Spot",
    )
    ax.plot(
        df["month"],
        df["market_forward_price_jpy_kwh"],
        color="#2f855a",
        linewidth=2.5,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Market Forward",
    )
    ax.plot(
        df["month"],
        df["sarimax_strict_oos_price_jpy_kwh"],
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="SARIMAX Strict OOS",
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Price (JPY/kWh)", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)

    forward_err = df["market_forward_price_jpy_kwh"] - df["actual_spot_price_jpy_kwh"]
    sarimax_err = df["sarimax_strict_oos_price_jpy_kwh"] - df["actual_spot_price_jpy_kwh"]

    ax_err.plot(
        df["month"],
        forward_err,
        color="#2f855a",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Forward - Actual",
    )
    ax_err.plot(
        df["month"],
        sarimax_err,
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="SARIMAX - Actual",
    )
    ax_err.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax_err.set_ylabel("Error", fontsize=11)
    ax_err.set_xlabel("Delivery Month", fontsize=12)
    ax_err.grid(True, axis="y", alpha=0.2)
    ax_err.legend(loc="upper left", framealpha=0.95)

    ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_err.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_err.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output)
