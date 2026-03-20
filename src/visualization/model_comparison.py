"""
Model Comparison Visualization Module.

Provides functions to generate comparison charts between actual prices
and predictions from multiple models.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def plot_monthly_comparison(
    monthly_df: pd.DataFrame,
    save_path: str,
    title: str = "Historical vs Model Predictions (Monthly Average)"
) -> str:
    """
    Generate line chart comparing actual prices vs model predictions.
    
    Chart elements:
    - Black solid line: Actual prices
    - Blue solid + shaded area: SARIMAX prediction + 90% CI
    - Green dashed + shaded area: Prophet prediction + 90% CI
    - Red dotted + shaded area: XGBoost prediction + 90% CI
    
    Args:
        monthly_df: DataFrame with columns:
            - year_month: Period or datetime index
            - actual_mean: Monthly average actual prices
            - {model}_mean, {model}_lower, {model}_upper: Per-model predictions
        save_path: Path to save the output PNG file
        title: Chart title
        
    Returns:
        Path to the saved chart file
    """
    logger.info(f"Generating monthly comparison chart: {title}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Convert year_month to datetime for plotting
    if 'year_month' in monthly_df.columns:
        if hasattr(monthly_df['year_month'].iloc[0], 'to_timestamp'):
            months = monthly_df['year_month'].apply(lambda x: x.to_timestamp())
        else:
            months = pd.to_datetime(monthly_df['year_month'].astype(str))
    else:
        months = monthly_df.index
    
    # Sort by date
    sort_idx = months.argsort()
    months = months.iloc[sort_idx]
    monthly_df = monthly_df.iloc[sort_idx]
    
    # Model configurations
    model_configs = {
        'sarimax': {
            'color': '#1f77b4',  # Blue
            'linestyle': '-',
            'label': 'SARIMAX',
            'alpha_fill': 0.2
        },
        'prophet': {
            'color': '#2ca02c',  # Green
            'linestyle': '--',
            'label': 'Prophet',
            'alpha_fill': 0.2
        },
        'xgboost': {
            'color': '#d62728',  # Red
            'linestyle': ':',
            'label': 'XGBoost',
            'alpha_fill': 0.2
        }
    }
    
    # Plot actual values
    if 'actual_mean' in monthly_df.columns:
        actual_values = monthly_df['actual_mean'].values
        ax.plot(months, actual_values, 
                color='black', linewidth=2.5, 
                label='Actual', zorder=10)
        logger.info(f"Plotted actual values: {len(actual_values)} points")
    
    # Plot each model's predictions
    for model_key, config in model_configs.items():
        mean_col = f'{model_key}_mean'
        lower_col = f'{model_key}_lower'
        upper_col = f'{model_key}_upper'
        
        if mean_col in monthly_df.columns:
            pred_values = monthly_df[mean_col].values
            
            # Plot prediction line
            ax.plot(months, pred_values,
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=1.5,
                    label=config['label'],
                    alpha=0.8)
            
            # Plot confidence interval if available
            if lower_col in monthly_df.columns and upper_col in monthly_df.columns:
                lower_values = monthly_df[lower_col].values
                upper_values = monthly_df[upper_col].values
                
                # Only fill where we have valid bounds
                valid_mask = ~(np.isnan(lower_values) | np.isnan(upper_values))
                if valid_mask.any():
                    ax.fill_between(
                        months[valid_mask],
                        lower_values[valid_mask],
                        upper_values[valid_mask],
                        color=config['color'],
                        alpha=config['alpha_fill'],
                        label=f'{config["label"]} 90% CI'
                    )
            
            logger.info(f"Plotted {config['label']}: {len(pred_values)} points")
    
    # Configure axes
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (JPY/kWh)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Chart saved to: {save_path}")
    
    return str(save_path)


def plot_model_errors(
    monthly_df: pd.DataFrame,
    save_path: str,
    title: str = "Model Prediction Errors by Month"
) -> str:
    """
    Generate bar chart showing prediction errors for each model by month.
    
    Args:
        monthly_df: DataFrame with actual and predicted values
        save_path: Path to save the output PNG file
        title: Chart title
        
    Returns:
        Path to the saved chart file
    """
    logger.info(f"Generating error comparison chart: {title}")
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Convert year_month to datetime
    if 'year_month' in monthly_df.columns:
        if hasattr(monthly_df['year_month'].iloc[0], 'to_timestamp'):
            months = monthly_df['year_month'].apply(lambda x: x.to_timestamp())
        else:
            months = pd.to_datetime(monthly_df['year_month'].astype(str))
    else:
        months = monthly_df.index
    
    # Calculate errors
    model_configs = {
        'sarimax': {'color': '#1f77b4', 'label': 'SARIMAX'},
        'prophet': {'color': '#2ca02c', 'label': 'Prophet'},
        'xgboost': {'color': '#d62728', 'label': 'XGBoost'}
    }
    
    x = np.arange(len(months))
    width = 0.25
    
    for i, (model_key, config) in enumerate(model_configs.items()):
        mean_col = f'{model_key}_mean'
        if mean_col in monthly_df.columns and 'actual_mean' in monthly_df.columns:
            errors = monthly_df[mean_col].values - monthly_df['actual_mean'].values
            ax.bar(x + i * width, errors, width, 
                   label=config['label'], color=config['color'], alpha=0.7)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Prediction Error (JPY/kWh)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.strftime('%Y-%m') for m in months], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Error chart saved to: {save_path}")
    
    return str(save_path)
