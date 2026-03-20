#!/usr/bin/env python3
"""
PPA Pricing System - Unified Command Line Interface

Main entry point for all PPA pricing operations.

Usage:
    python -m src.cli.main pricing --region tokyo --volume 2000000
    python -m src.cli.main forecast --days 90
    python -m src.cli.main analysis
"""

import argparse
import logging
import sys
from pathlib import Path
# train command is integrated into pricing command
# from .commands.train import run_train_command

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import JapanRegion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='ppa-pricing',
        description='PPA Pricing System for Japanese Electricity Market',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ppa-pricing pricing --region tokyo --volume 2000000
  ppa-pricing forecast --days 90 --models xgboost prophet
  ppa-pricing analysis --region kansai
  ppa-pricing train --region tokyo --models sarimax prophet xgboost
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='{pricing,forecast,analysis,train}'
    )
    
    # Pricing command
    _add_pricing_parser(subparsers)
    
    # Forecast command
    _add_forecast_parser(subparsers)
    
    # Analysis command
    _add_analysis_parser(subparsers)

    
    return parser


def _add_pricing_parser(subparsers):
    """Add pricing subcommand parser."""
    parser = subparsers.add_parser(
        'pricing',
        help='Calculate PPA prices using various models'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='tokyo',
        choices=[r.value for r in JapanRegion],
        help='Target Japanese electricity market region'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['historical', 'sarimax'],
        choices=['xgboost', 'prophet', 'sarimax', 'historical'],
        help='Models to train and compare'
    )
    
    parser.add_argument(
        '--train-years',
        type=int,
        default=3,
        help='Number of years in training window'
    )
    
    parser.add_argument(
        '--forecast-months',
        type=int,
        default=3,
        help='Number of months to forecast from latest data (default: 3)'
    )
    
    parser.add_argument(
        '--volume',
        type=float,
        default=2.0,
        help='Total contracted volume in MWh'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation and only run pricing'
    )
    
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force full optimization even if cached parameters exist'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualization chart (default: True)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_false',
        dest='visualize',
        help='Disable visualization chart generation'
    )
    
    parser.add_argument(
        '--external-forecast-file',
        type=str,
        help='Path to pre-computed forecast file (enables lazy execution)'
    )


def _add_forecast_parser(subparsers):
    """Add forecast subcommand parser."""
    parser = subparsers.add_parser(
        'forecast',
        help='Run medium-term price forecasting and evaluation'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='tokyo',
        choices=[r.value for r in JapanRegion],
        help='Target region for forecasting'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='Training window in years'
    )
    
    parser.add_argument(
        '--forecast-days',
        type=int,
        default=90,
        help='Forecast horizon in days'
    )
    
    parser.add_argument(
        '--step-days',
        type=int,
        default=30,
        help='Backtest step size in days'
    )
    
    parser.add_argument(
        '--skip-prophet',
        action='store_true',
        help='Skip Prophet model evaluation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for forecast results'
    )


def _add_analysis_parser(subparsers):
    """Add analysis subcommand parser."""
    parser = subparsers.add_parser(
        'analysis',
        help='Run data analysis and market studies'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='tokyo',
        choices=[r.value for r in JapanRegion],
        help='Target region for analysis'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for analysis results'
    )

    parser.add_argument(
        '--standardized-data-dir',
        type=str,
        default='data/processed/fundamental',
        help='Directory for standardized fundamental input tables'
    )




def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Handle missing command
    if not args.command:
        parser.print_help()
        return 1
    
    logger.info(f"Starting PPA Pricing System - Command: {args.command}")
    
    try:
        if args.command == 'pricing':
            from .commands.pricing import run_pricing_command
            return run_pricing_command(args)
        elif args.command == 'forecast':
            from .commands.forecast import run_forecast_command
            return run_forecast_command(args)
        elif args.command == 'analysis':
            from .commands.analysis import run_analysis_command
            return run_analysis_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
