"""
Main entry point for the backtesting environment.
Handles CLI arguments and orchestrates the backtesting process.
"""

import argparse
import sys
from pathlib import Path
import json
import logging

# Local imports
import config
import data_loader
import utils
from engine import BacktestEngine
from metrics import PerformanceMetrics
from strategies import get_strategy

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Modular Crypto Backtesting Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol BTCUSDT --strategy ema_crossover
  python main.py --symbol ETHUSDT --strategy ema_crossover --params '{"fast_period": 10, "slow_period": 30}'
  python main.py --symbol BTCUSDT --timeframe 1d --balance 50000 --no-plot
        """
    )
    
    # Data arguments
    parser.add_argument('--symbol', type=str, default=config.DEFAULT_SYMBOL,
                       help=f'Trading symbol (default: {config.DEFAULT_SYMBOL})')
    parser.add_argument('--timeframe', type=str, default=config.DEFAULT_TIMEFRAME,
                       help=f'Data timeframe (default: {config.DEFAULT_TIMEFRAME})')
    
    # Strategy arguments
    parser.add_argument('--strategy', type=str, default=config.DEFAULT_STRATEGY,
                       help=f'Strategy name (default: {config.DEFAULT_STRATEGY})')
    parser.add_argument('--params', type=str, default='{}',
                       help='Strategy parameters as JSON string (default: {})')
    
    # Backtest configuration
    parser.add_argument('--balance', type=float, default=config.DEFAULT_INITIAL_BALANCE,
                       help=f'Initial balance (default: {config.DEFAULT_INITIAL_BALANCE})')
    parser.add_argument('--fee-rate', type=float, default=config.DEFAULT_FEE_RATE,
                       help=f'Trading fee rate (default: {config.DEFAULT_FEE_RATE})')
    parser.add_argument('--slippage', type=float, default=config.DEFAULT_SLIPPAGE,
                       help=f'Slippage rate (default: {config.DEFAULT_SLIPPAGE})')
    
    # Output options
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable equity curve plotting')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving results to files')
    parser.add_argument('--sample-data', action='store_true',
                       help='Use generated sample data instead of loading from file')
    
    # Logging
    parser.add_argument('--log-level', type=str, default=config.LOG_LEVEL,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help=f'Logging level (default: {config.LOG_LEVEL})')
    
    return parser.parse_args()


def load_strategy_parameters(params_json: str, strategy_name: str) -> dict:
    """
    Load and validate strategy parameters from JSON string.
    
    Args:
        params_json: JSON string with parameters
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary with validated parameters
    """
    try:
        params = json.loads(params_json)
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a JSON object")
        
        # Validate parameters
        validated_params = utils.validate_parameters(params, strategy_name)
        
        logger.info(f"Strategy parameters loaded: {validated_params}")
        return validated_params
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in parameters: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        sys.exit(1)


def main():
    """
    Main function that orchestrates the backtesting process.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    utils.setup_logging(args.log_level)
    
    # Ensure required directories exist
    config.ensure_directories()
    
    logger.info("Starting crypto backtesting environment...")
    logger.info(f"Symbol: {args.symbol}, Strategy: {args.strategy}, Timeframe: {args.timeframe}")
    
    try:
        # Load data
        if args.sample_data:
            logger.info("Using generated sample data")
            data = utils.create_sample_data(args.symbol)
        else:
            logger.info(f"Loading data for {args.symbol} ({args.timeframe})")
            data = data_loader.get_data(args.symbol, args.timeframe)
        
        # Load strategy parameters
        strategy_params = load_strategy_parameters(args.params, args.strategy)
        
        # Initialize strategy
        logger.info(f"Initializing strategy: {args.strategy}")
        strategy_class = get_strategy(args.strategy)
        strategy = strategy_class(strategy_params)
        
        # Initialize backtesting engine
        logger.info("Initializing backtesting engine")
        engine = BacktestEngine(
            initial_balance=args.balance,
            fee_rate=args.fee_rate,
            slippage=args.slippage
        )
        
        # Run backtest
        logger.info("Running backtest simulation...")
        equity_curve = engine.run_backtest(data, strategy)
        trades_df = engine.get_trades_df()
        
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        metrics_calc = PerformanceMetrics(equity_curve, trades_df, args.balance)
        metrics = metrics_calc.get_all_metrics()
        
        # Print results
        metrics_calc.print_summary(metrics)
        
        # Plot equity curve
        if not args.no_plot and config.PLOT_EQUITY_CURVE:
            logger.info("Generating equity curve plot...")
            plot_path = config.RESULTS_DIR / f"{args.symbol}_{args.strategy}_equity_curve.png"
            utils.plot_equity_curve(
                equity_curve, 
                strategy_name=f"{args.strategy} ({args.symbol})",
                save_path=plot_path,
                show_trades=True
            )
        
        # Save results
        if not args.no_save and config.SAVE_RESULTS:
            logger.info("Saving results...")
            utils.save_results(equity_curve, trades_df, metrics, args.symbol, args.strategy)
        
        logger.info("Backtesting completed successfully!")
        
        # Return final portfolio value as exit code (for automation)
        final_return = metrics['total_return_pct']
        if final_return > 0:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Loss
    
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error(f"Expected file: {config.DATA_DIR / f'{args.symbol}_{args.timeframe}.csv'}")
        logger.error("Use --sample-data flag to generate sample data for testing")
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Backtesting interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error during backtesting: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()