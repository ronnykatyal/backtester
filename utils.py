"""
Utility functions for the backtesting environment.
Contains helper functions for plotting, data manipulation, and file operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging
from typing import Dict, Any, Optional

import config

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    if log_level is None:
        log_level = config.LOG_LEVEL
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    logger.info(f"Logging configured at {log_level} level")


def plot_equity_curve(equity_df: pd.DataFrame, strategy_name: str = "Strategy", 
                     save_path: Optional[Path] = None, show_trades: bool = True) -> None:
    """
    Plot the equity curve with optional trade markers.
    
    Args:
        equity_df: DataFrame with equity curve data
        strategy_name: Name of the strategy for title
        save_path: Path to save the plot (optional)
        show_trades: Whether to show trade markers
    """
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main equity curve
    ax1.plot(equity_df.index, equity_df['total_value'], 'b-', linewidth=2, label='Portfolio Value')
    ax1.plot(equity_df.index, equity_df['price'] * (equity_df['total_value'].iloc[0] / equity_df['price'].iloc[0]), 
             'k--', alpha=0.7, label='Buy & Hold')
    
    # Add trade markers if requested
    if show_trades and 'signal' in equity_df.columns:
        buy_signals = equity_df[equity_df['signal'] == 1]
        sell_signals = equity_df[equity_df['signal'] == -1]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['total_value'], 
                       color='green', marker='^', s=60, label='Buy', zorder=5)
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['total_value'], 
                       color='red', marker='v', s=60, label='Sell', zorder=5)
    
    ax1.set_title(f'{strategy_name} - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    # Drawdown plot
    if 'drawdown' in equity_df.columns:
        ax2.fill_between(equity_df.index, equity_df['drawdown'] * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(equity_df.index, equity_df['drawdown'] * 100, 'r-', linewidth=1)
    
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def save_results(equity_df: pd.DataFrame, trades_df: pd.DataFrame, 
                metrics: Dict[str, Any], symbol: str, strategy_name: str) -> None:
    """
    Save backtest results to files.
    
    Args:
        equity_df: Equity curve DataFrame
        trades_df: Trades DataFrame
        metrics: Performance metrics dictionary
        symbol: Trading symbol
        strategy_name: Strategy name
    """
    if not config.SAVE_RESULTS:
        return
    
    config.ensure_directories()
    
    # Create filename prefix
    prefix = f"{symbol}_{strategy_name}"
    
    # Save equity curve
    equity_path = config.RESULTS_DIR / f"{prefix}_equity.csv"
    equity_df.to_csv(equity_path)
    
    # Save trades
    if not trades_df.empty:
        trades_path = config.RESULTS_DIR / f"{prefix}_trades.csv"
        trades_df.to_csv(trades_path)
    
    # Save metrics
    metrics_path = config.RESULTS_DIR / f"{prefix}_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = value.item()
            elif pd.isna(value):
                serializable_metrics[key] = None
            else:
                serializable_metrics[key] = value
        
        json.dump(serializable_metrics, f, indent=2, default=str)
    
    logger.info(f"Results saved to {config.RESULTS_DIR}")


def validate_parameters(params: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Validate and sanitize strategy parameters.
    
    Args:
        params: Parameters dictionary
        strategy_name: Name of the strategy
        
    Returns:
        Validated parameters dictionary
    """
    validated_params = params.copy()
    
    # Get default parameters for the strategy
    default_params = config.STRATEGY_PARAMS.get(strategy_name, {})
    
    # Fill in missing parameters with defaults
    for key, default_value in default_params.items():
        if key not in validated_params:
            validated_params[key] = default_value
            logger.info(f"Using default value for {key}: {default_value}")
    
    # Validate parameter types and ranges
    if strategy_name == "ema_crossover":
        if validated_params.get('fast_period', 0) >= validated_params.get('slow_period', 1):
            raise ValueError("Fast period must be less than slow period")
        
        if validated_params.get('position_size', 0) <= 0 or validated_params.get('position_size', 0) > 1:
            raise ValueError("Position size must be between 0 and 1")
    
    return validated_params


def format_number(value: float, as_percentage: bool = False, decimal_places: int = 2) -> str:
    """
    Format numbers for display.
    
    Args:
        value: Number to format
        as_percentage: Whether to format as percentage
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    if as_percentage:
        return f"{value * 100:.{decimal_places}f}%"
    
    if abs(value) >= 1000000:
        return f"${value/1000000:.{decimal_places}f}M"
    elif abs(value) >= 1000:
        return f"${value/1000:.{decimal_places}f}K"
    else:
        return f"${value:.{decimal_places}f}"


def create_sample_data(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing (simple random walk).
    
    Args:
        symbol: Trading symbol
        days: Number of days of data to generate
        
    Returns:
        DataFrame with sample OHLCV data
    """
    logger.warning("Creating sample data - use real data for actual backtesting")
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate price data (random walk)
    np.random.seed(42)  # For reproducible results
    initial_price = 30000 if 'BTC' in symbol else 2000
    
    returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated {len(df)} days of sample data for {symbol}")
    return df