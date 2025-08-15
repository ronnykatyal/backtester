"""
Global configuration settings for the backtesting environment.
All configurable parameters should be defined here to avoid hardcoding.
"""

import os
from pathlib import Path

# Data Configuration
DATA_DIR = Path("data")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1h"

# Trading Configuration
DEFAULT_INITIAL_BALANCE = 10000.0  # Starting capital in USDT
DEFAULT_FEE_RATE = 0.001  # 0.1% trading fee
DEFAULT_SLIPPAGE = 0.0005  # 0.05% slippage

# Strategy Configuration
STRATEGIES_DIR = Path("strategies")
DEFAULT_STRATEGY = "ema_crossover"

# Output Configuration
RESULTS_DIR = Path("results")
PLOT_EQUITY_CURVE = True
SAVE_RESULTS = True

# Performance Metrics Configuration
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate for Sharpe calculation
TRADING_DAYS_PER_YEAR = 365  # Crypto trades 365 days

# Data Validation
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
MIN_DATA_POINTS = 100  # Minimum data points required for backtesting

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create necessary directories
def ensure_directories():
    """Create required directories if they don't exist."""
    for directory in [DATA_DIR, RESULTS_DIR]:
        directory.mkdir(exist_ok=True)

# Strategy parameters (can be overridden via CLI)
STRATEGY_PARAMS = {
    "ema_crossover": {
        "fast_period": 12,
        "slow_period": 26,
        "position_size": 1.0  # Use full balance (1.0 = 100%)
    }
}