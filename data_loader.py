"""
Data loading and preprocessing utilities for OHLCV data.
Handles CSV loading, data validation, and basic preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Union

import config

logger = logging.getLogger(__name__)


def load_csv_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.
    
    Expected CSV format:
    timestamp,open,high,low,close,volume
    2023-01-01 00:00:00,16500.0,16600.0,16400.0,16550.0,1000.0
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Validate required columns
        missing_cols = set(config.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data for basic consistency and completeness.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        True if data is valid, raises ValueError if not
    """
    # Check minimum data points
    if len(df) < config.MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data: {len(df)} < {config.MIN_DATA_POINTS}")
    
    # Check for missing values
    if df.isnull().any().any():
        logger.warning("Data contains missing values - will forward fill")
        df.fillna(method='ffill', inplace=True)
    
    # Validate OHLC relationships
    invalid_bars = (
        (df['high'] < df['open']) | 
        (df['high'] < df['close']) |
        (df['low'] > df['open']) | 
        (df['low'] > df['close']) |
        (df['high'] < df['low'])
    )
    
    if invalid_bars.sum() > 0:
        logger.warning(f"Found {invalid_bars.sum()} invalid OHLC bars")
        # Remove invalid bars
        df = df[~invalid_bars]
    
    # Check for negative values
    price_cols = ['open', 'high', 'low', 'close']
    negative_prices = (df[price_cols] <= 0).any(axis=1)
    if negative_prices.sum() > 0:
        logger.warning(f"Found {negative_prices.sum()} bars with negative/zero prices")
        df = df[~negative_prices]
    
    logger.info(f"Data validation complete. Final dataset: {len(df)} bars")
    return True


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add commonly used technical indicators to the dataset.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicators
    """
    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility (20-period rolling std of returns)
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    logger.info("Technical indicators added successfully")
    return df


def get_data(symbol: str, timeframe: str = None) -> pd.DataFrame:
    """
    Main function to load and preprocess data for a given symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '1h', '1d')
        
    Returns:
        Preprocessed DataFrame ready for backtesting
    """
    if timeframe is None:
        timeframe = config.DEFAULT_TIMEFRAME
    
    # Construct file path
    filename = f"{symbol}_{timeframe}.csv"
    file_path = config.DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load and process data
    df = load_csv_data(file_path)
    validate_data(df)
    df = add_technical_indicators(df)
    
    logger.info(f"Data loaded for {symbol} ({timeframe}): {len(df)} bars")
    return df