"""
EMA Crossover Strategy Implementation.
Generates buy/sell signals based on exponential moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseStrategy:
    """
    Base class for all trading strategies.
    All strategies must inherit from this class and implement the required methods.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize strategy with parameters.
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the given data.
        Must be implemented by each strategy.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with additional 'signal' column (1=buy, -1=sell, 0=hold)
        """
        raise NotImplementedError("Strategy must implement generate_signals method")
    
    def get_position_size(self, current_balance: float, price: float) -> float:
        """
        Calculate position size for a trade.
        Can be overridden by strategies for custom position sizing.
        
        Args:
            current_balance: Available balance
            price: Current asset price
            
        Returns:
            Position size (number of shares/coins)
        """
        position_fraction = self.params.get('position_size', 1.0)
        return (current_balance * position_fraction) / price


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.
    
    Generates signals based on crossover between fast and slow EMAs:
    - Buy when fast EMA crosses above slow EMA
    - Sell when fast EMA crosses below slow EMA
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize EMA Crossover strategy.
        
        Expected parameters:
        - fast_period: Period for fast EMA (default: 12)
        - slow_period: Period for slow EMA (default: 26)
        - position_size: Fraction of balance to use (default: 1.0)
        """
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'position_size': 1.0
        }
        if params:
            default_params.update(params)
        
        super().__init__(default_params)
        
        if self.params['fast_period'] >= self.params['slow_period']:
            raise ValueError("Fast period must be less than slow period")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate EMA crossover signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'signal' column added
        """
        df = data.copy()
        
        # Calculate EMAs if not already present
        fast_col = f"ema_{self.params['fast_period']}"
        slow_col = f"ema_{self.params['slow_period']}"
        
        if fast_col not in df.columns:
            df[fast_col] = df['close'].ewm(span=self.params['fast_period']).mean()
        if slow_col not in df.columns:
            df[slow_col] = df['close'].ewm(span=self.params['slow_period']).mean()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate crossover signals
        # Buy signal: fast EMA crosses above slow EMA
        buy_condition = (
            (df[fast_col] > df[slow_col]) & 
            (df[fast_col].shift(1) <= df[slow_col].shift(1))
        )
        
        # Sell signal: fast EMA crosses below slow EMA  
        sell_condition = (
            (df[fast_col] < df[slow_col]) & 
            (df[fast_col].shift(1) >= df[slow_col].shift(1))
        )
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Log signal statistics
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        logger.info(f"EMA Crossover signals generated: {buy_signals} buys, {sell_signals} sells")
        
        return df
    
    def __str__(self):
        """String representation of the strategy."""
        return f"EMACrossover(fast={self.params['fast_period']}, slow={self.params['slow_period']})"