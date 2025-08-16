"""
EMA Crossover Strategy Implementation.
Generates buy/sell signals based on exponential moving average crossovers.

Strategy Logic:
- Buy Signal: Fast EMA crosses above Slow EMA (golden cross)
- Sell Signal: Fast EMA crosses below Slow EMA (death cross)
- Position Sizing: Configurable fraction of available balance
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
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
        logger.info(f"Initializing {self.name} with parameters: {self.params}")
        
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
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data contains required columns for the strategy.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("Data is empty")
            
        return True


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.
    
    This strategy generates trading signals based on the crossover of two 
    Exponential Moving Averages (EMAs):
    
    Buy Signal (Golden Cross):
    - Fast EMA crosses above Slow EMA
    - Indicates potential upward trend
    
    Sell Signal (Death Cross):
    - Fast EMA crosses below Slow EMA  
    - Indicates potential downward trend
    
    Parameters:
    - fast_period: Period for fast EMA (default: 12)
    - slow_period: Period for slow EMA (default: 26)
    - position_size: Fraction of balance to use (0.0-1.0, default: 1.0)
    - min_signal_strength: Minimum % difference between EMAs for signal (default: 0.0)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize EMA Crossover strategy.
        
        Args:
            params: Dictionary containing strategy parameters
        """
        # Set default parameters
        default_params = {
            'fast_period': 12,
            'slow_period': 26, 
            'position_size': 1.0,
            'min_signal_strength': 0.0  # Minimum % difference for signal
        }
        
        # Merge with user parameters
        if params:
            default_params.update(params)
        
        super().__init__(default_params)
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"EMA Crossover Strategy initialized:")
        logger.info(f"  Fast EMA Period: {self.params['fast_period']}")
        logger.info(f"  Slow EMA Period: {self.params['slow_period']}")
        logger.info(f"  Position Size: {self.params['position_size']:.1%}")
        logger.info(f"  Min Signal Strength: {self.params['min_signal_strength']:.2%}")
    
    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        position_size = self.params['position_size']
        min_signal_strength = self.params['min_signal_strength']
        
        # Validate EMA periods
        if not isinstance(fast_period, (int, float)) or fast_period <= 0:
            raise ValueError(f"Fast period must be positive number, got: {fast_period}")
            
        if not isinstance(slow_period, (int, float)) or slow_period <= 0:
            raise ValueError(f"Slow period must be positive number, got: {slow_period}")
            
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
        
        # Validate position size
        if not isinstance(position_size, (int, float)) or not (0 < position_size <= 1):
            raise ValueError(f"Position size must be between 0 and 1, got: {position_size}")
        
        # Validate signal strength
        if not isinstance(min_signal_strength, (int, float)) or min_signal_strength < 0:
            raise ValueError(f"Min signal strength must be non-negative, got: {min_signal_strength}")
    
    def calculate_emas(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with EMA columns added
        """
        df = data.copy()
        
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        
        # Calculate EMAs
        fast_ema_col = f'ema_{fast_period}'
        slow_ema_col = f'ema_{slow_period}'
        
        df[fast_ema_col] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df[slow_ema_col] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate EMA difference for signal strength
        df['ema_diff'] = df[fast_ema_col] - df[slow_ema_col]
        df['ema_diff_pct'] = df['ema_diff'] / df[slow_ema_col]
        
        return df
    
    def detect_crossovers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect EMA crossover points.
        
        Args:
            data: DataFrame with EMA data
            
        Returns:
            DataFrame with crossover flags
        """
        df = data.copy()
        
        fast_ema_col = f'ema_{self.params["fast_period"]}'
        slow_ema_col = f'ema_{self.params["slow_period"]}'
        
        # Previous period values for crossover detection
        df['fast_ema_prev'] = df[fast_ema_col].shift(1)
        df['slow_ema_prev'] = df[slow_ema_col].shift(1)
        df['ema_diff_prev'] = df['ema_diff'].shift(1)
        
        # Crossover conditions
        # Golden cross: fast EMA crosses above slow EMA
        df['golden_cross'] = (
            (df[fast_ema_col] > df[slow_ema_col]) & 
            (df['fast_ema_prev'] <= df['slow_ema_prev'])
        )
        
        # Death cross: fast EMA crosses below slow EMA
        df['death_cross'] = (
            (df[fast_ema_col] < df[slow_ema_col]) & 
            (df['fast_ema_prev'] >= df['slow_ema_prev'])
        )
        
        return df
    
    def apply_signal_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional filters to refine signals.
        
        Args:
            data: DataFrame with crossover data
            
        Returns:
            DataFrame with filtered signals
        """
        df = data.copy()
        
        min_strength = self.params['min_signal_strength']
        
        # Apply minimum signal strength filter
        if min_strength > 0:
            # For buy signals, require fast EMA to be sufficiently above slow EMA
            strong_golden_cross = df['golden_cross'] & (df['ema_diff_pct'] >= min_strength)
            
            # For sell signals, require fast EMA to be sufficiently below slow EMA  
            strong_death_cross = df['death_cross'] & (df['ema_diff_pct'] <= -min_strength)
            
            df['filtered_golden_cross'] = strong_golden_cross
            df['filtered_death_cross'] = strong_death_cross
        else:
            df['filtered_golden_cross'] = df['golden_cross']
            df['filtered_death_cross'] = df['death_cross']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate EMA crossover trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'signal' column added (1=buy, -1=sell, 0=hold)
        """
        # Validate input data
        self.validate_data(data)
        
        logger.info(f"Generating EMA crossover signals for {len(data)} data points...")
        
        # Calculate EMAs
        df = self.calculate_emas(data)
        
        # Detect crossovers
        df = self.detect_crossovers(df)
        
        # Apply signal filters
        df = self.apply_signal_filters(df)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate final signals
        df.loc[df['filtered_golden_cross'], 'signal'] = 1   # Buy signal
        df.loc[df['filtered_death_cross'], 'signal'] = -1   # Sell signal
        
        # Count and log signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        total_signals = buy_signals + sell_signals
        
        logger.info(f"EMA Crossover signals generated:")
        logger.info(f"  Buy signals (Golden Cross): {buy_signals}")
        logger.info(f"  Sell signals (Death Cross): {sell_signals}")
        logger.info(f"  Total signals: {total_signals}")
        
        if total_signals > 0:
            # Log signal dates for debugging
            signal_dates = df[df['signal'] != 0].index
            logger.info(f"  Signal dates: {signal_dates.strftime('%Y-%m-%d').tolist()}")
        
        # Clean up intermediate columns (optional - keep for debugging)
        columns_to_keep = [
            'open', 'high', 'low', 'close', 'volume',
            f'ema_{self.params["fast_period"]}',
            f'ema_{self.params["slow_period"]}',
            'ema_diff', 'ema_diff_pct', 'signal'
        ]
        
        # Keep only essential columns plus any existing technical indicators
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        extra_columns = [col for col in df.columns if col.startswith(('sma_', 'ema_', 'rsi', 'macd'))]
        
        final_columns = list(set(existing_columns + extra_columns))
        df = df[final_columns]
        
        return df
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current parameters.
        
        Returns:
            Dictionary with strategy details
        """
        return {
            'name': 'EMA Crossover Strategy',
            'description': 'Generates signals based on EMA crossovers',
            'type': 'Trend Following',
            'parameters': self.params.copy(),
            'signals': {
                'buy': 'Fast EMA crosses above Slow EMA (Golden Cross)',
                'sell': 'Fast EMA crosses below Slow EMA (Death Cross)'
            },
            'risk_level': 'Medium',
            'suitable_markets': ['Trending', 'Volatile'],
            'timeframes': ['1h', '4h', '1d']
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return (f"EMACrossover(fast={self.params['fast_period']}, "
                f"slow={self.params['slow_period']}, "
                f"position_size={self.params['position_size']:.1%})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"EMACrossoverStrategy({self.params})"


# Example usage and testing functions
def test_strategy():
    """
    Test function to verify strategy works correctly.
    """
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample price data with trend
    prices = [100]
    for i in range(99):
        change = np.random.normal(0.01, 0.02)  # 1% daily return, 2% volatility
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Test strategy
    strategy = EMACrossoverStrategy({
        'fast_period': 5,
        'slow_period': 20,
        'position_size': 0.8,
        'min_signal_strength': 0.01
    })
    
    # Generate signals
    result = strategy.generate_signals(data)
    
    print("Strategy test completed successfully!")
    print(f"Generated {(result['signal'] != 0).sum()} signals")
    print(f"Strategy info: {strategy}")
    
    return result


if __name__ == "__main__":
    # Run test if script is executed directly
    test_strategy()