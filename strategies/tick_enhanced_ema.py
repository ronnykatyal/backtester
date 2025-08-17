"""
Tick-Enhanced EMA Crossover Strategy
Combines your winning 8/21 EMA strategy with tick data insights for improved performance
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from strategies.ema_crossover import EMACrossoverStrategy

logger = logging.getLogger(__name__)


class TickEnhancedEMAStrategy(EMACrossoverStrategy):
    """
    Enhanced EMA strategy that uses tick data for signal confirmation.
    
    Your winning strategy (8/21 EMA @ 75% position) enhanced with:
    - Order flow confirmation (buying vs selling pressure)
    - Volume imbalance filtering (directional volume analysis)
    - Taker volume validation (aggressive trading detection)
    - Large order detection (institutional activity tracking)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize tick-enhanced EMA strategy."""
        
        # Start with your winning parameters
        default_params = {
            'fast_period': 8,
            'slow_period': 21,
            'position_size': 0.75,
            'min_signal_strength': 0.0,
            
            # Tick-based filters
            'order_flow_threshold': 0.6,      # Require 60%+ buying pressure for long
            'taker_volume_threshold': 0.4,    # Require 40%+ taker volume for confirmation
            'volume_imbalance_threshold': 0.1, # Require 10%+ volume imbalance
            'enable_tick_filters': True,       # Enable/disable tick filtering
            'confluence_required': 2           # How many tick filters must confirm (0-4)
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(default_params)
        
        logger.info(f"Tick-Enhanced EMA Strategy initialized:")
        logger.info(f"  Tick Filters Enabled: {self.params['enable_tick_filters']}")
        logger.info(f"  Order Flow Threshold: {self.params['order_flow_threshold']:.1%}")
        logger.info(f"  Confluence Required: {self.params['confluence_required']}")
    
    def validate_tick_data(self, data: pd.DataFrame) -> bool:
        """Validate that data contains required tick features."""
        required_tick_features = [
            'order_flow_ratio', 'volume_imbalance', 'taker_volume_ratio', 'large_order_count'
        ]
        
        missing_features = [col for col in required_tick_features if col not in data.columns]
        
        if missing_features:
            if self.params['enable_tick_filters']:
                logger.warning(f"Missing tick features: {missing_features}. Disabling tick filters.")
                self.params['enable_tick_filters'] = False
            return False
        
        return True
    
    def calculate_tick_confluence(self, data: pd.DataFrame, signal_direction: int) -> pd.Series:
        """
        Calculate tick-based confluence score for each signal.
        
        Args:
            data: DataFrame with tick features
            signal_direction: 1 for buy signals, -1 for sell signals
            
        Returns:
            Series with confluence scores (0-4)
        """
        confluence_score = pd.Series(0, index=data.index)
        
        if not self.params['enable_tick_filters']:
            return confluence_score
        
        # Filter 1: Order Flow Direction
        if signal_direction == 1:  # Buy signal
            filter1 = data['order_flow_ratio'] >= self.params['order_flow_threshold']
        else:  # Sell signal  
            filter1 = data['order_flow_ratio'] <= (1 - self.params['order_flow_threshold'])
        
        # Filter 2: Volume Imbalance
        if signal_direction == 1:
            filter2 = data['volume_imbalance'] >= self.params['volume_imbalance_threshold']
        else:
            filter2 = data['volume_imbalance'] <= -self.params['volume_imbalance_threshold']
        
        # Filter 3: Taker Volume (Aggressive trading)
        filter3 = data['taker_volume_ratio'] >= self.params['taker_volume_threshold']
        
        # Filter 4: Large Order Activity (Institutional interest)
        large_order_threshold = data['large_order_count'].quantile(0.7)  # Top 30%
        filter4 = data['large_order_count'] >= large_order_threshold
        
        # Calculate confluence score
        confluence_score = (
            filter1.astype(int) + 
            filter2.astype(int) + 
            filter3.astype(int) + 
            filter4.astype(int)
        )
        
        return confluence_score
    
    def apply_tick_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply tick-based filters to EMA signals."""
        df = data.copy()
        
        if not self.params['enable_tick_filters']:
            logger.info("Tick filters disabled - using pure EMA signals")
            return df
        
        # Calculate confluence for buy and sell signals separately
        buy_confluence = self.calculate_tick_confluence(df, signal_direction=1)
        sell_confluence = self.calculate_tick_confluence(df, signal_direction=-1)
        
        # Store confluence scores
        df['buy_confluence'] = buy_confluence
        df['sell_confluence'] = sell_confluence
        
        # Apply confluence filtering
        confluence_required = self.params['confluence_required']
        
        # Filter buy signals
        buy_mask = (df['signal'] == 1) & (buy_confluence >= confluence_required)
        
        # Filter sell signals  
        sell_mask = (df['signal'] == -1) & (sell_confluence >= confluence_required)
        
        # Update signals based on confluence
        df['signal_original'] = df['signal'].copy()
        df.loc[~(buy_mask | sell_mask), 'signal'] = 0  # Remove low-confluence signals
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate tick-enhanced EMA crossover signals.
        
        Process:
        1. Generate base EMA crossover signals (your winning strategy)
        2. Validate tick data availability  
        3. Apply tick-based confluence filters
        4. Return filtered high-confidence signals
        """
        # Validate input data
        self.validate_data(data)
        self.validate_tick_data(data)
        
        logger.info(f"Generating tick-enhanced EMA signals for {len(data)} data points...")
        
        # Step 1: Generate base EMA signals using parent class
        df = super().generate_signals(data)
        
        # Step 2: Apply tick-based filters
        df = self.apply_tick_filters(df)
        
        # Count and log signals
        original_buy_signals = (df.get('signal_original', df['signal']) == 1).sum()
        original_sell_signals = (df.get('signal_original', df['signal']) == -1).sum()
        filtered_buy_signals = (df['signal'] == 1).sum()
        filtered_sell_signals = (df['signal'] == -1).sum()
        
        logger.info(f"Tick-Enhanced EMA signals generated:")
        logger.info(f"  Original EMA Buy signals: {original_buy_signals}")
        logger.info(f"  Filtered Buy signals: {filtered_buy_signals}")
        logger.info(f"  Original EMA Sell signals: {original_sell_signals}")
        logger.info(f"  Filtered Sell signals: {filtered_sell_signals}")
        
        if self.params['enable_tick_filters']:
            filter_effectiveness = (
                (original_buy_signals + original_sell_signals - 
                 filtered_buy_signals - filtered_sell_signals) / 
                max(original_buy_signals + original_sell_signals, 1)
            )
            logger.info(f"  Filter effectiveness: {filter_effectiveness:.1%} signals removed")
        
        return df
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get enhanced strategy information."""
        base_info = super().get_strategy_info()
        
        base_info.update({
            'name': 'Tick-Enhanced EMA Crossover Strategy',
            'description': 'EMA crossover with tick data confluence filtering',
            'type': 'Trend Following + Market Microstructure',
            'tick_features': [
                'Order flow analysis',
                'Volume imbalance detection', 
                'Taker volume confirmation',
                'Large order activity tracking'
            ],
            'enhancement': 'Reduces false signals, improves entry timing'
        })
        
        return base_info
    
    def __str__(self) -> str:
        """String representation."""
        return (f"TickEnhancedEMA(fast={self.params['fast_period']}, "
                f"slow={self.params['slow_period']}, "
                f"position_size={self.params['position_size']:.1%}, "
                f"tick_filters={'ON' if self.params['enable_tick_filters'] else 'OFF'})")


# Example usage for testing
if __name__ == "__main__":
    # Create sample enhanced data for testing
    import numpy as np
    
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # Sample OHLCV data
    prices = [100000]  # Start at $100K BTC
    for i in range(99):
        change = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    # Create enhanced dataset with tick features
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices], 
        'close': prices,
        'volume': np.random.uniform(50, 200, 100),
        
        # Mock tick features (your real data has these)
        'order_flow_ratio': np.random.uniform(0.3, 0.7, 100),
        'volume_imbalance': np.random.normal(0, 0.2, 100),
        'taker_volume_ratio': np.random.uniform(0.2, 0.8, 100),
        'large_order_count': np.random.poisson(5, 100)
    }, index=dates)
    
    # Test strategy
    strategy = TickEnhancedEMAStrategy({
        'fast_period': 8,
        'slow_period': 21,
        'position_size': 0.75,
        'enable_tick_filters': True,
        'confluence_required': 2
    })
    
    # Generate signals
    result = strategy.generate_signals(data)
    
    print(f"Strategy: {strategy}")
    print(f"Total signals generated: {(result['signal'] != 0).sum()}")
    print(f"Buy signals: {(result['signal'] == 1).sum()}")
    print(f"Sell signals: {(result['signal'] == -1).sum()}")
    
    if 'signal_original' in result.columns:
        original_signals = (result['signal_original'] != 0).sum()
        filtered_signals = (result['signal'] != 0).sum()
        print(f"Filter effectiveness: {(original_signals - filtered_signals)/original_signals:.1%} signals removed")