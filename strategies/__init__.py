"""
Strategy module initialization.
Automatically imports all strategy classes for easy access.
"""

from .ema_crossover import EMACrossoverStrategy
from .tick_enhanced_ema import TickEnhancedEMAStrategy

# Registry of available strategies
AVAILABLE_STRATEGIES = {
    "ema_crossover": EMACrossoverStrategy,
    "tick_enhanced_ema": TickEnhancedEMAStrategy,
}

def get_strategy(strategy_name: str):
    """
    Get strategy class by name.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy class
    """
    if strategy_name not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(AVAILABLE_STRATEGIES.keys())}")
    
    return AVAILABLE_STRATEGIES[strategy_name]