"""
Core backtesting engine that simulates trading based on strategy signals.
Handles position management, trade execution, and portfolio tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

import config

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Core backtesting engine that executes trades based on strategy signals.
    """
    
    def __init__(self, initial_balance: float = None, fee_rate: float = None, slippage: float = None):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_balance: Starting capital
            fee_rate: Trading fee as decimal (0.001 = 0.1%)
            slippage: Slippage as decimal (0.0005 = 0.05%)
        """
        self.initial_balance = initial_balance or config.DEFAULT_INITIAL_BALANCE
        self.fee_rate = fee_rate or config.DEFAULT_FEE_RATE
        self.slippage = slippage or config.DEFAULT_SLIPPAGE
        
        # Portfolio state
        self.balance = self.initial_balance
        self.position = 0.0  # Number of coins held
        self.position_value = 0.0  # Value of position in USDT
        
        # Trade tracking
        self.trades = []
        self.equity_curve = []
        
        logger.info(f"Engine initialized: ${self.initial_balance:,.2f} balance, {self.fee_rate:.4f} fee rate")
    
    def calculate_trade_cost(self, quantity: float, price: float, is_buy: bool) -> Tuple[float, float]:
        """
        Calculate the actual execution price and fees for a trade.
        
        Args:
            quantity: Number of coins to trade
            price: Market price
            is_buy: True for buy order, False for sell
            
        Returns:
            Tuple of (execution_price, total_fee)
        """
        # Apply slippage (worse price for trader)
        slippage_factor = 1 + self.slippage if is_buy else 1 - self.slippage
        execution_price = price * slippage_factor
        
        # Calculate fee
        trade_value = quantity * execution_price
        fee = trade_value * self.fee_rate
        
        return execution_price, fee
    
    def calculate_max_position_size(self, available_balance: float, price: float, position_fraction: float = 1.0) -> float:
        """
        Calculate maximum position size accounting for slippage and fees.
        
        Args:
            available_balance: Available cash balance
            price: Current market price
            position_fraction: Fraction of balance to use (0.0 to 1.0)
            
        Returns:
            Maximum quantity that can be purchased
        """
        target_investment = available_balance * position_fraction
        
        # Account for slippage and fees in the calculation
        # Total cost = quantity * price * (1 + slippage) * (1 + fee_rate)
        slippage_factor = 1 + self.slippage
        total_cost_factor = slippage_factor * (1 + self.fee_rate)
        
        # Add small buffer for floating point precision (0.1%)
        safety_buffer = 1.001
        
        # Calculate max quantity with safety buffer
        max_quantity = target_investment / (price * total_cost_factor * safety_buffer)
        
        return max_quantity
    
    def execute_trade(self, signal: int, price: float, timestamp: pd.Timestamp, strategy) -> bool:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: Trading signal (1=buy, -1=sell, 0=hold)
            price: Current market price
            timestamp: Current timestamp
            strategy: Strategy instance for position sizing
            
        Returns:
            True if trade was executed, False otherwise
        """
        if signal == 0:
            return False
        
        trade_executed = False
        
        if signal == 1 and self.position == 0:  # Buy signal and no position
            # Calculate position size accounting for trading costs
            position_fraction = strategy.params.get('position_size', 1.0)
            quantity = self.calculate_max_position_size(self.balance, price, position_fraction)
            
            if quantity > 0:
                execution_price, fee = self.calculate_trade_cost(quantity, price, is_buy=True)
                total_cost = quantity * execution_price + fee
                
                # Double-check we have enough balance (should always pass now)
                if total_cost <= self.balance:
                    # Execute buy
                    self.balance -= total_cost
                    self.position = quantity
                    self.position_value = quantity * execution_price
                    
                    # Record trade
                    trade = {
                        'timestamp': timestamp,
                        'type': 'BUY',
                        'quantity': quantity,
                        'price': execution_price,
                        'fee': fee,
                        'balance_after': self.balance,
                        'position_after': self.position
                    }
                    self.trades.append(trade)
                    trade_executed = True
                    
                    logger.info(f"BUY EXECUTED: {quantity:.6f} @ ${execution_price:.2f}, fee: ${fee:.2f}, cost: ${total_cost:.2f}")
                else:
                    logger.warning(f"Buy calculation error: cost ${total_cost:.2f} > balance ${self.balance:.2f}")
            else:
                logger.info(f"Buy rejected: invalid quantity {quantity}")
        
        elif signal == -1 and self.position > 0:  # Sell signal and have position
            execution_price, fee = self.calculate_trade_cost(self.position, price, is_buy=False)
            sale_proceeds = self.position * execution_price - fee
            
            # Execute sell
            self.balance += sale_proceeds
            sold_quantity = self.position
            self.position = 0.0
            self.position_value = 0.0
            
            # Record trade
            trade = {
                'timestamp': timestamp,
                'type': 'SELL',
                'quantity': sold_quantity,
                'price': execution_price,
                'fee': fee,
                'balance_after': self.balance,
                'position_after': self.position
            }
            self.trades.append(trade)
            trade_executed = True
            
            logger.info(f"SELL EXECUTED: {sold_quantity:.6f} @ ${execution_price:.2f}, fee: ${fee:.2f}")
        
        else:
            # Log why trade was rejected
            if signal == 1 and self.position > 0:
                logger.debug(f"Buy signal ignored: already have position ({self.position:.6f})")
            elif signal == -1 and self.position == 0:
                logger.debug(f"Sell signal ignored: no position to sell")
        
        return trade_executed
    
    def update_portfolio_value(self, current_price: float) -> float:
        """
        Update and return current total portfolio value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Total portfolio value (balance + position value)
        """
        self.position_value = self.position * current_price
        total_value = self.balance + self.position_value
        return total_value
    
    def run_backtest(self, data: pd.DataFrame, strategy) -> pd.DataFrame:
        """
        Run the complete backtest simulation.
        
        Args:
            data: DataFrame with OHLCV data and signals
            strategy: Strategy instance
            
        Returns:
            DataFrame with trade history and portfolio values
        """
        logger.info(f"Starting backtest with {len(data)} data points")
        
        # Reset engine state
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.trades = []
        self.equity_curve = []
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data)
        
        # Simulate trading
        for timestamp, row in data_with_signals.iterrows():
            current_price = row['close']
            signal = row['signal']
            
            # Execute trade if signal present
            trade_executed = self.execute_trade(signal, current_price, timestamp, strategy)
            
            # Update portfolio value and record
            total_value = self.update_portfolio_value(current_price)
            
            equity_point = {
                'timestamp': timestamp,
                'price': current_price,
                'balance': self.balance,
                'position': self.position,
                'position_value': self.position_value,
                'total_value': total_value,
                'signal': signal
            }
            self.equity_curve.append(equity_point)
        
        logger.info(f"Backtest completed: {len(self.trades)} trades executed")
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        return equity_df
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.set_index('timestamp', inplace=True)
        return trades_df