"""
Performance metrics calculation for backtesting results.
Computes standard trading performance indicators and risk metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

import config

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.
    """
    
    def __init__(self, equity_curve: pd.DataFrame, trades_df: pd.DataFrame, initial_balance: float):
        """
        Initialize with backtest results.
        
        Args:
            equity_curve: DataFrame with portfolio values over time
            trades_df: DataFrame with individual trade details
            initial_balance: Starting capital
        """
        self.equity_curve = equity_curve.copy()
        self.trades_df = trades_df.copy()
        self.initial_balance = initial_balance
        
        # Calculate returns
        self.equity_curve['returns'] = self.equity_curve['total_value'].pct_change()
        self.equity_curve['cumulative_returns'] = (
            (self.equity_curve['total_value'] / initial_balance) - 1
        )
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic performance metrics.
        
        Returns:
            Dictionary with basic metrics
        """
        final_value = self.equity_curve['total_value'].iloc[-1]
        total_return = (final_value / self.initial_balance) - 1
        
        metrics = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': len(self.trades_df),
            'num_days': len(self.equity_curve),
        }
        
        # Annualized return
        if metrics['num_days'] > 0:
            days_per_year = config.TRADING_DAYS_PER_YEAR
            years = metrics['num_days'] / days_per_year
            metrics['annualized_return'] = (final_value / self.initial_balance) ** (1/years) - 1
            metrics['annualized_return_pct'] = metrics['annualized_return'] * 100
        else:
            metrics['annualized_return'] = 0
            metrics['annualized_return_pct'] = 0
        
        return metrics
    
    def calculate_drawdown_metrics(self) -> Dict[str, float]:
        """
        Calculate drawdown-related metrics.
        
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate running maximum (peak)
        running_max = self.equity_curve['total_value'].expanding().max()
        
        # Calculate drawdown
        drawdown = (self.equity_curve['total_value'] - running_max) / running_max
        self.equity_curve['drawdown'] = drawdown
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find max drawdown period
        drawdown_start = None
        drawdown_end = None
        max_dd_duration = 0
        
        if not drawdown.empty:
            # Find the index of maximum drawdown
            max_dd_idx = drawdown.idxmin()
            
            # Find start of drawdown (last peak before max DD)
            peak_before = running_max.loc[:max_dd_idx]
            if len(peak_before) > 0:
                drawdown_start = peak_before.idxmax()
            
            # Find end of drawdown (when we recover to new high)
            recovery_data = self.equity_curve.loc[max_dd_idx:]['total_value']
            peak_value = running_max.loc[max_dd_idx]
            recovery_idx = recovery_data[recovery_data >= peak_value].first_valid_index()
            
            if recovery_idx is not None:
                drawdown_end = recovery_idx
                max_dd_duration = (drawdown_end - drawdown_start).days
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'drawdown_start': drawdown_start,
            'drawdown_end': drawdown_end,
            'max_drawdown_duration_days': max_dd_duration,
        }
    
    def calculate_trade_metrics(self) -> Dict[str, float]:
        """
        Calculate trade-specific metrics.
        
        Returns:
            Dictionary with trade metrics
        """
        if self.trades_df.empty:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'avg_winning_trade': 0,
                'avg_losing_trade': 0,
                'largest_win': 0,
                'largest_loss': 0,
            }
        
        # Match buy and sell trades to calculate trade P&L
        buy_trades = self.trades_df[self.trades_df['type'] == 'BUY'].copy()
        sell_trades = self.trades_df[self.trades_df['type'] == 'SELL'].copy()
        
        trade_pnl = []
        
        # Simple matching: each sell follows a buy
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades.iloc[i]['price'] + (buy_trades.iloc[i]['fee'] / buy_trades.iloc[i]['quantity'])
            sell_price = sell_trades.iloc[i]['price'] - (sell_trades.iloc[i]['fee'] / sell_trades.iloc[i]['quantity'])
            
            pnl_per_unit = sell_price - buy_price
            total_pnl = pnl_per_unit * sell_trades.iloc[i]['quantity']
            trade_pnl.append(total_pnl)
        
        if not trade_pnl:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'avg_winning_trade': 0,
                'avg_losing_trade': 0,
                'largest_win': 0,
                'largest_loss': 0,
            }
        
        trade_pnl = np.array(trade_pnl)
        
        # Win/Loss statistics
        winning_trades = trade_pnl[trade_pnl > 0]
        losing_trades = trade_pnl[trade_pnl < 0]
        
        win_rate = len(winning_trades) / len(trade_pnl) if len(trade_pnl) > 0 else 0
        
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'completed_trades': len(trade_pnl),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_trade_return': trade_pnl.mean(),
            'avg_winning_trade': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_losing_trade': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades.min() if len(losing_trades) > 0 else 0,
        }
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        returns = self.equity_curve['returns'].dropna()
        
        if len(returns) == 0:
            return {
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
            }
        
        # Annualized volatility
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        
        # Sharpe ratio
        excess_returns = returns.mean() - (config.RISK_FREE_RATE / config.TRADING_DAYS_PER_YEAR)
        sharpe_ratio = excess_returns / daily_vol if daily_vol > 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns / downside_vol if downside_vol > 0 else 0
        annualized_sortino = sortino_ratio * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        
        # Calmar ratio (annual return / max drawdown)
        max_drawdown = abs(self.equity_curve['drawdown'].min()) if 'drawdown' in self.equity_curve.columns else 0
        annualized_return = ((self.equity_curve['total_value'].iloc[-1] / self.initial_balance) ** 
                           (config.TRADING_DAYS_PER_YEAR / len(self.equity_curve)) - 1)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'volatility': daily_vol,
            'annualized_volatility': annualized_vol,
            'annualized_volatility_pct': annualized_vol * 100,
            'sharpe_ratio': annualized_sharpe,
            'sortino_ratio': annualized_sortino,
            'calmar_ratio': calmar_ratio,
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return all performance metrics.
        
        Returns:
            Dictionary with all calculated metrics
        """
        logger.info("Calculating performance metrics...")
        
        metrics = {}
        metrics.update(self.calculate_basic_metrics())
        metrics.update(self.calculate_drawdown_metrics())
        metrics.update(self.calculate_trade_metrics())
        metrics.update(self.calculate_risk_metrics())
        
        logger.info("Performance metrics calculation complete")
        return metrics
    
    def print_summary(self, metrics: Dict[str, Any] = None):
        """
        Print a formatted summary of performance metrics.
        
        Args:
            metrics: Pre-calculated metrics dict (optional)
        """
        if metrics is None:
            metrics = self.get_all_metrics()
        
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        # Basic Performance
        print(f"\n📊 BASIC PERFORMANCE")
        print(f"Initial Balance:      ${metrics['initial_balance']:>12,.2f}")
        print(f"Final Value:          ${metrics['final_value']:>12,.2f}")
        print(f"Total Return:         {metrics['total_return_pct']:>12.2f}%")
        print(f"Annualized Return:    {metrics['annualized_return_pct']:>12.2f}%")
        
        # Risk Metrics
        print(f"\n⚠️  RISK METRICS")
        print(f"Max Drawdown:         {metrics['max_drawdown_pct']:>12.2f}%")
        print(f"Volatility (Annual):  {metrics['annualized_volatility_pct']:>12.2f}%")
        print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>12.2f}")
        print(f"Sortino Ratio:        {metrics['sortino_ratio']:>12.2f}")
        print(f"Calmar Ratio:         {metrics['calmar_ratio']:>12.2f}")
        
        # Trade Statistics
        print(f"\n💹 TRADE STATISTICS")
        print(f"Total Trades:         {metrics['num_trades']:>12}")
        print(f"Completed Rounds:     {metrics['completed_trades']:>12}")
        print(f"Win Rate:             {metrics['win_rate_pct']:>12.2f}%")
        print(f"Profit Factor:        {metrics['profit_factor']:>12.2f}")
        print(f"Avg Trade P&L:        ${metrics['avg_trade_return']:>11.2f}")
        
        print("="*60)