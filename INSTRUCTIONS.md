# 🚀 Crypto Backtesting Framework - Complete Instructions

A modular Python framework for backtesting cryptocurrency trading strategies with realistic trading costs and comprehensive performance metrics.

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation & Setup](#-installation--setup)
3. [Running Your First Backtest](#-running-your-first-backtest)
4. [Understanding the Results](#-understanding-the-results)
5. [Data Requirements](#-data-requirements)
6. [Strategy Development](#-strategy-development)
7. [Configuration Options](#-configuration-options)
8. [Advanced Usage](#-advanced-usage)
9. [Troubleshooting](#-troubleshooting)
10. [Performance Optimization](#-performance-optimization)

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Basic understanding of trading concepts
- 5 minutes of setup time

### One-Minute Test
```bash
# Clone and test immediately
git clone https://github.com/ronnykatyal/backtester.git
cd backtester
pip install -r requirements.txt
python main.py --sample-data
```

You should see trading results and an equity curve plot!

## 🔧 Installation & Setup

### Step 1: Environment Setup
```bash
# Option A: Using virtual environment (recommended)
python -m venv backtester_env
source backtester_env/bin/activate  # On Windows: backtester_env\Scripts\activate
pip install -r requirements.txt

# Option B: Direct installation
pip install pandas numpy matplotlib
```

### Step 2: Directory Structure
```bash
# Ensure these directories exist
mkdir -p data results
```

### Step 3: Verify Installation
```bash
python main.py --sample-data --no-plot --no-save
```

Expected output: Performance summary with trades executed.

## 🏃‍♂️ Running Your First Backtest

### Basic Commands

#### 1. Test with Sample Data
```bash
# Default EMA crossover strategy
python main.py --sample-data

# Custom parameters
python main.py --sample-data --params '{"fast_period": 8, "slow_period": 21}'

# Different balance and fees
python main.py --sample-data --balance 50000 --fee-rate 0.002
```

#### 2. Using Real Data
```bash
# First, add your data file: data/BTCUSDT_1h.csv
python main.py --symbol BTCUSDT --timeframe 1h

# Different symbols/timeframes
python main.py --symbol ETHUSDT --timeframe 4h
python main.py --symbol ADAUSDT --timeframe 1d
```

### Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--symbol` | Trading pair | BTCUSDT | `--symbol ETHUSDT` |
| `--timeframe` | Data timeframe | 1h | `--timeframe 4h` |
| `--strategy` | Strategy name | ema_crossover | `--strategy rsi` |
| `--params` | Strategy parameters (JSON) | {} | `--params '{"fast_period": 10}'` |
| `--balance` | Initial balance | 10000 | `--balance 50000` |
| `--fee-rate` | Trading fee rate | 0.001 | `--fee-rate 0.002` |
| `--slippage` | Slippage rate | 0.0005 | `--slippage 0.001` |
| `--sample-data` | Use generated data | False | `--sample-data` |
| `--no-plot` | Disable plotting | False | `--no-plot` |
| `--no-save` | Disable saving results | False | `--no-save` |

## 📊 Understanding the Results

### Performance Summary Breakdown

```
============================================================
BACKTEST PERFORMANCE SUMMARY
============================================================

📊 BASIC PERFORMANCE
Initial Balance:      $   10,000.00    ← Starting capital
Final Value:          $   10,504.36    ← Ending portfolio value
Total Return:                 5.04%    ← Overall profit/loss %
Annualized Return:            5.04%    ← Yearly return rate

⚠️  RISK METRICS
Max Drawdown:               -19.21%    ← Worst peak-to-valley loss
Volatility (Annual):         19.66%    ← Price fluctuation measure
Sharpe Ratio:                 0.25     ← Risk-adjusted return
Sortino Ratio:                0.23     ← Downside risk-adjusted return
Calmar Ratio:                 0.26     ← Return vs max drawdown

💹 TRADE STATISTICS
Total Trades:                   11     ← Number of buy/sell orders
Completed Rounds:                5     ← Full buy-sell cycles
Win Rate:                    20.00%    ← Percentage of profitable trades
Profit Factor:                1.31     ← Gross profit ÷ gross loss
Avg Trade P&L:        $      73.33    ← Average profit per trade
============================================================
```

### Key Metrics Explained

#### 📈 **Return Metrics**
- **Total Return**: Overall profit/loss percentage
- **Annualized Return**: Expected yearly return based on performance
- **Good**: >10% annually | **Acceptable**: 5-10% | **Poor**: <5%

#### ⚠️ **Risk Metrics**
- **Max Drawdown**: Largest peak-to-valley decline
- **Target**: <20% | **Acceptable**: 20-30% | **High Risk**: >30%
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Good**: >1.0 | **Acceptable**: 0.5-1.0 | **Poor**: <0.5

#### 💹 **Trading Metrics**
- **Win Rate**: Percentage of profitable trades
- **Don't obsess**: Low win rate can still be profitable with good profit factor
- **Profit Factor**: Total gains ÷ total losses
- **Target**: >1.5 | **Acceptable**: 1.2-1.5 | **Poor**: <1.2

### Generated Files

After each backtest, you'll find:
- `results/SYMBOL_STRATEGY_equity.csv` - Portfolio value over time
- `results/SYMBOL_STRATEGY_trades.csv` - Individual trade details
- `results/SYMBOL_STRATEGY_metrics.json` - All performance metrics
- `results/SYMBOL_STRATEGY_equity_curve.png` - Visual equity curve

## 📁 Data Requirements

### CSV Format
Your data files should follow this format:

```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,16500.0,16600.0,16400.0,16550.0,1000.0
2023-01-01 01:00:00,16550.0,16650.0,16500.0,16600.0,1200.0
2023-01-01 02:00:00,16600.0,16700.0,16550.0,16650.0,800.0
```

### File Naming Convention
- Location: `data/` folder
- Format: `{SYMBOL}_{TIMEFRAME}.csv`
- Examples:
  - `data/BTCUSDT_1h.csv`
  - `data/ETHUSDT_4h.csv`
  - `data/ADAUSDT_1d.csv`

### Data Sources
Popular sources for crypto data:
- **Binance API**: Free historical data
- **CoinGecko API**: Multiple exchanges
- **Yahoo Finance**: `yfinance` Python library
- **Alpha Vantage**: Free tier available

### Minimum Data Requirements
- **Minimum bars**: 100 (configurable in `config.py`)
- **Recommended**: 1000+ bars for reliable results
- **Timeframes**: Any interval (1m, 5m, 1h, 4h, 1d, etc.)

## 🎯 Strategy Development

### Adding New Strategies

#### Step 1: Create Strategy File
Create `strategies/your_strategy.py`:

```python
from .ema_crossover import BaseStrategy
import pandas as pd

class YourStrategy(BaseStrategy):
    def __init__(self, params=None):
        default_params = {
            'your_param': 14,
            'position_size': 1.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def generate_signals(self, data):
        df = data.copy()
        
        # Your strategy logic here
        # Example: Simple moving average crossover
        df['sma_short'] = df['close'].rolling(10).mean()
        df['sma_long'] = df['close'].rolling(30).mean()
        
        # Initialize signals
        df['signal'] = 0
        
        # Generate buy/sell signals
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1   # Buy
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1  # Sell
        
        return df
```

#### Step 2: Register Strategy
Add to `strategies/__init__.py`:

```python
from .your_strategy import YourStrategy

AVAILABLE_STRATEGIES = {
    "ema_crossover": EMACrossoverStrategy,
    "your_strategy": YourStrategy,  # Add this line
}
```

#### Step 3: Test Your Strategy
```bash
python main.py --sample-data --strategy your_strategy
```

### Strategy Best Practices

#### Signal Generation Rules
- **Buy Signal**: `signal = 1`
- **Sell Signal**: `signal = -1`
- **Hold/No Action**: `signal = 0`
- **Only trade when position status allows**:
  - Buy when not holding position
  - Sell when holding position

#### Parameter Guidelines
```python
# Good parameter structure
default_params = {
    'period': 14,              # Indicator periods
    'threshold': 0.02,         # Signal thresholds  
    'position_size': 1.0,      # Always include this
    'min_signal_strength': 0.0 # Optional filters
}
```

#### Performance Tips
- Use vectorized pandas operations
- Avoid loops when possible
- Test with small datasets first
- Validate parameters in `__init__`

### Example Strategy Ideas

#### 1. RSI Strategy
```python
# RSI overbought/oversold signals
df['rsi'] = calculate_rsi(df['close'], period=14)
df.loc[df['rsi'] < 30, 'signal'] = 1   # Oversold = Buy
df.loc[df['rsi'] > 70, 'signal'] = -1  # Overbought = Sell
```

#### 2. Bollinger Bands
```python
# Mean reversion strategy
df['bb_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
df['bb_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)
df.loc[df['close'] < df['bb_lower'], 'signal'] = 1   # Buy at lower band
df.loc[df['close'] > df['bb_upper'], 'signal'] = -1  # Sell at upper band
```

#### 3. MACD Strategy
```python
# Trend following with MACD
df['macd'] = df['close'].ewm(12).mean() - df['close'].ewm(26).mean()
df['signal_line'] = df['macd'].ewm(9).mean()
df.loc[df['macd'] > df['signal_line'], 'signal'] = 1   # Buy
df.loc[df['macd'] < df['signal_line'], 'signal'] = -1  # Sell
```

## ⚙️ Configuration Options

### Global Settings (`config.py`)

```python
# Trading Configuration
DEFAULT_INITIAL_BALANCE = 10000.0  # Starting capital
DEFAULT_FEE_RATE = 0.001          # 0.1% fee per trade
DEFAULT_SLIPPAGE = 0.0005         # 0.05% slippage

# Risk Management
RISK_FREE_RATE = 0.02             # 2% for Sharpe calculation
TRADING_DAYS_PER_YEAR = 365       # Crypto trades daily

# Output Settings
PLOT_EQUITY_CURVE = True          # Generate plots
SAVE_RESULTS = True               # Save result files
```

### Strategy Parameters

#### EMA Crossover Options
```python
params = {
    'fast_period': 12,           # Fast EMA period (1-50)
    'slow_period': 26,           # Slow EMA period (fast_period+1 to 200)
    'position_size': 1.0,        # Portfolio fraction to use (0.1-1.0)
    'min_signal_strength': 0.0   # Minimum EMA difference % (0.0-0.05)
}
```

#### Usage Examples
```bash
# Conservative approach
python main.py --sample-data --params '{"position_size": 0.5}'

# Stronger signals only
python main.py --sample-data --params '{"min_signal_strength": 0.02}'

# Different EMA periods
python main.py --sample-data --params '{"fast_period": 5, "slow_period": 15}'
```

## 🚀 Advanced Usage

### Batch Testing Multiple Parameters
```bash
# Create a script for parameter optimization
for fast in 5 8 12; do
  for slow in 15 21 26; do
    echo "Testing EMA $fast/$slow"
    python main.py --sample-data --params "{\"fast_period\": $fast, \"slow_period\": $slow}" --no-plot
  done
done
```

### Custom Data Preprocessing
Modify `data_loader.py` to add your own indicators:

```python
def add_custom_indicators(df):
    # Add your custom technical indicators
    df['custom_indicator'] = your_calculation(df['close'])
    return df
```

### Performance Analysis
```python
# Load saved results for analysis
import pandas as pd
import json

# Load metrics
with open('results/BTCUSDT_ema_crossover_metrics.json', 'r') as f:
    metrics = json.load(f)

# Load trades
trades = pd.read_csv('results/BTCUSDT_ema_crossover_trades.csv')

# Analyze trade distribution
winning_trades = trades[trades['pnl'] > 0]
losing_trades = trades[trades['pnl'] < 0]
```

### Multiple Symbol Testing
```bash
# Test across multiple cryptocurrencies
symbols=("BTCUSDT" "ETHUSDT" "ADAUSDT" "BNBUSDT")
for symbol in "${symbols[@]}"; do
    echo "Testing $symbol"
    python main.py --symbol $symbol --strategy ema_crossover
done
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. "No trades executed" (All returns 0%)
**Cause**: Strategy not generating signals or position sizing issues
**Solutions**:
```bash
# Check with more sensitive parameters
python main.py --sample-data --params '{"fast_period": 5, "slow_period": 15}'

# Verify sample data generation
python main.py --sample-data --log-level DEBUG
```

#### 2. "FileNotFoundError: Data file not found"
**Cause**: Missing data file
**Solutions**:
```bash
# Use sample data for testing
python main.py --sample-data

# Check file location and naming
ls data/BTCUSDT_1h.csv

# Verify CSV format
head data/BTCUSDT_1h.csv
```

#### 3. "ValueError: Fast period must be less than slow period"
**Cause**: Invalid EMA parameters
**Solution**:
```bash
# Ensure fast < slow
python main.py --sample-data --params '{"fast_period": 12, "slow_period": 26}'
```

#### 4. Poor Performance (Low returns, high drawdown)
**Causes & Solutions**:
- **High fees**: Reduce `--fee-rate` or increase `position_size`
- **Bad parameters**: Try different EMA periods
- **Market conditions**: Test on different time periods
- **Over-trading**: Increase `min_signal_strength`

#### 5. Import Errors
**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.7+

# Verify module structure
python -c "import config, engine, metrics"
```

### Debug Mode
```bash
# Enable detailed logging
python main.py --sample-data --log-level DEBUG

# This shows:
# - Signal generation details
# - Trade execution logs  
# - Position sizing calculations
# - Error stack traces
```

### Performance Debugging
```bash
# Test with minimal data
python main.py --sample-data --no-plot --no-save

# Profile memory usage
python -m memory_profiler main.py --sample-data

# Time execution
time python main.py --sample-data
```

## ⚡ Performance Optimization

### Speed Improvements

#### 1. Optimize Data Loading
```python
# In data_loader.py, use efficient pandas operations
df['sma'] = df['close'].rolling(window=20, min_periods=1).mean()  # Faster
# Instead of: df['sma'] = df['close'].apply(lambda x: ...)  # Slower
```

#### 2. Vectorized Operations
```python
# Good: Vectorized operations
df.loc[condition, 'signal'] = 1

# Avoid: Loops
for i in range(len(df)):
    if condition:
        df.iloc[i]['signal'] = 1
```

#### 3. Memory Management
```python
# Process data in chunks for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### Backtesting Speed Tips

1. **Use sample data for development**:
   ```bash
   python main.py --sample-data  # Fast testing
   ```

2. **Disable unnecessary outputs**:
   ```bash
   python main.py --no-plot --no-save  # Faster execution
   ```

3. **Optimize strategy calculations**:
   ```python
   # Cache expensive calculations
   @functools.lru_cache(maxsize=128)
   def expensive_calculation(data):
       return result
   ```

4. **Profile your strategy**:
   ```bash
   python -m cProfile main.py --sample-data
   ```

## 📈 Next Steps

### Immediate Improvements
1. **Add real data**: Download historical crypto data
2. **Test different strategies**: RSI, MACD, Bollinger Bands
3. **Optimize parameters**: Find best EMA periods for your data
4. **Risk management**: Implement stop-losses and position sizing

### Advanced Features to Add
1. **Multi-timeframe analysis**: Combine different timeframes
2. **Portfolio backtesting**: Test multiple assets simultaneously
3. **Walk-forward optimization**: Dynamic parameter adjustment
4. **Real-time trading**: Connect to exchange APIs
5. **Web dashboard**: Visualize results in browser

### Learning Resources
- **Books**: "Algorithmic Trading" by Ernest Chan
- **Courses**: Quantitative Finance courses on Coursera
- **Communities**: QuantConnect, Zipline forums
- **Data sources**: Binance API, Alpha Vantage, Quandl

---

## 🎯 Summary

You now have a complete crypto backtesting framework! Start with:

1. **Test with sample data**: `python main.py --sample-data`
2. **Add real data**: Place CSV files in `data/` folder  
3. **Experiment with parameters**: Use `--params` to optimize
4. **Develop new strategies**: Add to `strategies/` folder
5. **Analyze results**: Check `results/` folder for detailed output

**Happy backtesting!** 📈

---

*For questions or issues, check the troubleshooting section or create an issue on GitHub.*