# 🚀 QUICKSTART - Crypto Backtesting in 2 Minutes

*Get your first backtest running immediately!*

## ⚡ 30-Second Setup

```bash
# 1. Clone and enter
git clone https://github.com/ronnykatyal/backtester.git
cd backtester

# 2. Install requirements  
pip install pandas numpy matplotlib

# 3. Run first backtest
python main.py --sample-data
```

**Done!** You should see trading results and a chart.

## 🎯 Your First Results

You'll see something like this:
```
📊 BASIC PERFORMANCE
Initial Balance:      $   10,000.00
Final Value:          $   10,504.36
Total Return:                 5.04%

💹 TRADE STATISTICS  
Total Trades:                   11
Win Rate:                    20.00%
```

## ⚙️ Try Different Settings

```bash
# Different EMA periods
python main.py --sample-data --params '{"fast_period": 8, "slow_period": 21}'

# Larger starting balance
python main.py --sample-data --balance 50000

# Higher trading fees (more realistic)
python main.py --sample-data --fee-rate 0.002
```

## 📊 Understanding Your Results

| Metric | Good | Okay | Poor |
|--------|------|------|------|
| **Total Return** | >10% | 5-10% | <5% |
| **Max Drawdown** | <15% | 15-25% | >25% |
| **Win Rate** | >40% | 20-40% | <20% |
| **Profit Factor** | >1.5 | 1.2-1.5 | <1.2 |

## 📁 Add Real Data

1. **Get crypto data** (CSV format):
   ```csv
   timestamp,open,high,low,close,volume
   2023-01-01 00:00:00,16500.0,16600.0,16400.0,16550.0,1000.0
   ```

2. **Save as**: `data/BTCUSDT_1h.csv`

3. **Run with real data**:
   ```bash
   python main.py --symbol BTCUSDT
   ```

## 🎯 What's Next?

- **📖 Read full instructions**: [INSTRUCTIONS.md](INSTRUCTIONS.md)
- **🔧 Modify strategies**: Edit `strategies/ema_crossover.py`
- **📈 Add new strategies**: Create files in `strategies/` folder
- **📊 Analyze results**: Check `results/` folder

## 🆘 Quick Fixes

**No trades executed?**
```bash
python main.py --sample-data --params '{"fast_period": 5, "slow_period": 15}'
```

**Import errors?**
```bash
pip install -r requirements.txt
```

**Need help?**
Check [INSTRUCTIONS.md](INSTRUCTIONS.md) troubleshooting section.

---

**🎉 Congratulations!** You're now backtesting crypto strategies. Time to optimize and profit! 📈
