"""
Windows-Compatible Strategy Optimization Script
Automatically tests different parameter combinations and saves results.

Usage: Run this script from C:\vs code\my code\backtester
"""

import subprocess
import json
import csv
import re
import sys
from datetime import datetime

def extract_metrics(output):
    """Extract metrics from backtest output"""
    metrics = {}
    
    # Extract key metrics using regex
    patterns = {
        'total_return': r'Total Return:\s+(-?\d+\.?\d*)%',
        'max_drawdown': r'Max Drawdown:\s+(-?\d+\.?\d*)%',
        'win_rate': r'Win Rate:\s+(-?\d+\.?\d*)%',
        'total_trades': r'Total Trades:\s+(\d+)',
        'sharpe_ratio': r'Sharpe Ratio:\s+(-?\d+\.?\d*)',
        'profit_factor': r'Profit Factor:\s+(-?\d+\.?\d*)',
        'final_value': r'Final Value:\s+\$\s*(-?\d+\.?\d*)'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[metric] = float(match.group(1))
        else:
            metrics[metric] = 0.0
    
    return metrics

def run_backtest(strategy, params, balance=10000, fee_rate=0.001):
    """Run a single backtest with given parameters"""
    params_str = json.dumps(params)
    
    # Use UTF-8 encoding for Windows
    cmd = [
        sys.executable, 'main.py', 
        '--sample-data', 
        '--strategy', strategy,
        '--params', params_str,
        '--balance', str(balance),
        '--fee-rate', str(fee_rate),
        '--no-plot', '--no-save'
    ]
    
    try:
        # Set encoding explicitly for Windows
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30,
            encoding='utf-8',
            errors='replace'  # Replace problematic characters
        )
        
        if result.returncode == 0:
            return extract_metrics(result.stdout)
        else:
            print(f"Error running backtest: {result.stderr[:200]}...")
            return None
    except subprocess.TimeoutExpired:
        print("Backtest timed out")
        return None
    except Exception as e:
        print(f"Exception running backtest: {e}")
        return None

def optimize_ema_strategy(fast_period, slow_period):
    """Optimize risk management and signal filtering for given EMA parameters"""
    
    print(f"Optimizing EMA Strategy: Fast={fast_period}, Slow={slow_period}")
    print("=" * 60)
    
    # Smaller parameter ranges for faster testing
    position_sizes = [0.33, 0.5, 0.75, 1.0]
    signal_strengths = [0.0, 0.005, 0.01, 0.02]
    
    results = []
    total_tests = len(position_sizes) * len(signal_strengths)
    current_test = 0
    
    # Test all combinations
    for pos_size in position_sizes:
        for sig_strength in signal_strengths:
            current_test += 1
            
            params = {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'position_size': pos_size,
                'min_signal_strength': sig_strength
            }
            
            print(f"[{current_test}/{total_tests}] Testing: pos_size={pos_size}, sig_strength={sig_strength}")
            
            metrics = run_backtest('ema_crossover', params)
            
            if metrics:
                result = {
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'position_size': pos_size,
                    'min_signal_strength': sig_strength,
                    **metrics
                }
                results.append(result)
                
                # Show quick results
                print(f"   -> Return: {metrics.get('total_return', 0):.2f}%, "
                      f"Drawdown: {metrics.get('max_drawdown', 0):.2f}%, "
                      f"Trades: {int(metrics.get('total_trades', 0))}")
            else:
                print("   -> Failed to get results")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{fast_period}_{slow_period}_{timestamp}.csv"
    
    if results:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {filename}")
        
        # Show best results
        show_best_results(results)
    else:
        print("\nNo successful results to save.")
    
    return results

def show_best_results(results):
    """Display the best performing combinations"""
    if not results:
        print("No results to display")
        return
    
    print("\nTOP PERFORMING COMBINATIONS:")
    print("=" * 60)
    
    # Filter out results with no trades
    valid_results = [r for r in results if r.get('total_trades', 0) > 0]
    
    if not valid_results:
        print("No combinations generated any trades.")
        print("Recommendation: Try more sensitive EMA periods like 8/21 or 5/15")
        return
    
    # Sort by different criteria
    metrics_to_sort = [
        ('total_return', 'Highest Return'),
        ('sharpe_ratio', 'Best Risk-Adjusted Return'),
        ('profit_factor', 'Best Profit Factor')
    ]
    
    for metric, description in metrics_to_sort:
        # Filter out invalid results
        metric_results = [r for r in valid_results if r.get(metric, 0) > 0]
        if metric_results:
            best = max(metric_results, key=lambda x: x.get(metric, 0))
            
            print(f"\n{description}:")
            print(f"   Position Size: {best['position_size']}")
            print(f"   Signal Strength: {best['min_signal_strength']}")
            print(f"   Total Return: {best.get('total_return', 0):.2f}%")
            print(f"   Max Drawdown: {best.get('max_drawdown', 0):.2f}%")
            print(f"   Win Rate: {best.get('win_rate', 0):.2f}%")
            print(f"   Sharpe Ratio: {best.get('sharpe_ratio', 0):.3f}")
            print(f"   Total Trades: {int(best.get('total_trades', 0))}")
            
            # Generate command to test this combination
            params = {
                'fast_period': best['fast_period'],
                'slow_period': best['slow_period'],
                'position_size': best['position_size'],
                'min_signal_strength': best['min_signal_strength']
            }
            params_str = json.dumps(params)
            print(f"   Command: python main.py --sample-data --params '{params_str}'")

def test_simple_combinations():
    """Test a few simple combinations first"""
    print("Testing simple combinations first...")
    print("=" * 40)
    
    # Test some known good combinations
    test_cases = [
        (5, 15),
        (8, 21), 
        (12, 26)
    ]
    
    for fast, slow in test_cases:
        print(f"\nQuick test: EMA {fast}/{slow}")
        params = {'fast_period': fast, 'slow_period': slow, 'position_size': 1.0, 'min_signal_strength': 0.0}
        metrics = run_backtest('ema_crossover', params)
        
        if metrics:
            print(f"  Return: {metrics.get('total_return', 0):.2f}%, Trades: {int(metrics.get('total_trades', 0))}")
        else:
            print("  Failed")

def main():
    """Main optimization function"""
    print("EMA Strategy Optimization Tool (Windows Compatible)")
    print("=" * 50)
    
    # First, test simple combinations
    test_simple_combinations()
    
    print("\n" + "=" * 50)
    print("Now let's optimize your chosen EMA parameters:")
    
    try:
        fast = input("Fast period (e.g., 8): ").strip()
        slow = input("Slow period (e.g., 21): ").strip()
        
        # Use defaults if empty
        fast = int(fast) if fast else 8
        slow = int(slow) if slow else 21
        
        if fast >= slow:
            print("Error: Fast period must be less than slow period!")
            return
        
        print(f"\nTesting {fast}/{slow} EMA with different risk management...")
        optimize_ema_strategy(fast, slow)
        
    except KeyboardInterrupt:
        print("\nOptimization cancelled by user")
    except ValueError:
        print("Error: Please enter valid numbers")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()