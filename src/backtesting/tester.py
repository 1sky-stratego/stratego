import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import importlib.util
import sys


# Path Setup
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# File Paths
data_dir = project_root / "src" / "data" / "collected"
csv_data = data_dir / "ai_gpu_energy_stocks.csv"
json_data = data_dir / "ai_gpu_energy_stocks.json"
excel_data = data_dir / "ai_gpu_energy_stocks.xlsx"
summary_data = data_dir / "summary_stats.json"

# Strategy Paths
strategy_dir = project_root / "strategies"
strat1 = strategy_dir / "base_strategy.py"


def load_strategy_from_file(file_path):
    """
    Load strategy class from external Python file
    
    Args:
        file_path: Path to your strategy Python file
    
    Returns:
        Strategy class from the file
    """
    spec = importlib.util.spec_from_file_location("strategy_module", file_path)
    strategy_module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = strategy_module
    spec.loader.exec_module(strategy_module)
    
    # Look for Strategy class in the module
    # Assumes your strategy class inherits from backtesting.Strategy
    for attr_name in dir(strategy_module):
        attr = getattr(strategy_module, attr_name)
        if (isinstance(attr, type) and 
            issubclass(attr, Strategy) and 
            attr != Strategy):
            return attr
    
    raise ValueError(f"No Strategy class found in {file_path}")

# Load your strategy from external file
MyStrategy = load_strategy_from_file(strat1)

def run_backtest(data, strategy_class=MyStrategy):
    """
    Run backtest with your data and strategy
    
    Args:
        data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        strategy_class: Your strategy class
    
    Returns:
        results: Backtest results
        bt: Backtest object for plotting
    """
    
    # Ensure your data has the required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create backtest instance
    bt = Backtest(
        data, 
        strategy_class,
        cash=10000,           # Starting cash
        commission=0.002,     # Commission per trade (0.2%)
        exclusive_orders=True # Only one position at a time
    )
    
    # Run the backtest
    results = bt.run()
    
    return results, bt

def print_results(results):
    """Print formatted backtest results"""
    print("="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(results)
    
    # Extract key metrics
    print(f"\nKey Metrics:")
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {results['Win Rate [%]']:.2f}%")
    print(f"Total Trades: {results['# Trades']}")

def optimize_parameters(data, strategy_class=MyStrategy, **param_ranges):
    """
    Optimize strategy parameters
    
    Args:
        data: Your OHLCV data
        strategy_class: Your strategy class
        **param_ranges: Parameter ranges to optimize
                       e.g., param1=range(5, 20), param2=range(10, 50)
    
    Returns:
        Optimization results
    """
    
    bt = Backtest(data, strategy_class, cash=10000, commission=0.002)
    
    results = bt.optimize(
        maximize='Sharpe Ratio',  # What to optimize for
        **param_ranges
    )
    
    return results

# Example usage
if __name__ == "__main__":
    
    # Your strategy is now loaded from the external file
    print(f"Loaded strategy: {MyStrategy.__name__}")
    
    # Load your data (replace this with your actual data loading)
    # Your data should be a DataFrame with columns: Open, High, Low, Close, Volume
    # and datetime index
    
    # Example data structure:
    # data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
    # OR
    # data = your_data_loading_function()
    
    # For demonstration, create sample data (replace with your actual data)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    price = 100
    prices = []
    
    for _ in dates:
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, len(prices))
    }, index=dates)
    
    print("Sample data shape:", sample_data.shape)
    print("\nFirst few rows:")
    print(sample_data.head())
    
    # Run backtest with your strategy and data
    # results, bt = run_backtest(your_data, MyStrategy)
    results, bt = run_backtest(sample_data, MyStrategy)
    
    # Print results
    print_results(results)
    
    # Plot results (opens in browser/window)
    print("\nGenerating plot...")
    bt.plot()
    
    # Example parameter optimization (uncomment when you have parameters to optimize)
    # optimization_results = optimize_parameters(
    #     your_data,
    #     MyStrategy,
    #     param1=range(5, 20, 2),
    #     param2=range(10, 50, 5)
    # )
    # print("\nOptimization Results:")
    # print(optimization_results)