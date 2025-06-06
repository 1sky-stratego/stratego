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
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)
csv_data = 'src/backtesting/stock_data/ai_gpu_energy_stocks_2020-01-01_2025-06-06.csv'

# Strategy Paths
strategy_dir = project_root / "strategies"
strat1 = strategy_dir / "base_strategy.py"

# Tickers

tickers = [
            # AI/GPU Leaders
            'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'MRVL', 'XLNX', 'LRCX', 'KLAC', 'AMAT',
            'ASML', 'TSM', 'MU', 'MCHP', 'ADI', 'TXN', 'NXPI', 'SWKS', 'QRVO', 'MPWR',
            
            # AI Software/Cloud
            'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'CRM', 'NOW', 'SNOW', 'PLTR', 'AI',
            'C3AI', 'UPST', 'PATH', 'BBAI', 'SOUN', 'GFAI', 'AITX', 'AGFY', 'VERI', 'DTEA',
            
            # Energy Infrastructure
            'TSLA', 'ENPH', 'SEDG', 'FSLR', 'SPWR', 'RUN', 'NEE', 'AEP', 'EXC', 'D',
            'SO', 'DUK', 'XEL', 'SRE', 'PEG', 'ED', 'EIX', 'PCG', 'AWK', 'CNP',
            
            # Energy Storage/Battery
            'PLUG', 'BE', 'BLDP', 'FCEL', 'BALLARD', 'QS', 'NKLA', 'HYLN', 'CLSK', 'RIOT',
            
            # Data Centers/Infrastructure  
            'DLR', 'PLD', 'AMT', 'CCI', 'SBAC', 'EQIX', 'CONE', 'REIT', 'VTR', 'CORR',
            
            # Additional AI/Tech
            'AAPL', 'IBM', 'CSCO', 'HPQ', 'DELL', 'VMW', 'WORK', 'ZM', 'DDOG', 'CRWD',
            'OKTA', 'SPLK', 'TEAM', 'ATLASSIAN', 'MDB', 'ELASTIC', 'DOCN', 'NET', 'FSLY', 'ESTC'
        ]

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
    for attr_name in dir(strategy_module):
        attr = getattr(strategy_module, attr_name)
        if (isinstance(attr, type) and 
            issubclass(attr, Strategy) and 
            attr != Strategy):
            return attr
    
    raise ValueError(f"No Strategy class found in {file_path}")

def create_synthetic_timeseries(symbol_data, days=252):
    """
    Create synthetic time series data from single point data
    This simulates historical price movement based on the current data point
    """
    row = symbol_data.iloc[0]
    
    # Extract key metrics
    current_price = row['current_price']
    high_52w = row['price_52w_high']
    low_52w = row['price_52w_low']
    ytd_return = row['ytd_return'] / 100  # Convert to decimal
    volatility = row['volatility_annualized'] / 100  # Convert to decimal
    sma_20 = row['sma_20']
    sma_50 = row['sma_50']
    
    # Create date range (going backwards from collection date)
    end_date = pd.to_datetime(row['data_collected_at'])
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic price path
    np.random.seed(hash(row['symbol']) % 2**32)  # Consistent seed per symbol
    
    # Calculate daily volatility
    daily_vol = volatility / np.sqrt(252)
    
    # Generate random returns
    daily_returns = np.random.normal(0, daily_vol, len(date_range))
    
    # Adjust returns to match YTD performance
    # Scale returns so that cumulative return matches YTD
    cumulative_return = np.exp(np.cumsum(daily_returns)) - 1
    scaling_factor = ytd_return / cumulative_return[-1] if cumulative_return[-1] != 0 else 1
    daily_returns = daily_returns * scaling_factor
    
    # Calculate price series
    prices = [current_price]
    for ret in reversed(daily_returns[1:]):  # Work backwards
        prev_price = prices[-1] / (1 + ret)
        prices.append(prev_price)
    
    prices = list(reversed(prices))  # Reverse to get chronological order
    prices = np.array(prices)
    
    # Ensure prices stay within 52-week range (approximately)
    prices = np.clip(prices, low_52w * 0.95, high_52w * 1.05)
    
    # Create OHLCV data
    ohlcv = pd.DataFrame(index=date_range)
    ohlcv['Close'] = prices
    
    # Generate OHLC from Close prices
    daily_vol_factor = daily_vol * 0.5  # Reduce intraday volatility
    
    # Open is previous close (shifted)
    ohlcv['Open'] = ohlcv['Close'].shift(1)
    ohlcv['Open'].iloc[0] = ohlcv['Close'].iloc[0]
    
    # High and Low based on Close and some random variation
    np.random.seed(hash(row['symbol'] + 'hl') % 2**32)
    high_factor = np.random.uniform(0.5, 1.5, len(ohlcv)) * daily_vol_factor
    low_factor = np.random.uniform(0.5, 1.5, len(ohlcv)) * daily_vol_factor
    
    ohlcv['High'] = ohlcv['Close'] + high_factor * ohlcv['Close']
    ohlcv['Low'] = ohlcv['Close'] - low_factor * ohlcv['Close']
    
    # Ensure OHLC relationships are maintained
    ohlcv['High'] = np.maximum(ohlcv['High'], np.maximum(ohlcv['Open'], ohlcv['Close']))
    ohlcv['Low'] = np.minimum(ohlcv['Low'], np.minimum(ohlcv['Open'], ohlcv['Close']))
    
    # Volume (use average volume with some variation)
    np.random.seed(hash(row['symbol'] + 'vol') % 2**32)
    volume_variation = np.random.uniform(0.5, 2.0, len(ohlcv))
    ohlcv['Volume'] = row['avg_volume_30d'] * volume_variation
    
    # Add additional indicators
    ohlcv['SMA_20'] = ohlcv['Close'].rolling(20).mean()
    ohlcv['SMA_50'] = ohlcv['Close'].rolling(50).mean()
    
    # Fill NaN values for SMAs
    ohlcv['SMA_20'].fillna(method='bfill', inplace=True)
    ohlcv['SMA_50'].fillna(method='bfill', inplace=True)
    
    # Add metadata
    ohlcv['YTD_Return'] = ytd_return
    ohlcv['Volatility'] = volatility
    ohlcv['Market_Cap_Category'] = row['market_cap_category']
    ohlcv['Symbol'] = row['symbol']
    
    return ohlcv

def load_and_prepare_data(file_path=csv_data, min_days=100):
    """
    Load stock data and create synthetic time series for backtesting
    """
    print(f"Loading data from: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    
    # Convert data_collected_at to datetime
    df['data_collected_at'] = pd.to_datetime(df['data_collected_at'])
    
    # Group by symbol to create individual stock datasets
    stock_data = {}
    
    print("Creating synthetic time series for each symbol...")
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        if len(symbol_df) > 0:
            try:
                # Create synthetic time series
                ohlcv_data = create_synthetic_timeseries(symbol_df, days=min_days + 52)
                
                if len(ohlcv_data) > min_days:  # Ensure we have enough data
                    stock_data[symbol] = ohlcv_data
                    print(f"  {symbol}: Created {len(ohlcv_data)} days of data")
                else:
                    print(f"  {symbol}: Insufficient data generated")
                    
            except Exception as e:
                print(f"  {symbol}: Error creating data - {e}")
                continue
    
    print(f"Successfully created data for {len(stock_data)} symbols")
    return stock_data

def run_backtest(data, strategy_class, symbol=None):
    """
    Run backtest with your data and strategy
    """
    if isinstance(data, dict):
        if symbol is None:
            symbol = list(data.keys())[0]
        if symbol not in data:
            raise ValueError(f"Symbol {symbol} not found in data")
        backtest_data = data[symbol]
        print(f"Running backtest for symbol: {symbol}")
    else:
        backtest_data = data
        print("Running backtest on provided data")
    
    print(f"Data shape: {backtest_data.shape}")
    print(f"Date range: {backtest_data.index.min()} to {backtest_data.index.max()}")
    
    # Create backtest instance
    bt = Backtest(
        backtest_data, 
        strategy_class,
        cash=10000,
        commission=0.002,
        exclusive_orders=True
    )
    
    # Run the backtest
    results = bt.run()
    
    return results, bt

def run_multi_symbol_backtest(stock_data, strategy_class, symbols=None):
    """
    Run backtests on multiple symbols and compare results
    """
    if symbols is None:
        symbols = list(stock_data.keys())
    
    results_summary = {}
    
    for symbol in symbols:
        if symbol not in stock_data:
            print(f"Warning: Symbol {symbol} not found in data")
            continue
            
        try:
            print(f"\n{'='*20} {symbol} {'='*20}")
            results, bt = run_backtest(stock_data, strategy_class, symbol)
            results_summary[symbol] = {
                'results': results,
                'backtest': bt
            }
            print_results(results, symbol)
            
        except Exception as e:
            print(f"Error backtesting {symbol}: {str(e)}")
            continue
    
    return results_summary

def print_results(results, symbol=None):
    """Print formatted backtest results"""
    title = f"BACKTEST RESULTS"
    if symbol:
        title += f" - {symbol}"
    
    print("="*50)
    print(title)
    print("="*50)
    
    # Extract key metrics
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Buy & Hold Return: {results['Buy & Hold Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {results['Win Rate [%]']:.2f}%")
    print(f"Total Trades: {results['# Trades']}")
    print(f"Start Date: {results['Start']}")
    print(f"End Date: {results['End']}")
    print(f"Duration: {results['Duration']}")

def compare_results(results_summary):
    """Compare results across multiple symbols"""
    if not results_summary:
        print("No results to compare")
        return
    
    print("\n" + "="*80)
    print("MULTI-SYMBOL COMPARISON")
    print("="*80)
    
    comparison_data = []
    for symbol, data in results_summary.items():
        results = data['results']
        comparison_data.append({
            'Symbol': symbol,
            'Return [%]': results['Return [%]'],
            'Buy&Hold [%]': results['Buy & Hold Return [%]'],
            'Sharpe': results['Sharpe Ratio'],
            'Max DD [%]': results['Max. Drawdown [%]'],
            'Win Rate [%]': results['Win Rate [%]'],
            'Trades': results['# Trades']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Return [%]', ascending=False)
    
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    
    # Best performers
    if len(comparison_df) > 0:
        print(f"\nBest Total Return: {comparison_df.iloc[0]['Symbol']} ({comparison_df.iloc[0]['Return [%]']:.2f}%)")
        best_sharpe_idx = comparison_df['Sharpe'].idxmax()
        print(f"Best Sharpe Ratio: {comparison_df.loc[best_sharpe_idx, 'Symbol']} ({comparison_df.loc[best_sharpe_idx, 'Sharpe']:.2f})")
    
    return comparison_df

# Load your strategy from external file
try:
    MyStrategy = load_strategy_from_file(strat1)
    print(f"Successfully loaded strategy: {MyStrategy.__name__}")
except Exception as e:
    print(f"Error loading strategy: {e}")
    print("Using a simple example strategy instead")
    
    # Fallback simple strategy
    class SimpleStrategy(Strategy):
        def init(self):
            # Simple moving average crossover
            self.sma20 = self.I(lambda x: pd.Series(x).rolling(20).mean(), self.data.Close)
            self.sma50 = self.I(lambda x: pd.Series(x).rolling(50).mean(), self.data.Close)
        
        def next(self):
            if self.sma20[-1] > self.sma50[-1] and self.sma20[-2] <= self.sma50[-2]:
                self.buy()
            elif self.sma20[-1] < self.sma50[-1] and self.sma20[-2] >= self.sma50[-2]:
                self.sell()
    
    MyStrategy = SimpleStrategy

# Example usage
if __name__ == "__main__":
    try:
        # Load your actual stock data
        print("Loading stock data...")
        stock_data = load_and_prepare_data(csv_data)
        
        if not stock_data:
            print("No data loaded. Please check your data file.")
            sys.exit(1)
        
        print(f"Loaded data for {len(stock_data)} symbols")
        
        # Get list of symbols
        print(f"Available symbols: {tickers}...")
        
        # Run backtest on single symbol
        if tickers:
            print(f"\nRunning single symbol backtest...")
            first_symbol = tickers[0]
            results, bt = run_backtest(stock_data, MyStrategy, first_symbol)
            print_results(results, first_symbol)
            
            # Plot results
            print(f"\nGenerating plot for {first_symbol}...")
            try:
                bt.plot()
            except Exception as e:
                print(f"Error generating plot: {e}")
        
        
        
        multi_results = run_multi_symbol_backtest(stock_data, MyStrategy, tickers)
        
        # Compare results
        if multi_results:
            comparison_df = compare_results(multi_results)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()