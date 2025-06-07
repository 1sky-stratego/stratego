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
from itertools import product
import warnings
warnings.filterwarnings('ignore')

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

class PortfolioData:
    """Class to handle multi-stock portfolio data"""
    def __init__(self, stock_data_dict):
        self.stock_data = stock_data_dict
        self.symbols = list(stock_data_dict.keys())
        self.combined_data = self._create_combined_dataframe()
        
    def _create_combined_dataframe(self):
        """Create a combined dataframe with all stocks"""
        # Find common date range
        start_dates = [df.index.min() for df in self.stock_data.values()]
        end_dates = [df.index.max() for df in self.stock_data.values()]
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Create combined dataframe
        combined = pd.DataFrame()
        
        for symbol, data in self.stock_data.items():
            # Filter to common date range
            filtered_data = data[(data.index >= common_start) & (data.index <= common_end)].copy()
            
            # Add symbol suffix to columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in filtered_data.columns:
                    combined[f"{col}_{symbol}"] = filtered_data[col]
        
        # Sort by date
        combined = combined.sort_index()
        
        # Forward fill any missing values
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        return combined
    
    def get_symbol_data(self, symbol):
        """Get data for a specific symbol from combined dataframe"""
        if symbol not in self.symbols:
            return None
            
        symbol_df = pd.DataFrame()
        symbol_df['Open'] = self.combined_data[f'Open_{symbol}']
        symbol_df['High'] = self.combined_data[f'High_{symbol}']
        symbol_df['Low'] = self.combined_data[f'Low_{symbol}']
        symbol_df['Close'] = self.combined_data[f'Close_{symbol}']
        symbol_df['Volume'] = self.combined_data[f'Volume_{symbol}']
        
        return symbol_df

class MultiStockStrategy(Strategy):
    """
    Base class for multi-stock portfolio strategies
    Override the trade_logic method to implement your strategy
    """
    
    # Strategy parameters for optimization
    sma_fast = 10
    sma_slow = 30
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    max_positions = 10
    position_size = 0.1  # 10% of portfolio per position
    
    def init(self):
        """Initialize strategy indicators and data structures"""
        self.portfolio = PortfolioData(self.data._stock_data)
        self.symbols = self.portfolio.symbols
        self.positions = {}
        self.indicators = {}
        
        # Initialize indicators for each stock
        for symbol in self.symbols:
            symbol_data = self.portfolio.get_symbol_data(symbol)
            if symbol_data is not None and len(symbol_data) > max(self.sma_slow, self.rsi_period):
                self.indicators[symbol] = {
                    'sma_fast': self.I(lambda x, p=self.sma_fast: pd.Series(x).rolling(p).mean(), 
                                     symbol_data.Close, name=f'SMA_fast_{symbol}'),
                    'sma_slow': self.I(lambda x, p=self.sma_slow: pd.Series(x).rolling(p).mean(), 
                                     symbol_data.Close, name=f'SMA_slow_{symbol}'),
                    'rsi': self.I(lambda x, p=self.rsi_period: self._calculate_rsi(pd.Series(x), p), 
                                symbol_data.Close, name=f'RSI_{symbol}'),
                    'close': symbol_data.Close
                }
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def get_portfolio_value(self):
        """Calculate current portfolio value"""
        cash = self.broker.cash
        positions_value = sum([trade.size * trade.entry_price for trade in self.broker.trades])
        return cash + positions_value
    
    def get_position_size_value(self):
        """Calculate position size in dollars"""
        portfolio_value = self.get_portfolio_value()
        return portfolio_value * self.position_size
    
    def count_open_positions(self):
        """Count number of open positions"""
        return len([trade for trade in self.broker.trades if trade.is_open])
    
    def next(self):
        """Main strategy logic - override this method"""
        self.trade_logic()
    
    def trade_logic(self):
        """
        Default trading logic - override this method in your strategy
        This is a simple SMA crossover with RSI filter
        """
        current_positions = self.count_open_positions()
        
        for symbol in self.symbols:
            if symbol not in self.indicators:
                continue
                
            indicators = self.indicators[symbol]
            
            # Skip if not enough data
            if (len(indicators['sma_fast']) < 2 or 
                len(indicators['sma_slow']) < 2 or 
                len(indicators['rsi']) < 1):
                continue
            
            current_price = indicators['close'][-1]
            sma_fast_current = indicators['sma_fast'][-1]
            sma_fast_prev = indicators['sma_fast'][-2]
            sma_slow_current = indicators['sma_slow'][-1]
            sma_slow_prev = indicators['sma_slow'][-2]
            current_rsi = indicators['rsi'][-1]
            
            # Check if we have a position in this symbol
            has_position = any(trade.is_open and f"_{symbol}" in str(trade) for trade in self.broker.trades)
            
            # Buy signal: SMA crossover up + RSI not overbought + room for more positions
            if (not has_position and 
                current_positions < self.max_positions and
                sma_fast_current > sma_slow_current and 
                sma_fast_prev <= sma_slow_prev and
                current_rsi < self.rsi_overbought):
                
                position_value = self.get_position_size_value()
                size = position_value / current_price
                
                if size > 0 and self.broker.cash >= position_value:
                    self.buy(size=size, tag=f"BUY_{symbol}")
            
            # Sell signal: SMA crossover down or RSI overbought
            elif (has_position and 
                  (sma_fast_current < sma_slow_current and sma_fast_prev >= sma_slow_prev) or
                  current_rsi > self.rsi_overbought):
                
                # Close position for this symbol
                for trade in self.broker.trades:
                    if trade.is_open and f"_{symbol}" in str(trade.tag):
                        trade.close()

def load_strategy_from_file(file_path):
    """Load strategy class from external Python file"""
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
    """Create synthetic time series data from single point data"""
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
    """Load stock data and create synthetic time series for backtesting"""
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

def run_portfolio_backtest(stock_data, strategy_class, selected_symbols=None, **strategy_params):
    """Run backtest on portfolio of multiple stocks"""
    
    # Filter symbols if specified
    if selected_symbols:
        filtered_stock_data = {symbol: data for symbol, data in stock_data.items() 
                             if symbol in selected_symbols}
    else:
        filtered_stock_data = stock_data
    
    if not filtered_stock_data:
        raise ValueError("No valid symbols found for backtesting")
    
    print(f"Running portfolio backtest with {len(filtered_stock_data)} symbols:")
    print(f"Symbols: {list(filtered_stock_data.keys())}")
    
    # Create portfolio data object
    portfolio = PortfolioData(filtered_stock_data)
    
    # Use any stock's data as the base (they all have same date range now)
    base_symbol = list(filtered_stock_data.keys())[0]
    base_data = filtered_stock_data[base_symbol].copy()
    
    # Add stock data to the base data for strategy access
    base_data._stock_data = filtered_stock_data
    
    # Create backtest instance
    bt = Backtest(
        base_data, 
        strategy_class,
        cash=100000,  # Larger cash for portfolio
        commission=0.002,
        exclusive_orders=False  # Allow multiple positions
    )
    
    # Run the backtest with parameters
    results = bt.run(**strategy_params)
    
    return results, bt

def optimize_strategy(stock_data, strategy_class, selected_symbols=None, param_ranges=None, max_tries=100):
    """
    Optimize strategy parameters
    
    Args:
        stock_data: Dictionary of stock data
        strategy_class: Strategy class to optimize
        selected_symbols: List of symbols to trade (None for all)
        param_ranges: Dictionary of parameter ranges for optimization
        max_tries: Maximum number of parameter combinations to try
    
    Returns:
        Best parameters and results
    """
    
    if param_ranges is None:
        # Default parameter ranges
        param_ranges = {
            'sma_fast': [5, 10, 15, 20],
            'sma_slow': [20, 30, 40, 50],
            'rsi_period': [10, 14, 20],
            'rsi_oversold': [20, 30, 40],
            'rsi_overbought': [60, 70, 80],
            'max_positions': [5, 10, 15, 20],
            'position_size': [0.05, 0.1, 0.15, 0.2]
        }
    
    print("Starting parameter optimization...")
    print(f"Parameter ranges: {param_ranges}")
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    all_combinations = list(product(*param_values))
    
    # Limit number of combinations to try
    if len(all_combinations) > max_tries:
        print(f"Too many combinations ({len(all_combinations)}), randomly sampling {max_tries}")
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_tries]
    
    best_params = None
    best_return = -np.inf
    best_results = None
    optimization_results = []
    
    print(f"Testing {len(all_combinations)} parameter combinations...")
    
    for i, param_combo in enumerate(all_combinations):
        try:
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            # Skip invalid combinations
            if params.get('sma_fast', 10) >= params.get('sma_slow', 30):
                continue
            if params.get('rsi_oversold', 30) >= params.get('rsi_overbought', 70):
                continue
            
            # Run backtest with these parameters
            results, bt = run_portfolio_backtest(stock_data, strategy_class, selected_symbols, **params)
            
            # Check if this is the best result so far
            total_return = results['Return [%]']
            
            optimization_results.append({
                'params': params.copy(),
                'return': total_return,
                'sharpe': results['Sharpe Ratio'],
                'max_dd': results['Max. Drawdown [%]'],
                'trades': results['# Trades']
            })
            
            if total_return > best_return:
                best_return = total_return
                best_params = params.copy()
                best_results = results
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(all_combinations)} combinations. Best so far: {best_return:.2f}%")
                
        except Exception as e:
            print(f"Error with parameters {param_combo}: {e}")
            continue
    
    print("\nOptimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best return: {best_return:.2f}%")
    
    # Create results summary
    results_df = pd.DataFrame(optimization_results)
    results_df = results_df.sort_values('return', ascending=False)
    
    return best_params, best_results, results_df

def print_results(results, title="BACKTEST RESULTS"):
    """Print formatted backtest results"""
    print("="*60)
    print(title)
    print("="*60)
    
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

# Load your strategy from external file or use default
try:
    MyStrategy = load_strategy_from_file(strat1)
    print(f"Successfully loaded strategy: {MyStrategy.__name__}")
    
    # Make it inherit from MultiStockStrategy if it doesn't already
    if not issubclass(MyStrategy, MultiStockStrategy):
        print("Converting strategy to multi-stock strategy...")
        
        # Create a wrapper that inherits from MultiStockStrategy
        class WrappedStrategy(MultiStockStrategy):
            pass
        
        # Copy methods from original strategy
        for attr_name in dir(MyStrategy):
            if not attr_name.startswith('_') and callable(getattr(MyStrategy, attr_name)):
                setattr(WrappedStrategy, attr_name, getattr(MyStrategy, attr_name))
        
        MyStrategy = WrappedStrategy
        
except Exception as e:
    print(f"Error loading strategy: {e}")
    print("Using default multi-stock strategy")
    MyStrategy = MultiStockStrategy

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
        
        # Select subset of symbols for faster testing
        test_symbols = tickers
        
        # Run single backtest with default parameters
        print("\n" + "="*60)
        print("RUNNING PORTFOLIO BACKTEST WITH DEFAULT PARAMETERS")
        print("="*60)
        
        results, bt = run_portfolio_backtest(stock_data, MyStrategy, test_symbols)
        print_results(results, "PORTFOLIO BACKTEST - DEFAULT PARAMETERS")
        
        # Run parameter optimization
        print("\n" + "="*60)
        print("RUNNING PARAMETER OPTIMIZATION")
        print("="*60)
        
        # Define parameter ranges for optimization
        param_ranges = {
            'sma_fast': [10, 15, 20],
            'sma_slow': [30, 40, 50],
            'max_positions': [5, 10, 15],
            'position_size': [0.1, 0.15, 0.2]
        }
        
        best_params, best_results, optimization_df = optimize_strategy(
            stock_data, 
            MyStrategy, 
            test_symbols, 
            param_ranges,
            max_tries=10
        )
        
        print_results(best_results, "PORTFOLIO BACKTEST - OPTIMIZED PARAMETERS")
        
        # Show top 10 parameter combinations
        print("\nTop 10 Parameter Combinations:")
        print("="*80)
        top_10 = optimization_df.head(10)
        for idx, row in top_10.iterrows():
            print(f"Return: {row['return']:.2f}% | Sharpe: {row['sharpe']:.2f} | Params: {row['params']}")
        
        # Run final backtest with best parameters and plot
        print(f"\nRunning final backtest with best parameters...")
        final_results, final_bt = run_portfolio_backtest(
            stock_data, MyStrategy, test_symbols, **best_params
        )
        
        try:
            print("Generating plot...")
            final_bt.plot()
        except Exception as e:
            print(f"Error generating plot: {e}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()