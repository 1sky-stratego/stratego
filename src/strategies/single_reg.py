import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
import yfinance as yf

# Your existing setup code remains the same...
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv()

MODEL_DIR = 'models_single'
os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stocks
target_stocks_str = os.getenv('TARGET_STOCKS', '')
target_stocks = [t.strip() for t in target_stocks_str.split(',') if t.strip()] if target_stocks_str else []
predict_ticker = os.getenv('PREDICT_TICKER', None)

# Parameters
PARAM_prediction_horizon = int(os.getenv('PARAM_PREDICTION_HORIZON', 5))
PARAM_confidence_threshold = float(os.getenv('PARAM_CONFIDENCE_THRESHOLD', 0.02))
PARAM_min_data_points = int(os.getenv('PARAM_MIN_DATA_POINTS', 100))

# Your existing FEATURE_COLUMNS and calculate_indicators function remain the same...
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_5', 'ema_10', 'ema_20',
    'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
    'rsi',
    'macd', 'macd_signal', 'macd_histogram',
    'bb_position', 'bb_width',
    'volume_ratio',
    'momentum_5', 'momentum_10', 'momentum_20',
    'volatility',
    'support_distance', 'resistance_distance'
]

def calculate_indicators(df):
    """Your existing calculate_indicators function - keeping it the same"""
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if 'Close' in df.columns:
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                               'Close': 'close', 'Volume': 'volume'})
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # All your indicator calculations...
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    df['price_to_sma20'] = df['close'] / df['sma_20'].replace(0, np.nan)
    df['price_to_sma50'] = df['close'] / df['sma_50'].replace(0, np.nan)
    df['sma20_to_sma50'] = df['sma_20'] / df['sma_50'].replace(0, np.nan)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    df['bb_position'] = np.where(bb_range != 0, 
                                (df['close'] - bb_lower) / bb_range, 
                                0.5)
    df['bb_width'] = bb_range / bb_mid.replace(0, np.nan)
    
    # Volume ratio
    avg_volume = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / avg_volume.replace(0, np.nan)
    
    # Momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    
    # Volatility
    returns = df['close'].pct_change()
    df['volatility'] = returns.rolling(20).std()
    
    # Support/Resistance
    rolling_min = df['low'].rolling(20).min()
    rolling_max = df['high'].rolling(20).max()
    df['support_distance'] = (df['close'] - rolling_min) / rolling_min.replace(0, np.nan)
    df['resistance_distance'] = (rolling_max - df['close']) / rolling_max.replace(0, np.nan)
    
    return df

def walk_forward_backtest(tickers, train_start, train_end, test_start, test_end, 
                         rebalance_freq='monthly', transaction_cost=0.001):
    """
    Proper walk-forward backtesting with realistic constraints
    
    Args:
        tickers: List of stock symbols to test
        train_start, train_end: Training period dates
        test_start, test_end: Testing period dates
        rebalance_freq: How often to retrain model ('monthly', 'quarterly')
        transaction_cost: Cost per trade (0.001 = 0.1%)
    """
    
    # Download data for all tickers
    all_data = {}
    for ticker in tickers:
        logger.info(f"Downloading data for {ticker}")
        # Get extra data for indicators
        extended_start = pd.to_datetime(train_start) - timedelta(days=100)
        df = yf.download(ticker, start=extended_start, end=test_end, 
                        interval='1d', progress=False, auto_adjust=True)
        
        if df.empty:
            logger.warning(f"No data for {ticker}")
            continue
            
        df = calculate_indicators(df)
        df['ticker'] = ticker
        all_data[ticker] = df
    
    if not all_data:
        logger.error("No data available for backtesting")
        return
    
    # Create rebalancing dates
    rebalance_dates = pd.date_range(start=test_start, end=test_end, 
                                   freq='MS' if rebalance_freq == 'monthly' else 'QS')
    
    # Initialize portfolio tracking
    portfolio_value = 100000  # Starting capital
    positions = {ticker: 0 for ticker in tickers}  # Number of shares held
    cash = portfolio_value
    portfolio_history = []
    transaction_costs = 0
    
    for rebalance_date in rebalance_dates:
        logger.info(f"Rebalancing on {rebalance_date.date()}")
        
        # Train model on data up to rebalance date
        model, scaler = train_model_for_date(all_data, train_start, rebalance_date)
        if model is None:
            continue
        
        # Generate signals for each ticker
        signals = {}
        for ticker in tickers:
            if ticker not in all_data:
                continue
                
            # Get data up to rebalance date (avoid look-ahead bias)
            ticker_data = all_data[ticker][all_data[ticker].index <= rebalance_date]
            
            if len(ticker_data) < 50:  # Need minimum data
                continue
                
            signal = generate_signal(ticker_data, model, scaler)
            signals[ticker] = signal
        
        # Calculate portfolio weights based on signals
        weights = calculate_portfolio_weights(signals)
        
        # Rebalance portfolio
        new_positions, trade_costs = rebalance_portfolio(
            positions, weights, cash, all_data, rebalance_date, transaction_cost
        )
        
        positions = new_positions
        cash -= trade_costs
        transaction_costs += trade_costs
        
        # Calculate portfolio value
        portfolio_value = cash
        for ticker, shares in positions.items():
            if ticker in all_data and rebalance_date in all_data[ticker].index:
                price = all_data[ticker].loc[rebalance_date, 'close']
                portfolio_value += shares * price
        
        portfolio_history.append({
            'date': rebalance_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions': positions.copy(),
            'signals': signals.copy()
        })
    
    # Calculate final performance metrics
    results_df = pd.DataFrame(portfolio_history)
    results_df.set_index('date', inplace=True)
    
    # Calculate benchmark (equal-weighted buy and hold)
    benchmark = calculate_benchmark(all_data, test_start, test_end, tickers)
    
    # Performance analysis
    analyze_performance(results_df, benchmark, transaction_costs)
    
    return results_df, benchmark

def train_model_for_date(all_data, train_start, current_date):
    """Train model using data available up to current_date"""
    train_data = []
    
    for ticker, df in all_data.items():
        # Only use data up to current date for training
        ticker_train = df[(df.index >= pd.to_datetime(train_start)) & 
                         (df.index < current_date)].copy()
        
        if len(ticker_train) < 100:  # Minimum data requirement
            continue
        
        # Create target variable (future return)
        ticker_train['target'] = ticker_train['close'].pct_change(PARAM_prediction_horizon).shift(-PARAM_prediction_horizon)
        ticker_train = ticker_train.dropna(subset=FEATURE_COLUMNS + ['target'])
        
        if len(ticker_train) > 0:
            train_data.append(ticker_train)
    
    if not train_data:
        logger.warning(f"No training data available for {current_date}")
        return None, None
    
    # Combine all training data
    combined_data = pd.concat(train_data)
    X = combined_data[FEATURE_COLUMNS]
    y = combined_data['target']
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

def generate_signal(ticker_data, model, scaler):
    """Generate trading signal for a single ticker"""
    try:
        # Use most recent complete data point
        latest_data = ticker_data[FEATURE_COLUMNS].dropna().iloc[-1:].copy()
        
        if len(latest_data) == 0:
            return 'HOLD'
        
        X_scaled = scaler.transform(latest_data)
        predicted_return = model.predict(X_scaled)[0]
        
        # Convert to signal based on confidence threshold
        if abs(predicted_return) < PARAM_confidence_threshold:
            return 'HOLD'
        elif predicted_return > 0:
            return 'BUY'
        else:
            return 'SELL'
            
    except Exception as e:
        logger.warning(f"Error generating signal: {e}")
        return 'HOLD'

def calculate_portfolio_weights(signals):
    """Convert signals to portfolio weights"""
    weights = {}
    buy_stocks = [ticker for ticker, signal in signals.items() if signal == 'BUY']
    
    if len(buy_stocks) == 0:
        # If no buy signals, hold cash
        return {ticker: 0 for ticker in signals.keys()}
    
    # Equal weight among buy signals
    weight_per_stock = 1.0 / len(buy_stocks)
    
    for ticker in signals.keys():
        if ticker in buy_stocks:
            weights[ticker] = weight_per_stock
        else:
            weights[ticker] = 0
    
    return weights

def rebalance_portfolio(current_positions, target_weights, cash, all_data, date, transaction_cost):
    """Rebalance portfolio to target weights"""
    new_positions = current_positions.copy()
    total_trades_cost = 0
    
    # Calculate current portfolio value
    current_value = cash
    for ticker, shares in current_positions.items():
        if ticker in all_data and date in all_data[ticker].index:
            price = all_data[ticker].loc[date, 'close']
            current_value += shares * price
    
    # Calculate target positions
    for ticker, target_weight in target_weights.items():
        if ticker not in all_data or date not in all_data[ticker].index:
            continue
            
        price = all_data[ticker].loc[date, 'close']
        target_value = current_value * target_weight
        target_shares = int(target_value / price) if price > 0 else 0
        
        current_shares = current_positions.get(ticker, 0)
        shares_to_trade = target_shares - current_shares
        
        if shares_to_trade != 0:
            trade_value = abs(shares_to_trade * price)
            trade_cost = trade_value * transaction_cost
            total_trades_cost += trade_cost
            new_positions[ticker] = target_shares
    
    return new_positions, total_trades_cost

def calculate_benchmark(all_data, start_date, end_date, tickers):
    """Calculate equal-weighted buy and hold benchmark"""
    benchmark_returns = []
    
    for date in pd.date_range(start=start_date, end=end_date, freq='D'):
        if date.weekday() >= 5:  # Skip weekends
            continue
            
        daily_returns = []
        for ticker in tickers:
            if ticker in all_data and date in all_data[ticker].index:
                if date != pd.to_datetime(start_date):
                    prev_date = all_data[ticker].index[all_data[ticker].index < date][-1]
                    if prev_date in all_data[ticker].index:
                        ret = (all_data[ticker].loc[date, 'close'] / 
                              all_data[ticker].loc[prev_date, 'close'] - 1)
                        daily_returns.append(ret)
        
        if daily_returns:
            benchmark_returns.append({
                'date': date,
                'return': np.mean(daily_returns)
            })
    
    benchmark_df = pd.DataFrame(benchmark_returns)
    benchmark_df.set_index('date', inplace=True)
    benchmark_df['cumulative_return'] = (1 + benchmark_df['return']).cumprod()
    
    return benchmark_df

def analyze_performance(results_df, benchmark_df, transaction_costs):
    """Analyze and display performance metrics"""
    if len(results_df) < 2:
        logger.warning("Not enough data for performance analysis")
        return
    
    # Calculate strategy returns
    strategy_returns = results_df['portfolio_value'].pct_change().dropna()
    
    # Align dates with benchmark
    aligned_dates = results_df.index.intersection(benchmark_df.index)
    if len(aligned_dates) == 0:
        logger.warning("No overlapping dates between strategy and benchmark")
        return
    
    strategy_aligned = strategy_returns.loc[aligned_dates]
    benchmark_aligned = benchmark_df.loc[aligned_dates, 'return']
    
    # Performance metrics
    total_return = results_df['portfolio_value'].iloc[-1] / results_df['portfolio_value'].iloc[0] - 1
    benchmark_total = benchmark_df['cumulative_return'].iloc[-1] - 1
    
    annual_return = (1 + total_return) ** (252 / len(results_df)) - 1
    annual_vol = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    max_drawdown = calculate_max_drawdown(results_df['portfolio_value'])
    
    print(f"\n=== BACKTESTING RESULTS ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Benchmark Return: {benchmark_total:.2%}")
    print(f"Excess Return: {total_return - benchmark_total:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Total Transaction Costs: ${transaction_costs:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results_df.index, results_df['portfolio_value'], label='Strategy', linewidth=2)
    
    # Plot benchmark aligned to same start value
    start_value = results_df['portfolio_value'].iloc[0]
    benchmark_values = start_value * benchmark_df['cumulative_return']
    plt.plot(benchmark_df.index, benchmark_values, label='Benchmark', linewidth=2)
    
    plt.title('Portfolio Value Over Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    drawdown = calculate_drawdown_series(results_df['portfolio_value'])
    plt.fill_between(results_df.index, drawdown, 0, alpha=0.3, color='red')
    plt.title('Drawdown')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_max_drawdown(price_series):
    """Calculate maximum drawdown"""
    peak = price_series.expanding().max()
    drawdown = (price_series - peak) / peak
    return drawdown.min()

def calculate_drawdown_series(price_series):
    """Calculate drawdown series for plotting"""
    peak = price_series.expanding().max()
    drawdown = (price_series - peak) / peak * 100
    return drawdown

# Example usage
if __name__ == '__main__':
    # Define your test parameters
    tickers = target_stocks  # Example tickers
    train_start = '2020-01-01'
    train_end = '2022-01-01'
    test_start = '2022-01-01'
    test_end = '2024-01-01'
    
    # Run walk-forward backtest
    results, benchmark = walk_forward_backtest(
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        rebalance_freq='monthly',
        transaction_cost=0.001
    )