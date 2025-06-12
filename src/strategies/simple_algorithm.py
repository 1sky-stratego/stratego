import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
import yfinance as yf

# Path and env setup
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated parameters for better performance
param_1 = float(os.getenv('PARAM_1', 0.1))  # Reduced for less extreme sigmoid
param_2 = float(os.getenv('PARAM_2', 0.5))  # Reduced for less extreme sigmoid
buy_threshold = float(os.getenv('BUY_THRESHOLD', 0.6))  # Separate thresholds
sell_threshold = float(os.getenv('SELL_THRESHOLD', 0.4))  # for clearer logic

# Optimizable weight parameters
price_weight = float(os.getenv('PRICE_WEIGHT', 0.4))        # Weight for price momentum
trend_weight = float(os.getenv('TREND_WEIGHT', 0.3))        # Weight for trend direction
volume_weight = float(os.getenv('VOLUME_WEIGHT', 0.2))      # Weight for volume confirmation
volume_price_weight = float(os.getenv('VOLUME_PRICE_WEIGHT', 0.1))  # Weight for volume-price relationship


def get_stock_data(symbol, years=3):
    """
    Fetch stock data for the specified symbol and time period.
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    years : int, default=3
        Number of years of historical data to fetch
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with daily stock data including OHLCV and indicators
    
    Raises:
    -------
    ValueError
        If the symbol is invalid or no data is found
    """
    
    try:
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365 + 30)  # Add buffer for weekends/holidays
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        data = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        # Check if data was retrieved
        if data.empty:
            raise ValueError(f"No data found for symbol '{symbol}'. Please check if the symbol is valid.")
        
        # Clean up column names (remove any extra spaces)
        data.columns = data.columns.str.strip()
        
        # Calculate technical indicators
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        
        # Volume indicators
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        
        # Price momentum indicators
        data['Price_vs_MA20'] = (data['Close'] - data['MA_20']) / data['MA_20'] * 100
        data['Price_Change_5d'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5) * 100
        data['Price_Change_1d'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100
        
        # Price direction indicators
        data['MA5_vs_MA20'] = (data['MA_5'] - data['MA_20']) / data['MA_20'] * 100
        data['Price_Trend'] = np.where(data['MA_5'] > data['MA_20'], 1, -1)  # 1 for uptrend, -1 for downtrend
        
        # Volume-Price relationship
        data['Volume_Price_Signal'] = np.where(
            (data['Price_Change_1d'] > 0) & (data['Volume_Ratio'] > 1), 1,  # High volume + price up = bullish
            np.where(
                (data['Price_Change_1d'] < 0) & (data['Volume_Ratio'] > 1), -1,  # High volume + price down = bearish
                0  # Normal volume = neutral
            )
        )
        
        # Add metadata
        data.attrs['symbol'] = symbol
        data.attrs['start_date'] = start_date.strftime('%Y-%m-%d')
        data.attrs['end_date'] = end_date.strftime('%Y-%m-%d')
        data.attrs['total_days'] = len(data)
        
        # Sort by date (most recent last)
        data = data.sort_index()
        
        print(f"Successfully fetched {len(data)} days of data for {symbol}")
        print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        raise


def improved_normalize(value, center=0, scale=10):
    """
    Improved normalization using tanh for smoother gradients and better behavior.
    
    Parameters:
    -----------
    value : float
        The value to normalize
    center : float, default=0
        Center point for normalization
    scale : float, default=10
        Scale factor for normalization sensitivity
    
    Returns:
    --------
    float
        Normalized value between 0 and 1
    """
    # Use tanh for smoother gradients than sigmoid
    # tanh outputs between -1 and 1, so we scale to 0-1
    normalized = (np.tanh((value - center) / scale) + 1) / 2
    return normalized


def predict(symbol, custom_weights=None):
    """
    Improved prediction function that considers price direction and uses better normalization.
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    custom_weights : dict, optional
        Custom weights for prediction components. Keys: 'price', 'trend', 'volume', 'volume_price'
        If None, uses global parameter values.
    
    Returns:
    --------
    float
        Prediction score between 0 and 1 (0.5 = neutral)
    """
    df = get_stock_data(symbol)
    
    # Handle missing data with neutral values
    if pd.isna(df['Price_vs_MA20'].iloc[-1]):
        price_vs_ma = 0.0
    else:
        price_vs_ma = df['Price_vs_MA20'].iloc[-1]
    
    if pd.isna(df['Volume_Ratio'].iloc[-1]):
        volume_ratio = 1.0
    else:
        volume_ratio = df['Volume_Ratio'].iloc[-1]
    
    if pd.isna(df['MA5_vs_MA20'].iloc[-1]):
        trend_signal = 0.0
    else:
        trend_signal = df['MA5_vs_MA20'].iloc[-1]
    
    if pd.isna(df['Volume_Price_Signal'].iloc[-1]):
        volume_price_signal = 0.0
    else:
        volume_price_signal = df['Volume_Price_Signal'].iloc[-1]
    
    # Improved normalization with better scaling
    # Price relative to MA20 (typically ranges from -20% to +20%)
    normalized_price = improved_normalize(price_vs_ma, center=0, scale=5)
    
    # Volume ratio (typically ranges from 0.5 to 3.0)
    normalized_volume = improved_normalize(volume_ratio, center=1, scale=0.5)
    
    # Trend signal (MA5 vs MA20, typically ranges from -5% to +5%)
    normalized_trend = improved_normalize(trend_signal, center=0, scale=2)
    
    # Volume-price signal is already -1, 0, or 1, so just normalize to 0-1
    normalized_volume_price = (volume_price_signal + 1) / 2
    
    # Use custom weights if provided, otherwise use global parameters
    if custom_weights:
        w_price = custom_weights.get('price', price_weight)
        w_trend = custom_weights.get('trend', trend_weight)
        w_volume = custom_weights.get('volume', volume_weight)
        w_volume_price = custom_weights.get('volume_price', volume_price_weight)
    else:
        w_price = price_weight
        w_trend = trend_weight
        w_volume = volume_weight
        w_volume_price = volume_price_weight
    
    # Normalize weights to sum to 1 (important for optimization)
    total_weight = w_price + w_trend + w_volume + w_volume_price
    if total_weight > 0:
        w_price /= total_weight
        w_trend /= total_weight
        w_volume /= total_weight
        w_volume_price /= total_weight
    else:
        # Fallback to equal weights if all weights are 0
        w_price = w_trend = w_volume = w_volume_price = 0.25
    
    # Combine signals with normalized weights
    prediction = (
        w_price * normalized_price +
        w_trend * normalized_trend +
        w_volume * normalized_volume +
        w_volume_price * normalized_volume_price
    )
    
    # Apply final sigmoid with learned parameters for fine-tuning
    final_prediction = 1 / (1 + np.exp(-param_1 * (prediction - 0.5)))
    
    return final_prediction


def execute(symbol, custom_weights=None):
    """
    Improved execution function with fixed threshold logic.
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    custom_weights : dict, optional
        Custom weights for prediction components
    
    Returns:
    --------
    str
        Decision: 'BUY', 'SELL', or 'HOLD'
    """
    score = predict(symbol, custom_weights)
    
    # Fixed threshold logic - now makes sense!
    if score >= buy_threshold:
        decision = 'BUY'
    elif score <= sell_threshold:
        decision = 'SELL'
    else:
        decision = 'HOLD'
    
    print(f"{symbol}: {decision} (score: {score:.4f})")
    return decision


def analyze_prediction(symbol, custom_weights=None):
    """
    Detailed analysis of prediction components for debugging and understanding.
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    custom_weights : dict, optional
        Custom weights for prediction components
    
    Returns:
    --------
    dict
        Dictionary with detailed analysis
    """
    df = get_stock_data(symbol)
    
    # Get latest values
    latest = df.iloc[-1]
    
    analysis = {
        'symbol': symbol,
        'close_price': latest['Close'],
        'price_vs_ma20': latest.get('Price_vs_MA20', 0),
        'volume_ratio': latest.get('Volume_Ratio', 1),
        'ma5_vs_ma20': latest.get('MA5_vs_MA20', 0),
        'volume_price_signal': latest.get('Volume_Price_Signal', 0),
        'price_trend': latest.get('Price_Trend', 0),
        'prediction_score': predict(symbol, custom_weights),
        'weights_used': custom_weights if custom_weights else {
            'price': price_weight,
            'trend': trend_weight,
            'volume': volume_weight,
            'volume_price': volume_price_weight
        }
    }
    
    return analysis


def optimize_weights_grid_search(symbols, weight_ranges=None, other_params=None):
    """
    Perform grid search optimization for weight parameters.
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols to test on
    weight_ranges : dict, optional
        Ranges for each weight parameter. Keys: 'price', 'trend', 'volume', 'volume_price'
        Each value should be a list/array of values to test
    other_params : dict, optional
        Other parameters to test (param_1, param_2, buy_threshold, sell_threshold)
    
    Returns:
    --------
    pandas.DataFrame
        Results of grid search with performance metrics
    """
    if weight_ranges is None:
        weight_ranges = {
            'price': np.arange(0.1, 0.8, 0.1),
            'trend': np.arange(0.1, 0.8, 0.1),
            'volume': np.arange(0.0, 0.6, 0.1),
            'volume_price': np.arange(0.0, 0.4, 0.1)
        }
    
    results = []
    total_combinations = (len(weight_ranges['price']) * 
                         len(weight_ranges['trend']) * 
                         len(weight_ranges['volume']) * 
                         len(weight_ranges['volume_price']))
    
    print(f"Testing {total_combinations} weight combinations on {len(symbols)} symbols...")
    
    combination_count = 0
    for w_price in weight_ranges['price']:
        for w_trend in weight_ranges['trend']:
            for w_volume in weight_ranges['volume']:
                for w_volume_price in weight_ranges['volume_price']:
                    
                    combination_count += 1
                    if combination_count % 50 == 0:
                        print(f"Progress: {combination_count}/{total_combinations}")
                    
                    # Create weight configuration
                    weights = {
                        'price': w_price,
                        'trend': w_trend,
                        'volume': w_volume,
                        'volume_price': w_volume_price
                    }
                    
                    # Test on all symbols
                    predictions = []
                    scores = []
                    
                    for symbol in symbols:
                        try:
                            score = predict(symbol, weights)
                            scores.append(score)
                            
                            if score >= buy_threshold:
                                predictions.append('BUY')
                            elif score <= sell_threshold:
                                predictions.append('SELL')
                            else:
                                predictions.append('HOLD')
                        except:
                            continue
                    
                    if len(predictions) > 0:
                        # Calculate metrics
                        buy_count = predictions.count('BUY')
                        sell_count = predictions.count('SELL')
                        hold_count = predictions.count('HOLD')
                        
                        # Add weight configuration and results
                        results.append({
                            'price_weight': w_price,
                            'trend_weight': w_trend,
                            'volume_weight': w_volume,
                            'volume_price_weight': w_volume_price,
                            'total_weight': w_price + w_trend + w_volume + w_volume_price,
                            'avg_score': np.mean(scores),
                            'score_std': np.std(scores),
                            'buy_count': buy_count,
                            'sell_count': sell_count,
                            'hold_count': hold_count,
                            'total_predictions': len(predictions),
                            'buy_ratio': buy_count / len(predictions),
                            'sell_ratio': sell_count / len(predictions),
                            'hold_ratio': hold_count / len(predictions)
                        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by interesting metrics
    print("\nTop 10 by Average Score:")
    print(results_df.nlargest(10, 'avg_score')[['price_weight', 'trend_weight', 'volume_weight', 
                                                'volume_price_weight', 'avg_score', 'buy_ratio', 'sell_ratio']])
    
    print("\nTop 10 by Balanced Predictions (least HOLD-heavy):")
    print(results_df.nsmallest(10, 'hold_ratio')[['price_weight', 'trend_weight', 'volume_weight', 
                                                  'volume_price_weight', 'hold_ratio', 'buy_ratio', 'sell_ratio']])
    
    return results_df


def main():
    """Main function for testing the improved prediction system."""
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print("Improved Stock Prediction Testing")
    print("=" * 50)
    print(f"Current Weights: Price={price_weight}, Trend={trend_weight}, Volume={volume_weight}, Volume-Price={volume_price_weight}")
    print(f"Buy Threshold: {buy_threshold}")
    print(f"Sell Threshold: {sell_threshold}")
    print("=" * 50)
    
    results = []
    
    for symbol in test_symbols:
        try:
            decision = execute(symbol)
            analysis = analyze_prediction(symbol)
            results.append(analysis)
            
            # Print detailed analysis
            print(f"\nDetailed Analysis for {symbol}:")
            print(f"  Price vs MA20: {analysis['price_vs_ma20']:.2f}%")
            print(f"  Volume Ratio: {analysis['volume_ratio']:.2f}")
            print(f"  MA5 vs MA20: {analysis['ma5_vs_ma20']:.2f}%")
            print(f"  Volume-Price Signal: {analysis['volume_price_signal']}")
            print(f"  Final Score: {analysis['prediction_score']:.4f}")
            print(f"  Weights: {analysis['weights_used']}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error predicting {symbol}: {str(e)}")
    
    # Summary
    decisions = [execute(r['symbol']) for r in results]
    buy_count = decisions.count('BUY')
    sell_count = decisions.count('SELL')
    hold_count = decisions.count('HOLD')
    
    print(f"\nSummary:")
    print(f"BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")
    
    # Example weight optimization
    print(f"\nTo optimize weights, run:")
    print(f"results_df = optimize_weights_grid_search({test_symbols})")


if __name__ == "__main__":
    main()