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

param_1 = float(os.getenv('PARAM_1', 0.921))
param_2 = float(os.getenv('PARAM_2', 1.532))
threshold = float(os.getenv('DECISION_THRESHOLD', 0.4))


def get_stock_data(symbol, years=3):
    """
    Fetch stock data for the specified symbol and time period.
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    years : int, default=3
        Number of years of historical data to fetch
    include_indicators : bool, default=False
        Whether to include basic technical indicators
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with daily stock data including OHLCV and optional indicators
    
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
        
        # Only add required indicators for the prediction
        # Moving average (20-day)
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        
        # Volume moving average
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        
        # Price relative to moving averages
        data['Price_vs_MA20'] = (data['Close'] - data['MA_20']) / data['MA_20'] * 100
        
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


def predict(symbol):
    df = get_stock_data(symbol)

    if pd.isna(df['Price_vs_MA20'].iloc[-1]):
        price_vs_ma = 0.5
    else:
        price_vs_ma = df['Price_vs_MA20'].iloc[-1]

    normalized_price = 1 / (1 + np.exp(-param_1 * price_vs_ma))

    if pd.isna(df['Volume_Ratio'].iloc[-1]):
        volume_ratio = 0.5
    else:
        volume_ratio = df['Volume_Ratio'].iloc[-1]

    normalized_volume = 1 / (1 + np.exp(-param_2 * volume_ratio))

    prediction = 0.5 + normalized_volume * (normalized_price - 0.5)
    return prediction

def execute(symbol):
    
    score = predict(symbol)

    if score - threshold > 0.5:
        #BUY
        print('BUY')
        return
    
    if score + threshold < 0.5:
        #SELL
        print('SELL')
        return
    
    else:
        print('HOLD')
        return


def main():
    """Main function for testing the prediction system."""
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print("Stock Prediction Testing")
    print("=" * 40)
    
    for symbol in test_symbols:
        try:
            execute(symbol)
        except Exception as e:
            print(f"Error predicting {symbol}: {str(e)}")


if __name__ == "__main__":
    main()