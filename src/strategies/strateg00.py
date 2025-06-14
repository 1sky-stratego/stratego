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

# Get Paramaters from .env
param_1 = float(os.getenv('PARAM_1', 0.1))  # Controls 
param_2 = float(os.getenv('PARAM_2', 0.5))  # Controls
buy_threshold = float(os.getenv('BUY_THRESHOLD', 0.6))  # Controls how often a buy order is placed, lower is more frequent
sell_threshold = float(os.getenv('SELL_THRESHOLD', 0.4))  # Controls how often a sell order is placed, higher is more frequent

# Optimizable weight parameters
price_weight = float(os.getenv('PRICE_WEIGHT', 0.4)) # Controls how much the current price of the stock effects the decision
trend_weight = float(os.getenv('TREND_WEIGHT', 0.3)) # Controls how much the current trend of the stock effects the decision
volume_weight = float(os.getenv('VOLUME_WEIGHT', 0.2)) # Controls how much the current volume of the stock effects the decision

# Scrapes using yfinance to get a general view of the current market in the sector over the last year and returns sentiments 
# More weight is placed on more recent dates:
# 6+ mo ago : 20%
# 3-6 mo ago : 25%  
# 1-3 mo ago : 30%
# <1 mo ago : 25%
# over_trend : [0 - 1] overall trend; 0 is a bear market, 1 is a bull market
# volume : [float] normalized average volume of a stock in the market
# volatility : [0 - 1] volatility of the market; 0 is least volatile
# 


# XLE is the Energy sector, XLK is the Tech sector
# Get a list of tickers to scrape

# Get the environment variable
target_stocks_str = os.getenv("TARGET_STOCKS", "")

# Convert it into a list
#tickers = [ticker.strip() for ticker in target_stocks_str.split(",") if ticker.strip()]

# These are for testing
tickers = ["NVDA","AMD","INTC","QCOM"]

# Weights based on recency
now = datetime.now() # current time
start = now - timedelta(days=365 + 30)


windows = { 
    '> 6 mo': (now - timedelta(days=365), now - timedelta(days=180)),
    '3 - 6 mo': (now - timedelta(days=180), now - timedelta(days=90)),
    '1 - 3 mo': (now - timedelta(days=90), now - timedelta(days=30)),
    '< 1 mo': (now - timedelta(days=30), now - timedelta(days=1))
}

weights = {
    '> 6 mo': 0.2,
    '3 - 6 mo': 0.25,
    '1 - 3 mo': 0.3,
    '< 1 mo': 0.25
}

def get_market():

    # Download data
    data = yf.download(tickers, period='1y', interval='1d', group_by='ticker', auto_adjust=True)
   
    # Set up data processing

    trends = []
    volumes = [] 
    volatilities = []

    for ticker in tickers:
        try:
            df = data[ticker]
            df = df.dropna()
            if df.empty or 'Close' not in df or 'Volume' not in df:
                continue

            # Volume calculation
            volumes.append(df['Volume'].mean())
            
            # Calculate volatility
            log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            volatility = np.sqrt(((log_returns - log_returns.mean()) ** 2).sum() / (len(log_returns) - 1))

            volatilities.append(volatility)

            # Weight trend   
            total_trend = 0

            for label, (start, end) in windows.items():
                w = weights[label]
                segment = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
                if len(segment) > 1:
                    # Trend calculation
                    trend = (segment['Close'].iloc[-1] - segment['Close'].iloc[0]) / segment['Close'].iloc[0]
                    total_trend += w * trend


            trends.append(total_trend)

        except Exception as e:
            continue

    # normalize
    over_trend = np.clip(np.mean(trends), -1, 1)
    over_trend = (over_trend + 1) / 2

    volume = np.mean(volumes)
    volume = volume / max(volumes)

    volatility = np.mean(volatilities)
    volatility = volatility / max(volatilities)

    return{
        "over_trend": round(over_trend,3),
        "volume": round(volume, 3),
        "volatility": round(volatility,3),
        "max volume for norm": max(volumes),
        "max volatility for norm": max(volatilities)
    }

    
def stock_vs_market(ticker, sentiment):

    over_trend = sentiment["over_trend"]
    volume = sentiment["volume"]
    volatility = sentiment["volatility"]
    volu_norm = sentiment["max volume for norm"]
    vola_norm = sentiment["max volatility for norm"]


    # Get data for the stock
    df= yf.download(ticker, period='1y', interval='1d', group_by='ticker', auto_adjust=True)
    df = df.dropna()
    df.columns = df.columns.droplevel(0)


    # Volume & volatility calculation
    stock_volu = 0
    stock_vola = 0
    stock_trend = 0

    for label, (start, end) in windows.items():
        w = weights[label]
        segment = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

        if len(segment) > 1:
            # Weighted calculation

            stock_volu += segment['Volume'].mean() * w

            log_returns = np.log(segment['Close'] / segment['Close'].shift(1)).replace([np.inf, -np.inf], np.nan).dropna()

            stock_vola += np.sqrt(((log_returns - log_returns.mean()) ** 2).sum() / (len(log_returns) - 1)) * w

            trend = (segment['Close'].iloc[-1] - segment['Close'].iloc[0]) / segment['Close'].iloc[0]
            stock_trend += w * trend

    stock_volu = stock_volu / volu_norm
    stock_vola = stock_vola / vola_norm

    if over_trend + stock_trend < sell_threshold:
        return 'SELL'
    
    if over_trend + stock_trend > buy_threshold:
        return 'BUY'
    
    else:
        return 'HOLD'


    



# Testing statement

market = get_market()

print(stock_vs_market('AAPL', market))