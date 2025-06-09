import yfinance as yf
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
project_root = Path(__file__).resolve().parent.parent.parent.parent
data_dir = project_root / "src" / "data" / "collected"
csv_filename = data_dir / "training_data.csv"
data_dir.mkdir(parents=True, exist_ok=True)

# Helper indicator functions

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    bb_width = upper_band - lower_band
    bb_position = (series - lower_band) / bb_width
    return bb_position, bb_width

def calculate_momentum(series, periods):
    return series.diff(periods)

def calculate_volatility(returns, window=20):
    return returns.rolling(window).std() * np.sqrt(252)

def calculate_support_resistance(series, window=20):
    # Simple support/resistance as min/max in rolling window relative to current close
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()
    support_distance = series - rolling_min
    resistance_distance = rolling_max - series
    return support_distance, resistance_distance

def calculate_volume_ratio(volume, window=20):
    return volume / volume.rolling(window).mean()

class TrainingDataScraper:
    def __init__(self):
        self.target_stocks = [
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

    def get_stock_data(self, symbol):
        try:
            logger.info(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5y", interval="1d")
            
            if hist.empty or len(hist) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(hist)} points")
                return None

            df = hist.copy()
            df['symbol'] = symbol

            # Basic returns & log returns for volatility
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # SMAs and EMAs
            df['sma_5'] = df['Close'].rolling(window=5).mean()
            df['sma_10'] = df['Close'].rolling(window=10).mean()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()

            df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()

            # Price to SMA ratios
            df['price_to_sma20'] = df['Close'] / df['sma_20']
            df['price_to_sma50'] = df['Close'] / df['sma_50']
            df['sma20_to_sma50'] = df['sma_20'] / df['sma_50']

            # RSI
            df['rsi'] = calculate_rsi(df['Close'])

            # MACD
            macd_line, macd_signal, macd_hist = calculate_macd(df['Close'])
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist

            # Bollinger Bands
            bb_pos, bb_width = calculate_bollinger_bands(df['Close'])
            df['bb_position'] = bb_pos
            df['bb_width'] = bb_width

            # Volume ratio
            df['volume_ratio'] = calculate_volume_ratio(df['Volume'])

            # Momentum
            df['momentum_5'] = calculate_momentum(df['Close'], 5)
            df['momentum_10'] = calculate_momentum(df['Close'], 10)
            df['momentum_20'] = calculate_momentum(df['Close'], 20)

            # Volatility (annualized)
            df['volatility'] = calculate_volatility(df['returns'])

            # Support and resistance distances
            support_dist, resistance_dist = calculate_support_resistance(df['Close'])
            df['support_distance'] = support_dist
            df['resistance_distance'] = resistance_dist

            # Drop rows with NaNs caused by rolling windows
            df.dropna(inplace=True)

            # Add timestamp column for consistency
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'timestamp'}, inplace=True)

            # Select and reorder columns for training
            cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20',
                    'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
                    'rsi', 'macd', 'macd_signal', 'macd_histogram',
                    'bb_position', 'bb_width', 'volume_ratio',
                    'momentum_5', 'momentum_10', 'momentum_20', 'volatility',
                    'support_distance', 'resistance_distance']

            return df[cols]

        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            return None

    def scrape_all(self, max_workers=5):
        all_dfs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_stock_data, sym): sym for sym in self.target_stocks}

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        all_dfs.append(data)
                        logger.info(f"Data collected for {sym} with {len(data)} rows")
                    else:
                        logger.warning(f"No data for {sym}")
                except Exception as e:
                    logger.error(f"Error retrieving data for {sym}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def save_data(self, df):
        if df.empty:
            logger.warning("No data to save.")
            return

        df.to_csv(csv_filename, index=False)
        logger.info(f"Training data saved to {csv_filename}")


def main():
    scraper = TrainingDataScraper()
    df = scraper.scrape_all(max_workers=5)
    if not df.empty:
        scraper.save_data(df)
        print(f"Scraped data shape: {df.shape}")
        print(df.head())
    else:
        print("No data scraped.")

if __name__ == "__main__":
    main()
