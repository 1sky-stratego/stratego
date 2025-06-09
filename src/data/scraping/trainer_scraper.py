import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv

# Load env vars
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv()

target_stocks = [t.strip() for t in os.getenv('TARGET_STOCKS', '').split(',') if t.strip()]
data_dir = os.getenv('DATA_DIR', 'data')
os.makedirs(data_dir, exist_ok=True)

def calculate_indicators(df):
    # Use lowercase column names since we convert them earlier
    close_col = 'close'
    high_col = 'high'
    low_col = 'low'
    volume_col = 'volume'
    
    # Moving averages
    df['sma_5'] = df[close_col].rolling(5).mean()
    df['sma_10'] = df[close_col].rolling(10).mean()
    df['sma_20'] = df[close_col].rolling(20).mean()
    df['sma_50'] = df[close_col].rolling(50).mean()
    
    df['ema_5'] = df[close_col].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df[close_col].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df[close_col].ewm(span=20, adjust=False).mean()
    
    # Ratios and relationships - add safety checks for division by zero
    df['price_to_sma20'] = df[close_col] / df['sma_20'].replace(0, np.nan)
    df['price_to_sma50'] = df[close_col] / df['sma_50'].replace(0, np.nan)
    df['sma20_to_sma50'] = df['sma_20'] / df['sma_50'].replace(0, np.nan)
    
    # RSI (Relative Strength Index)
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD and MACD signal line and histogram
    ema_12 = df[close_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[close_col].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands (BB)
    bb_mid = df[close_col].rolling(20).mean()
    bb_std = df[close_col].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    # Avoid division by zero
    bb_range = bb_upper - bb_lower
    df['bb_position'] = np.where(bb_range != 0, 
                                (df[close_col] - bb_lower) / bb_range, 
                                0.5)  # Default to middle position
    df['bb_width'] = bb_range / bb_mid.replace(0, np.nan)
    
    # Volume ratio: current volume divided by 20-day average volume
    avg_volume = df[volume_col].rolling(20).mean()
    df['volume_ratio'] = df[volume_col] / avg_volume.replace(0, np.nan)
    
    # Momentum indicators
    df['momentum_5'] = df[close_col] - df[close_col].shift(5)
    df['momentum_10'] = df[close_col] - df[close_col].shift(10)
    df['momentum_20'] = df[close_col] - df[close_col].shift(20)
    
    # Volatility: rolling std dev of returns over 20 days
    returns = df[close_col].pct_change()
    df['volatility'] = returns.rolling(20).std()
    
    # Support and resistance distance: simple proxies
    rolling_min = df[low_col].rolling(20).min()
    rolling_max = df[high_col].rolling(20).max()
    df['support_distance'] = (df[close_col] - rolling_min) / rolling_min.replace(0, np.nan)
    df['resistance_distance'] = (rolling_max - df[close_col]) / rolling_max.replace(0, np.nan)
    
    return df

def scrape_and_save():
    if not target_stocks:
        print("No target stocks found. Please check your .env file and TARGET_STOCKS variable.")
        return
    
    for ticker in target_stocks:
        print(f"Downloading data for {ticker}...")
        try:
            # Download data with auto_adjust=True to get adjusted prices
            df = yf.download(ticker, period='max', interval='1d', progress=False, auto_adjust=True)
            
            if df.empty:
                print(f"Warning: No data for {ticker}. Skipping.")
                continue

            # Print columns to debug
            print(f"Columns received for {ticker}: {list(df.columns)}")

            # Handle multi-index columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # If multi-index, get the first level (should be the column names)
                df.columns = df.columns.get_level_values(0)
                print(f"Flattened columns for {ticker}: {list(df.columns)}")

            # Check for required columns (yfinance sometimes returns different column names)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Map alternative column names
            column_mapping = {}
            for col in required_cols:
                if col in df.columns:
                    column_mapping[col] = col
                elif col.lower() in df.columns:
                    column_mapping[col.lower()] = col
                elif col.upper() in df.columns:
                    column_mapping[col.upper()] = col
            
            missing_cols = [col for col in required_cols if col not in column_mapping.values()]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} for {ticker}. Available columns: {list(df.columns)}")
                continue

            # Reset index to get timestamp as a column
            df = df.reset_index()
            
            # Ensure we have the Date column (yfinance uses 'Date' as index name)
            if 'Date' not in df.columns and df.index.name == 'Date':
                df = df.reset_index()
            elif 'Date' not in df.columns:
                # If no Date column, assume index is the date
                df['Date'] = df.index
            
            # Add symbol column
            df['symbol'] = ticker
            
            # Select and rename columns to match expected format
            try:
                df = df[['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
                # Rename columns to lowercase
                df.columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            except KeyError as e:
                print(f"Error selecting columns for {ticker}: {e}")
                print(f"Available columns: {list(df.columns)}")
                continue

            # Ensure numeric columns are actually numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values in essential columns
            df = df.dropna(subset=['close', 'volume'])
            
            if df.empty:
                print(f"Warning: No valid data remaining for {ticker} after cleaning. Skipping.")
                continue

            # Calculate indicators
            df = calculate_indicators(df)

            # Save to file
            file_path = Path(data_dir) / f"{ticker}.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved {len(df)} rows to {file_path}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    scrape_and_save()