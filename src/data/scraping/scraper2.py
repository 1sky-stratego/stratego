import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

print(os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# File paths
data_dir = project_root / "src" / "data" / "collected"
csv_filename = data_dir / "ai_gpu_energy_stocks_3year.csv"
data_dir.mkdir(parents=True, exist_ok=True)

class AIGPUEnergyStockScraper:
    def __init__(self):
        self.stocks_data = []

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

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the historical data"""
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # Price ratios
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']
        df['sma20_to_sma50'] = df['sma_20'] / df['sma_50']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_rolling_std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume indicators
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Momentum
        df['momentum_5'] = df['close'].pct_change(periods=5)
        df['momentum_10'] = df['close'].pct_change(periods=10)
        df['momentum_20'] = df['close'].pct_change(periods=20)
        df['daily_momentum'] = df['close'].pct_change()
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Daily positioning
        df['price_to_day_high'] = df['close'] / df['high']
        df['price_to_day_low'] = df['close'] / df['low']
        df['daily_range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 52-week positioning (rolling 252 trading days)
        df['52w_high'] = df['high'].rolling(window=252).max()
        df['52w_low'] = df['low'].rolling(window=252).min()
        df['price_to_52w_high'] = df['close'] / df['52w_high']
        df['price_to_52w_low'] = df['close'] / df['52w_low']
        df['52w_range_position'] = (df['close'] - df['52w_low']) / (df['52w_high'] - df['52w_low'])
        
        # YTD momentum (approximate - using 252 trading days)
        df['ytd_momentum'] = df['close'].pct_change(periods=252)
        
        # Support/Resistance (simplified using recent lows/highs)
        df['support_level'] = df['low'].rolling(window=50).min()
        df['resistance_level'] = df['high'].rolling(window=50).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        return df

    def get_stock_historical_data(self, symbol):
        """Get 3 years of historical data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get 3 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Reset index to make Date a column
            hist = hist.reset_index()
            hist.columns = hist.columns.str.lower()
            
            # Add symbol and name
            info = ticker.info
            hist['symbol'] = symbol
            hist['name'] = info.get('shortName', symbol)
            
            # Calculate technical indicators
            hist = self.calculate_technical_indicators(hist)
            
            # Add market cap category based on most recent price
            current_price = hist['close'].iloc[-1]
            hist['market_cap_category'] = self._categorize_by_price(current_price)
            
            # Add data collection timestamp
            hist['data_collected_at'] = datetime.now().isoformat()
            
            logger.info(f"Successfully collected {len(hist)} days of data for {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None

    def _categorize_by_price(self, price):
        if price > 500:
            return 'High Price'
        elif price > 100:
            return 'Medium Price'
        else:
            return 'Low Price'

    def scrape_all_stocks(self, max_workers=5):
        logger.info(f"Starting to scrape 3-year historical data for {len(self.target_stocks)} stocks...")

        successful_scrapes = []
        failed_scrapes = []
        all_data = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_stock_historical_data, symbol): symbol 
                for symbol in self.target_stocks
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_data.append(result)
                        successful_scrapes.append(symbol)
                        logger.info(f"Successfully scraped {symbol} - {len(result)} records")
                    else:
                        failed_scrapes.append(symbol)
                        logger.warning(f"Failed to scrape {symbol}")
                    time.sleep(0.1)
                except Exception as e:
                    failed_scrapes.append(symbol)
                    logger.error(f"Exception while scraping {symbol}: {str(e)}")

        # Combine all dataframes
        if all_data:
            self.stocks_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined data: {len(self.stocks_data)} total records")
        else:
            self.stocks_data = pd.DataFrame()
            
        return successful_scrapes, failed_scrapes

    def save_data(self):
        if self.stocks_data.empty:
            logger.warning("No data to save!")
            return None
        
        # File paths
        data_dir = project_root / "src" / "data" / "collected"
        csv_filename = data_dir / "ai_gpu_energy_stocks_3year.csv"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        self.stocks_data.to_csv(csv_filename, index=False)
        logger.info(f"Historical data saved to {csv_filename}")
        logger.info(f"Total records: {len(self.stocks_data)}")
        logger.info(f"Date range: {self.stocks_data['date'].min()} to {self.stocks_data['date'].max()}")
        logger.info(f"Stocks included: {self.stocks_data['symbol'].nunique()}")
        
        return csv_filename

def main():
    scraper = AIGPUEnergyStockScraper()
    successful, failed = scraper.scrape_all_stocks(max_workers=3)
    
    if successful:
        filename = scraper.save_data()
        print(f"\nData collection complete!")
        print(f"File saved: {filename}")
        
        if not scraper.stocks_data.empty:
            df = scraper.stocks_data
            print(f"\nQuick Analysis:")
            print(f"Total records collected: {len(df):,}")
            print(f"Number of stocks: {df['symbol'].nunique()}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Average closing price: ${df['close'].mean():.2f}")
            
            # Show recent performance for each stock
            latest_data = df.groupby('symbol').tail(1)
            if 'ytd_momentum' in latest_data.columns:
                print(f"\nTop 5 recent YTD performers (based on latest data):")
                top_performers = latest_data.nlargest(5, 'ytd_momentum')[['symbol', 'close', 'ytd_momentum']]
                for _, row in top_performers.iterrows():
                    print(f"  {row['symbol']}: ${row['close']:.2f} ({row['ytd_momentum']*100:.1f}%)")
    else:
        print("No data was successfully collected. Please check your internet connection.")
    
    if failed:
        print(f"\nFailed to collect data for: {failed}")

if __name__ == "__main__":
    main()