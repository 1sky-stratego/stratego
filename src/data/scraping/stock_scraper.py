import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# File paths
data_dir = project_root / "data" / "collected"
csv_filename = data_dir / "ai_gpu_energy_stocks.csv"
json_filename = data_dir / "ai_gpu_energy_stocks.json"
excel_filename = data_dir / "ai_gpu_energy_stocks.xlsx"
summary_filename = data_dir / "summary_stats.json"
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

    def get_stock_fundamentals(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            hist['returns'] = hist['Close'].pct_change()
            hist['sma_20'] = hist['Close'].rolling(window=20).mean()
            hist['sma_50'] = hist['Close'].rolling(window=50).mean()
            hist['volatility'] = hist['returns'].rolling(window=20).std() * np.sqrt(252)

            info = ticker.info
            current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1])

            stock_data = {
                'symbol': symbol,
                'name': info.get('shortName', symbol),
                'current_price': float(current_price),
                'avg_volume_30d': float(hist['Volume'].tail(30).mean()),
                'price_52w_high': float(hist['High'].max()),
                'price_52w_low': float(hist['Low'].min()),
                'ytd_return': float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100),
                'volatility_annualized': float(hist['volatility'].iloc[-1]) if not pd.isna(hist['volatility'].iloc[-1]) else None,
                'sma_20': float(hist['sma_20'].iloc[-1]) if not pd.isna(hist['sma_20'].iloc[-1]) else None,
                'sma_50': float(hist['sma_50'].iloc[-1]) if not pd.isna(hist['sma_50'].iloc[-1]) else None,
                'market_cap_category': self._categorize_by_price(current_price),
                'data_collected_at': datetime.now().isoformat(),
                'bars_count': len(hist)
            }

            return stock_data
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None

    def _categorize_by_price(self, price):
        if price > 500:
            return 'High Price'
        elif price > 100:
            return 'Medium Price'
        else:
            return 'Low Price'

    def scrape_all_stocks(self, max_workers=5):
        logger.info(f"Starting to scrape {len(self.target_stocks)} stocks...")

        successful_scrapes = []
        failed_scrapes = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_stock_fundamentals, symbol): symbol 
                for symbol in self.target_stocks
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        successful_scrapes.append(result)
                        logger.info(f"Successfully scraped {symbol}")
                    else:
                        failed_scrapes.append(symbol)
                        logger.warning(f"Failed to scrape {symbol}")
                    time.sleep(0.1)
                except Exception as e:
                    failed_scrapes.append(symbol)
                    logger.error(f"Exception while scraping {symbol}: {str(e)}")

        self.stocks_data = successful_scrapes
        return successful_scrapes, failed_scrapes

    def save_data(self):
        # Use your existing save_data implementation — no change needed
        pass

def main():
    scraper = AIGPUEnergyStockScraper()
    successful, failed = scraper.scrape_all_stocks(max_workers=3)
    if successful:
        files = scraper.save_data()
        print(f"\nData collection complete!")
        print(f"Files saved: {files}")

        df = pd.DataFrame(scraper.stocks_data)
        print(f"\nQuick Analysis:")
        print(f"Total stocks collected: {len(df)}")
        print(f"Average current price: ${df['current_price'].mean():.2f}")
        print(f"Average YTD return: {df['ytd_return'].mean():.2f}%")
        print(f"\nTop 5 YTD performers:")
        print(df.nlargest(5, 'ytd_return')[['symbol', 'current_price', 'ytd_return']])
    else:
        print("No data was successfully collected. Please check your internet connection.")

if __name__ == "__main__":
    main()
