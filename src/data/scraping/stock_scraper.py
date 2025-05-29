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
        """Save scraped data in multiple formats, appending to existing files"""
        if not self.stocks_data:
            logger.warning("No data to save!")
            return
        
        # File paths
        data_dir = project_root / "src" / "data" / "collected"
        csv_filename = data_dir / "ai_gpu_energy_stocks.csv"
        json_filename = data_dir / "ai_gpu_energy_stocks.json"
        excel_filename = data_dir / "ai_gpu_energy_stocks.xlsx"
        summary_filename = data_dir / "summary_stats.json"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        current_df = pd.DataFrame(self.stocks_data)
        timestamp = datetime.now().isoformat()
        
        # Handle CSV - append or create
        if os.path.exists(csv_filename):
            existing_df = pd.read_csv(csv_filename)
            # Remove existing entries for the same symbols and date to avoid duplicates
            today = datetime.now().date()
            if 'data_collected_at' in existing_df.columns:
                existing_df['collection_date'] = pd.to_datetime(existing_df['data_collected_at']).dt.date
                existing_df = existing_df[existing_df['collection_date'] != today]
                existing_df = existing_df.drop('collection_date', axis=1)
            
            # Combine dataframes
            combined_df = pd.concat([existing_df, current_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='last')  # Keep latest data per symbol
        else:
            combined_df = current_df
        
        combined_df.to_csv(csv_filename, index=False)
        logger.info(f"Data saved/updated in {csv_filename}")
        
        # Handle JSON - maintain historical data with timestamps
        all_data = []
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r') as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    all_data = existing_data
                else:
                    # Handle old format
                    all_data = [existing_data] if existing_data else []
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Could not read existing JSON file: {e}")
                all_data = []
        
        # Add current batch with timestamp
        current_batch = {
            'collection_timestamp': timestamp,
            'data': self.stocks_data,
            'summary': {
                'total_stocks': len(self.stocks_data),
                'avg_price': float(current_df['current_price'].mean()),
                'avg_ytd_return': float(current_df['ytd_return'].mean())
            }
        }
        all_data.append(current_batch)
        
        # Keep only last 30 collections to prevent file from growing too large
        if len(all_data) > 30:
            all_data = all_data[-30:]
        
        with open(json_filename, 'w') as f:
            json.dump(all_data, f, indent=2, default=str)
        logger.info(f"Data saved/updated in {json_filename}")
        
        # Handle Excel - overwrite with latest data and add historical sheet
        try:
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # Current data
                combined_df.to_excel(writer, sheet_name='Current_Data', index=False)
                
                # Create separate sheets by category (current data only)
                ai_gpu_symbols = ['NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'MRVL', 'XLNX', 'LRCX', 'KLAC', 'AMAT']
                ai_gpu_stocks = combined_df[combined_df['symbol'].isin(ai_gpu_symbols)]
                if not ai_gpu_stocks.empty:
                    ai_gpu_stocks.to_excel(writer, sheet_name='AI_GPU_Hardware', index=False)
                
                energy_symbols = ['TSLA', 'ENPH', 'SEDG', 'FSLR', 'SPWR', 'RUN', 'NEE', 'PLUG', 'BE', 'BLDP']
                energy_stocks = combined_df[combined_df['symbol'].isin(energy_symbols)]
                if not energy_stocks.empty:
                    energy_stocks.to_excel(writer, sheet_name='Energy', index=False)
                
                ai_software_symbols = ['MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'CRM', 'NOW', 'SNOW', 'PLTR', 'AI']
                ai_software_stocks = combined_df[combined_df['symbol'].isin(ai_software_symbols)]
                if not ai_software_stocks.empty:
                    ai_software_stocks.to_excel(writer, sheet_name='AI_Software', index=False)
                
                # Historical summary if we have JSON data
                if len(all_data) > 1:
                    historical_summary = []
                    for batch in all_data[-10:]:  # Last 10 collections
                        historical_summary.append({
                            'collection_date': batch['collection_timestamp'][:10],
                            'total_stocks': batch['summary']['total_stocks'],
                            'avg_price': batch['summary']['avg_price'],
                            'avg_ytd_return': batch['summary']['avg_ytd_return']
                        })
                    
                    hist_df = pd.DataFrame(historical_summary)
                    hist_df.to_excel(writer, sheet_name='Historical_Summary', index=False)
                    
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
        
        logger.info(f"Data saved/updated in {excel_filename}")
        
        # Update summary statistics
        summary = {
            'last_updated': timestamp,
            'total_stocks_current': len(combined_df),
            'total_collections': len(all_data),
            'current_stats': {
                'avg_price': float(combined_df['current_price'].mean()),
                'price_range': f"${combined_df['current_price'].min():.2f} - ${combined_df['current_price'].max():.2f}",
                'avg_ytd_return': float(combined_df['ytd_return'].mean()),
                'top_performers': combined_df.nlargest(5, 'ytd_return')[['symbol', 'ytd_return']].to_dict('records'),
                'bottom_performers': combined_df.nsmallest(5, 'ytd_return')[['symbol', 'ytd_return']].to_dict('records'),
                'most_volatile': combined_df.nlargest(5, 'volatility_annualized')[['symbol', 'volatility_annualized']].to_dict('records') if 'volatility_annualized' in combined_df.columns else []
            }
        }
        
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved/updated in {summary_filename}")
        
        return csv_filename, json_filename, excel_filename, summary_filename


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
