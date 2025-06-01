import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataScraper:
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
        
        # Create data directory in same folder as script
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.failed_downloads = []
        self.successful_downloads = []

    def download_single_stock(self, ticker, start_date, end_date, period="1d"):
        """
        Download data for a single stock
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Data frequency ('1d', '1h', '5m', etc.)
        
        Returns:
            DataFrame or None if failed
        """
        try:
            logger.info(f"Downloading {ticker}...")
            
            # Download data
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval=period)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return None
            
            # Clean the data
            data = data.dropna()
            
            # Ensure we have the required columns for backtesting
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"{ticker}: Missing columns {missing_columns}")
                return None
            
            # Add ticker column for reference
            data['Ticker'] = ticker
            
            logger.info(f"{ticker}: Downloaded {len(data)} rows from {data.index[0].date()} to {data.index[-1].date()}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
            return None

    def download_all_stocks(self, start_date="2020-01-01", end_date=None, period="1d", delay=0.1):
        """
        Download data for all stocks in the list
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            period: Data frequency ('1d', '1h', '5m', etc.)
            delay: Delay between downloads to avoid rate limiting
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting download for {len(self.target_stocks)} stocks")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Data will be saved to: {self.data_dir}")
        
        for i, ticker in enumerate(self.target_stocks, 1):
            logger.info(f"Progress: {i}/{len(self.target_stocks)} - {ticker}")
            
            # Download the data
            data = self.download_single_stock(ticker, start_date, end_date, period)
            
            if data is not None:
                # Save to CSV
                filename = f"{ticker}_{start_date}_{end_date}.csv"
                filepath = os.path.join(self.data_dir, filename)
                data.to_csv(filepath)
                
                self.successful_downloads.append({
                    'ticker': ticker,
                    'filename': filename,
                    'rows': len(data),
                    'date_range': f"{data.index[0].date()} to {data.index[-1].date()}"
                })
                
                logger.info(f"✓ {ticker} saved to {filename}")
            else:
                self.failed_downloads.append(ticker)
                logger.error(f"✗ Failed to download {ticker}")
            
            # Add delay to avoid rate limiting
            if delay > 0 and i < len(self.target_stocks):
                time.sleep(delay)
        
        self.print_summary()

    def download_batch(self, tickers_batch, start_date="2020-01-01", end_date=None):
        """
        Download multiple tickers at once (faster but less reliable for large lists)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            logger.info(f"Batch downloading {len(tickers_batch)} tickers...")
            
            # Download all at once
            data = yf.download(tickers_batch, start=start_date, end=end_date, group_by='ticker')
            
            # Save individual files
            for ticker in tickers_batch:
                try:
                    if len(tickers_batch) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker].dropna()
                    
                    if not ticker_data.empty:
                        filename = f"{ticker}_{start_date}_{end_date}.csv"
                        filepath = os.path.join(self.data_dir, filename)
                        ticker_data.to_csv(filepath)
                        
                        self.successful_downloads.append({
                            'ticker': ticker,
                            'filename': filename,
                            'rows': len(ticker_data),
                            'date_range': f"{ticker_data.index[0].date()} to {ticker_data.index[-1].date()}"
                        })
                        logger.info(f"✓ {ticker} saved")
                    else:
                        self.failed_downloads.append(ticker)
                        logger.warning(f"✗ No data for {ticker}")
                        
                except Exception as e:
                    self.failed_downloads.append(ticker)
                    logger.error(f"✗ Error processing {ticker}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Batch download failed: {str(e)}")
            # Fallback to individual downloads
            logger.info("Falling back to individual downloads...")
            for ticker in tickers_batch:
                data = self.download_single_stock(ticker, start_date, end_date)
                if data is not None:
                    filename = f"{ticker}_{start_date}_{end_date}.csv"
                    filepath = os.path.join(self.data_dir, filename)
                    data.to_csv(filepath)
                    self.successful_downloads.append({'ticker': ticker, 'filename': filename})

    def print_summary(self):
        """Print download summary"""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Successful downloads: {len(self.successful_downloads)}")
        print(f"Failed downloads: {len(self.failed_downloads)}")
        print(f"Data saved to: {self.data_dir}")
        
        if self.successful_downloads:
            print(f"\n✓ Successfully downloaded:")
            for item in self.successful_downloads:
                print(f"  {item['ticker']}: {item['rows']} rows ({item.get('date_range', 'N/A')})")
        
        if self.failed_downloads:
            print(f"\n✗ Failed to download:")
            for ticker in self.failed_downloads:
                print(f"  {ticker}")

    def load_stock_data(self, ticker, start_date, end_date):
        """
        Load previously downloaded stock data for backtesting
        
        Returns:
            DataFrame ready for backtesting library
        """
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Remove ticker column if it exists (not needed for backtesting)
            if 'Ticker' in data.columns:
                data = data.drop('Ticker', axis=1)
            
            logger.info(f"Loaded {ticker}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading {ticker}: {str(e)}")
            return None

def main():
    """Main execution function"""
    scraper = StockDataScraper()
    
    # Configuration
    start_date = "2020-01-01"  # Adjust as needed
    end_date = None  # None = today
    
    print(f"Starting data download for {len(scraper.target_stocks)} stocks...")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    
    # Download all stocks (individual method - more reliable)
    scraper.download_all_stocks(
        start_date=start_date,
        end_date=end_date,
        period="1d",  # Daily data
        delay=0.1     # Small delay between requests
    )
    
    # Example of how to load data for backtesting
    if scraper.successful_downloads:
        print(f"\nExample - Loading data for backtesting:")
        sample_ticker = scraper.successful_downloads[0]['ticker']
        data = scraper.load_stock_data(sample_ticker, start_date, end_date or datetime.now().strftime("%Y-%m-%d"))
        if data is not None:
            print(f"Sample data for {sample_ticker}:")
            print(data.head())
            print(f"Shape: {data.shape}")

if __name__ == "__main__":
    main()