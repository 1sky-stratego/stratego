import yfinance as yf
import pandas as pd
import numpy as np
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
        self.all_stock_data = []

    def get_market_cap_category(self, market_cap):
        """
        Categorize market cap into Small, Mid, Large cap
        
        Args:
            market_cap: Market cap in dollars
        
        Returns:
            String category
        """
        if pd.isna(market_cap) or market_cap == 0:
            return 'Unknown'
        
        if market_cap < 2e9:  # Less than $2B
            return 'Small Cap'
        elif market_cap < 10e9:  # $2B - $10B
            return 'Mid Cap'
        else:  # Greater than $10B
            return 'Large Cap'

    def calculate_indicators(self, data, ticker_info):
        """
        Calculate technical indicators and metrics for a stock
        
        Args:
            data: DataFrame with OHLCV data
            ticker_info: yfinance Ticker.info dictionary
        
        Returns:
            Dictionary with calculated metrics
        """
        if data.empty or len(data) < 50:  # Need at least 50 days for calculations
            return None
        
        try:
            # Current (most recent) values
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-30:].mean()  # 30-day average volume
            
            # 52-week high/low
            price_52w_high = data['High'].iloc[-252:].max() if len(data) >= 252 else data['High'].max()
            price_52w_low = data['Low'].iloc[-252:].min() if len(data) >= 252 else data['Low'].min()
            
            # YTD return calculation
            # Find the first trading day of current year
            current_year = datetime.now().year
            try:
                ytd_start_data = data[data.index.year == current_year]
                if not ytd_start_data.empty:
                    ytd_start_price = ytd_start_data['Close'].iloc[0]
                    ytd_return = ((current_price - ytd_start_price) / ytd_start_price) * 100
                else:
                    # If no current year data, use available data
                    ytd_return = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            except:
                ytd_return = 0.0
            
            # Annualized volatility (252 trading days)
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 1:
                daily_volatility = returns.std()
                volatility_annualized = daily_volatility * np.sqrt(252) * 100
            else:
                volatility_annualized = 0.0
            
            # Simple Moving Averages
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_price
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else current_price
            
            # Get company info
            try:
                company_name = ticker_info.get('longName', ticker_info.get('shortName', 'Unknown'))
                market_cap = ticker_info.get('marketCap', 0)
            except:
                company_name = 'Unknown'
                market_cap = 0
            
            # Market cap category
            market_cap_category = self.get_market_cap_category(market_cap)
            
            return {
                'name': company_name,
                'current_price': round(current_price, 2),
                'avg_volume_30d': int(current_volume),
                'price_52w_high': round(price_52w_high, 2),
                'price_52w_low': round(price_52w_low, 2),
                'ytd_return': round(ytd_return, 2),
                'volatility_annualized': round(volatility_annualized, 2),
                'sma_20': round(sma_20, 2) if not pd.isna(sma_20) else current_price,
                'sma_50': round(sma_50, 2) if not pd.isna(sma_50) else current_price,
                'market_cap_category': market_cap_category,
                'bars_count': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def download_single_stock(self, ticker, start_date, end_date, period="1d"):
        """
        Download and process data for a single stock
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Data frequency ('1d', '1h', '5m', etc.)
        
        Returns:
            Dictionary with processed stock data or None if failed
        """
        try:
            logger.info(f"Processing {ticker}...")
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Download historical data
            data = stock.history(start=start_date, end=end_date, interval=period)
            
            if data.empty:
                logger.warning(f"No historical data found for {ticker}")
                return None
            
            # Clean the data
            data = data.dropna()
            
            # Get company info
            try:
                ticker_info = stock.info
            except:
                logger.warning(f"Could not get info for {ticker}, using defaults")
                ticker_info = {}
            
            # Calculate all indicators and metrics
            metrics = self.calculate_indicators(data, ticker_info)
            
            if metrics is None:
                logger.warning(f"Could not calculate metrics for {ticker}")
                return None
            
            # Create the final record
            stock_record = {
                'symbol': ticker,
                'name': metrics['name'],
                'current_price': metrics['current_price'],
                'avg_volume_30d': metrics['avg_volume_30d'],
                'price_52w_high': metrics['price_52w_high'],
                'price_52w_low': metrics['price_52w_low'],
                'ytd_return': metrics['ytd_return'],
                'volatility_annualized': metrics['volatility_annualized'],
                'sma_20': metrics['sma_20'],
                'sma_50': metrics['sma_50'],
                'market_cap_category': metrics['market_cap_category'],
                'data_collected_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'bars_count': metrics['bars_count']
            }
            
            logger.info(f"✓ {ticker}: {metrics['bars_count']} bars, Current: ${metrics['current_price']}")
            return stock_record
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            return None

    def download_all_stocks(self, start_date="2020-01-01", end_date=None, period="1d", delay=0.5):
        """
        Download data for all stocks and compile into single dataset
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            period: Data frequency ('1d', '1h', '5m', etc.)
            delay: Delay between downloads to avoid rate limiting
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting data collection for {len(self.target_stocks)} stocks")
        logger.info(f"Historical data range: {start_date} to {end_date}")
        logger.info(f"Data will be saved to: {self.data_dir}")
        
        self.all_stock_data = []
        
        for i, ticker in enumerate(self.target_stocks, 1):
            logger.info(f"Progress: {i}/{len(self.target_stocks)} - {ticker}")
            
            # Download and process the stock
            stock_data = self.download_single_stock(ticker, start_date, end_date, period)
            
            if stock_data is not None:
                self.all_stock_data.append(stock_data)
                self.successful_downloads.append(ticker)
            else:
                self.failed_downloads.append(ticker)
                logger.error(f"✗ Failed to process {ticker}")
            
            # Add delay to avoid rate limiting
            if delay > 0 and i < len(self.target_stocks):
                time.sleep(delay)
        
        # Save all data to single CSV file
        self.save_unified_data(start_date, end_date)
        self.print_summary()

    def save_unified_data(self, start_date, end_date):
        """
        Save all collected data to a single CSV file
        """
        if not self.all_stock_data:
            logger.error("No data to save")
            return
        
        # Create DataFrame from all stock data
        df = pd.DataFrame(self.all_stock_data)
        
        # Define column order to match your specification
        column_order = [
            'symbol', 'name', 'current_price', 'avg_volume_30d', 'price_52w_high', 
            'price_52w_low', 'ytd_return', 'volatility_annualized', 'sma_20', 'sma_50', 
            'market_cap_category', 'data_collected_at', 'bars_count'
        ]
        
        # Reorder columns
        df = df[column_order]
        
        # Sort by symbol for consistency
        df = df.sort_values('symbol')
        
        # Save to CSV
        filename = f"ai_gpu_energy_stocks_{start_date}_{end_date}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"✓ Unified data saved to: {filename}")
        logger.info(f"✓ Total records: {len(df)}")
        
        # Also save summary statistics
        self.save_summary_stats(df, start_date, end_date)
        
        return filepath

    def save_summary_stats(self, df, start_date, end_date):
        """
        Save summary statistics of the collected data
        """
        try:
            # Calculate summary statistics
            summary = {
                'collection_info': {
                    'total_stocks': len(df),
                    'successful_downloads': len(self.successful_downloads),
                    'failed_downloads': len(self.failed_downloads),
                    'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_period': f"{start_date} to {end_date}"
                },
                'price_statistics': {
                    'avg_current_price': float(df['current_price'].mean()),
                    'median_current_price': float(df['current_price'].median()),
                    'min_current_price': float(df['current_price'].min()),
                    'max_current_price': float(df['current_price'].max())
                },
                'return_statistics': {
                    'avg_ytd_return': float(df['ytd_return'].mean()),
                    'median_ytd_return': float(df['ytd_return'].median()),
                    'best_ytd_return': float(df['ytd_return'].max()),
                    'worst_ytd_return': float(df['ytd_return'].min())
                },
                'volatility_statistics': {
                    'avg_volatility': float(df['volatility_annualized'].mean()),
                    'median_volatility': float(df['volatility_annualized'].median()),
                    'min_volatility': float(df['volatility_annualized'].min()),
                    'max_volatility': float(df['volatility_annualized'].max())
                },
                'market_cap_distribution': df['market_cap_category'].value_counts().to_dict(),
                'top_performers': {
                    'best_ytd': df.nlargest(5, 'ytd_return')[['symbol', 'name', 'ytd_return']].to_dict('records'),
                    'worst_ytd': df.nsmallest(5, 'ytd_return')[['symbol', 'name', 'ytd_return']].to_dict('records')
                }
            }
            
            # Save summary as JSON
            import json
            summary_filename = f"summary_stats_{start_date}_{end_date}.json"
            summary_filepath = os.path.join(self.data_dir, summary_filename)
            
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"✓ Summary statistics saved to: {summary_filename}")
            
        except Exception as e:
            logger.error(f"Error saving summary stats: {str(e)}")

    def print_summary(self):
        """Print download summary"""
        print("\n" + "="*80)
        print("DATA COLLECTION SUMMARY")
        print("="*80)
        print(f"Total stocks processed: {len(self.target_stocks)}")
        print(f"Successful: {len(self.successful_downloads)}")
        print(f"Failed: {len(self.failed_downloads)}")
        print(f"Success rate: {len(self.successful_downloads)/len(self.target_stocks)*100:.1f}%")
        print(f"Data saved to: {self.data_dir}")
        
        if self.successful_downloads:
            print(f"\n✓ Successfully processed stocks:")
            for i, ticker in enumerate(self.successful_downloads):
                if i < 10:  # Show first 10
                    stock_data = next((s for s in self.all_stock_data if s['symbol'] == ticker), None)
                    if stock_data:
                        print(f"  {ticker}: {stock_data['name'][:30]} - ${stock_data['current_price']:.2f}")
                elif i == 10:
                    print(f"  ... and {len(self.successful_downloads) - 10} more")
                    break
        
        if self.failed_downloads:
            print(f"\n✗ Failed to process:")
            for ticker in self.failed_downloads:
                print(f"  {ticker}")
        
        # Print some quick stats if we have data
        if self.all_stock_data:
            df = pd.DataFrame(self.all_stock_data)
            print(f"\nQuick Statistics:")
            print(f"Average YTD Return: {df['ytd_return'].mean():.2f}%")
            print(f"Average Volatility: {df['volatility_annualized'].mean():.2f}%")
            print(f"Market Cap Distribution: {dict(df['market_cap_category'].value_counts())}")

    def load_unified_data(self, start_date=None, end_date=None):
        """
        Load previously collected unified stock data
        
        Returns:
            DataFrame with all stock data
        """
        if start_date is None or end_date is None:
            # Look for most recent file
            files = [f for f in os.listdir(self.data_dir) if f.startswith('ai_gpu_energy_stocks_') and f.endswith('.csv')]
            if not files:
                logger.error("No unified data files found")
                return None
            filename = max(files)  # Get most recent
        else:
            filename = f"ai_gpu_energy_stocks_{start_date}_{end_date}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded unified data: {len(data)} stocks from {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading unified data: {str(e)}")
            return None

def main():
    """Main execution function"""
    scraper = StockDataScraper()
    
    # Configuration
    start_date = "2020-01-01"  # Adjust as needed
    end_date = None  # None = today
    
    print(f"Starting unified data collection for {len(scraper.target_stocks)} stocks...")
    print(f"Historical data range: {start_date} to {end_date or 'today'}")
    print("\nTarget columns:")
    print("symbol, name, current_price, avg_volume_30d, price_52w_high, price_52w_low,")
    print("ytd_return, volatility_annualized, sma_20, sma_50, market_cap_category,")
    print("data_collected_at, bars_count")
    
    # Download all stocks and create unified dataset
    scraper.download_all_stocks(
        start_date=start_date,
        end_date=end_date,
        period="1d",  # Daily data
        delay=0.5     # Delay between requests to avoid rate limiting
    )
    
    # Example of loading the unified data
    if scraper.successful_downloads:
        print(f"\nLoading unified dataset...")
        data = scraper.load_unified_data(start_date, end_date or datetime.now().strftime("%Y-%m-%d"))
        if data is not None:
            print(f"\nUnified Dataset Preview:")
            print(data.head())
            print(f"\nDataset Info:")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")

if __name__ == "__main__":
    main()