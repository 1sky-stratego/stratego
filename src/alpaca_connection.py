import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Load environment variables
load_dotenv()

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    
    # Initialize clients
    trading_client = TradingClient(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True  # Set to False for live trading
    )
    
    data_client = StockHistoricalDataClient(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY')
    )
    
    try:
        # Test account connection
        account = trading_client.get_account()
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${account.buying_power}")
        print(f"Portfolio Value: ${account.portfolio_value}")
        
        # Test data connection
        request = StockLatestQuoteRequest(symbol_or_symbols=["AAPL"])
        latest_quote = data_client.get_stock_latest_quote(request)
        print(f"AAPL Latest Quote: ${latest_quote['AAPL'].bid_price}")
        
        print("✅ Alpaca connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_alpaca_connection()
