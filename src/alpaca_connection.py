import os
from dotenv import load_dotenv
from pathlib import Path

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    
    # Load environment variables from project root
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    
    print(f"Looking for .env at: {env_path}")
    
    if not env_path.exists():
        print(f"❌ .env file not found at {env_path}")
        return False
    
    # Load the .env file
    load_dotenv(env_path)
    
    # Check if environment variables are loaded
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    print(f"Secret Key loaded: {'Yes' if secret_key else 'No'}")
    
    if not api_key or not secret_key:
        print("❌ API keys not found in environment variables")
        print("Make sure your .env file has:")
        print("ALPACA_API_KEY=your_key_here")
        print("ALPACA_SECRET_KEY=your_secret_here")
        return False
    
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        
        print(f"Using API Key: {api_key[:8]}..." if len(api_key) > 8 else api_key)
        
        # Initialize clients
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True  # Set to False for live trading
        )
        
        data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        
        # Test account connection
        print("Testing account connection...")
        account = trading_client.get_account()
        print(f"✅ Account Status: {account.status}")
        print(f"✅ Buying Power: ${account.buying_power}")
        print(f"✅ Portfolio Value: ${account.portfolio_value}")
        
        print("🎉 Alpaca connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_alpaca_connection()