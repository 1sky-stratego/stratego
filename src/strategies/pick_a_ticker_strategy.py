import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import alpaca_trade_api as tradeapi
import warnings
import logging
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

# Path setup
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
dotenv_path = load_dotenv()
print(f"Found .env file at: {dotenv_path}")
load_dotenv(dotenv_path=env_path)

csv_path = 'src/data/collected/ai_gpu_energy_stocks.csv'

# APIs
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
base_url = os.getenv('ALPACA_API_BASE_URL') 

# Import for backtesting compatibility
try:
    from backtesting import Strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    Strategy = object  # Fallback if backtesting not installed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import parameters
PARAM_prediction_horizon = int(os.getenv('PARAM_PREDICTION_HORIZON'))
PARAM_confidence_threshold = float(os.getenv('PARAM_CONFIDENCE_THRESHOLD'))
PARAM_min_data_points = int(os.getenv('PARAM_MIN_DATA_POINTS'))


class TradingAlgorithm:
    def __init__(self, api_key, secret_key, base_url="https://paper-api.alpaca.markets"):
        """
        Initialize trading algorithm with Alpaca API credentials
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key  
            base_url: Alpaca API base URL (paper or live)
        """
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        
        # Algorithm parameters (customize as needed)
        self.prediction_horizon = PARAM_prediction_horizon
        self.confidence_threshold = PARAM_confidence_threshold
        self.min_data_points = PARAM_min_data_points
        
        logger.info("Trading algorithm initialized")

    def load_data(self, file_path):
        """
        Load stock data from CSV file
        
        CSV column format: symbol,name,current_price,avg_volume_30d,price_52w_high,price_52w_low,ytd_return,volatility_annualized,sma_20,sma_50,market_cap_category,data_collected_at,bars_count,open,daily_high,daily_low
        """
        try:
            df = pd.read_csv(file_path) 
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['data_collected_at'])
            df.set_index('timestamp', inplace=True)
            
            # Map columns to standard OHLCV format
            # Since you only have current_price, we'll use it for all OHLC
            df['close'] = pd.to_numeric(df['current_price'], errors='coerce')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['yr_high'] = pd.to_numeric(df['price_52w_high'], errors='coerce') 
            df['yr_low'] = pd.to_numeric(df['price_52w_low'], errors='coerce')
            df['volume'] = pd.to_numeric(df['avg_volume_30d'], errors='coerce')
            df['day_high'] = pd.to_numeric(df['daily_high'], errors='coerce')
            df['day_low'] = pd.to_numeric(df['daily_low'], errors='coerce')
            
            # Keep your additional columns as they contain useful features
            df['sma_20'] = pd.to_numeric(df['sma_20'], errors='coerce')
            df['sma_50'] = pd.to_numeric(df['sma_50'], errors='coerce')
            df['ytd_return'] = pd.to_numeric(df['ytd_return'], errors='coerce')
            df['volatility_annualized'] = pd.to_numeric(df['volatility_annualized'], errors='coerce')
            
            # Check for required data
            if df['close'].isna().all():
                logger.error("No valid price data found")
                return None
            
            # Handle missing volume
            if df['volume'].isna().all():
                df['volume'] = 1000000  # Default volume
                logger.warning("Volume data missing, using default values")
            
            logger.info(f"Loaded {len(df)} data points from {file_path}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Symbols: {df['symbol'].unique()}")
            logger.info(f"Sample data:\n{df[['symbol', 'close', 'sma_20', 'sma_50', 'volume']].head()}")
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Print first few lines of file for debugging
            try:
                with open(file_path, 'r') as f:
                    logger.info("First 3 lines of file:")
                    for i, line in enumerate(f):
                        if i < 3:
                            logger.info(f"Line {i+1}: {line.strip()}")
                        else:
                            break
            except:
                pass
            return None
        
    def create_features(self, df):
        """Create technical indicators for linear regression"""
        df = df.copy()
        
        # You already have sma_20 and sma_50, let's use them and add more
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
        if 'sma_50' not in df.columns:
            df['sma_50'] = df['close'].rolling(50).mean()
        
        # Additional SMAs
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI (if not already present)
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Price ratios using your existing SMAs
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']
        df['sma20_to_sma50'] = df['sma_20'] / df['sma_50']
        
        # Volume indicators (using avg_volume_30d)
        if 'avg_volume_30d' in df.columns:
            df['volume_ratio'] = df['volume'] / df['avg_volume_30d']
        else:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum and volatility
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Use your existing volatility if available, otherwise calculate
        if 'volatility_annualized' not in df.columns:
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        else:
            df['volatility'] = df['volatility_annualized']
        
        # 52-week high/low ratios 
        if 'yr_high' in df.columns and 'yr_low' in df.columns:
            df['price_to_52w_high'] = df['close'] / df['yr_high']
            df['price_to_52w_low'] = df['close'] / df['yr_low']
            df['52w_range_position'] = (df['close'] - df['yr_low']) / (df['yr_high'] - df['yr_low'])
        
        # Daily high/low ratios
        if 'daily_high' in df.columns and 'daily_low' in df.columns:
            df['price_to_day_high'] = df['close'] / df['daily_high']
            df['price_to_day_low'] = df['close'] / df['daily_low']
            df['daily_range_position'] = (df['close'] - df['daily_low']) / (df['daily_high'] - df['daily_low'])
        
        # YTD return momentum 
        if 'ytd_return' in df.columns:
            df['ytd_momentum'] = df['ytd_return']

        # Daily momentum
        if 'daily_return' in df.columns:
            df['daily_momentum'] = df['ytd_return']
        
        # Support and Resistance levels
        if 'daily_low' in df.columns and 'daily_high' in df.columns:
            df['support'] = df['daily_low'].rolling(20).min()
            df['resistance'] = df['daily_high'].rolling(20).max()
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
        else:
            # Fallback: use close price for support/resistance
            df['support'] = df['close'].rolling(20).min()
            df['resistance'] = df['close'].rolling(20).max()
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
        
        return df   

    def train_linear_model(self, df):
        """Train linear regression model"""
        try:
            # Enhanced feature set that leverages your existing data
            feature_cols = [
                # Moving averages and ratios
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20',
                'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
                
                # Technical indicators  
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                
                # Bollinger Bands
                'bb_position', 'bb_width',
                
                # Volume and momentum
                'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20',
                
                # Volatility
                'volatility',

                # daily positioning
                'price_to_day_high', 'price_to_day_low', 'daily_range_position',

                # daily performance
                'daily_momentum',
                
                # 52-week positioning 
                'price_to_52w_high', 'price_to_52w_low', '52w_range_position',
                
                # YTD performance 
                'ytd_momentum',
                
                # Support/Resistance
                'support_distance', 'resistance_distance'
            ]
            
            # Filter features that actually exist in the dataframe
            available_features = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]
            
            logger.info(f"Using {len(available_features)} features: {available_features}")
            
            # Create feature matrix
            X = df[available_features].dropna()
            
            # Create target: future price change
            future_returns = df['close'].shift(-int(self.prediction_horizon)).pct_change()
            y = future_returns.dropna()
            
            # Align data
            min_idx = max(X.index.min(), y.index.min())
            max_idx = min(X.index.max(), y.index.max())
            
            X = X.loc[min_idx:max_idx]
            y = y.loc[min_idx:max_idx]
            
            if len(X) < self.min_data_points:
                logger.warning(f"Insufficient data for linear regression: {len(X)} < {self.min_data_points}")
                return None, None
            
            # Scale features and train model
            X_scaled = self.scaler.fit_transform(X)
            self.linear_model.fit(X_scaled, y)
            
            # Make prediction on latest data
            latest_features = X.iloc[-1:].values
            latest_scaled = self.scaler.transform(latest_features)
            prediction = self.linear_model.predict(latest_scaled)[0]
            
            # Calculate feature importance (coefficients)
            importance = dict(zip(available_features, self.linear_model.coef_))
            top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            logger.info(f"Top 5 important features: {top_features}")
            
            logger.info("Linear regression model trained successfully")
            return self.linear_model, prediction
            
        except Exception as e:
            logger.error(f"Linear regression training failed: {e}")
            return None, None
    
    def generate_signal(self, df, symbol):
        """
        Generate trading signal using Linear Regression only
        
        Returns: 'BUY', 'SELL', or 'HOLD'
        """
        if len(df) < self.min_data_points:
            logger.warning("Insufficient data for signal generation")
            return 'HOLD'
        
        # Create features
        df_features = self.create_features(df)
        
        # Train Linear Regression model  
        lr_model, lr_prediction = self.train_linear_model(df_features)
        
        # Generate signal based on Linear Regression prediction
        if lr_prediction is not None:
            if lr_prediction > self.confidence_threshold:
                decision = 'BUY'
            elif lr_prediction < -self.confidence_threshold:
                decision = 'SELL'
            else:
                decision = 'HOLD'
        else:
            decision = 'HOLD'
        
        logger.info(f"Generated signal for {symbol}: {decision}")
        logger.info(f"LR prediction: {lr_prediction:.4f} (threshold: ±{self.confidence_threshold})")
        
        return decision

    def execute_trade(self, symbol, signal, quantity=100):
        """Execute trade via Alpaca API"""
        try:
            # Get current position
            try:
                position = self.api.get_position(symbol)
                current_qty = int(position.qty)
            except:
                current_qty = 0
            
            # Check buying power
            try:
                account = self.api.get_account()
                buying_power = float(account.buying_power)

                # Get current price to establish trade value
                quote = self.api.get_latest_trade(symbol)
                stock_price = float(quote.price)
                trade_value = stock_price * quantity
            except Exception as e:
                logger.warning(f"Could not get account info: {e}")
                buying_power = 0
                trade_value = 0
            
            # Execute based on signals
            if signal == 'BUY':
                if trade_value > buying_power:
                    logger.warning(f"Insufficient buying power: need ${trade_value:.2f}, have ${buying_power:.2f}")
                    return
                
                self.api.submit_order(
                    symbol = symbol,
                    qty = quantity,
                    side = 'buy',
                    type = 'market',
                    time_in_force = 'gtc'
                )

                new_position = current_qty + quantity
                logger.info(f"Bought {quantity} shares of symbol. Position: {current_qty} → {new_position}")
                        
            elif signal == 'SELL':
                
                if current_qty > 0:
                    sell_qty = min(current_qty, quantity)
                    self.api.submit_order(
                        symbol = symbol,
                        qty = sell_qty,
                        side = 'sell',
                        type = 'market',
                        time_in_force = 'gtc'
                    )
                    new_position = current_qty - sell_qty
                    logger.info(f"Sold {sell_qty} shares of {symbol}. Position: {current_qty} → {new_position}")

        except Exception as  e:
            logger.error(f"Trade execution failed: {e}")


    def run_algorithm(self, data_file, symbol, quantity=100):
        """
        Main function to run the trading algorithm
        
        Args:
            data_file: Path to CSV file with stock data
            symbol: Stock symbol to trade
            quantity: Number of shares to trade
        """
        logger.info(f"Running algorithm for {symbol}")
        
        # Load data
        df = self.load_data(data_file)
        if df is None:
            logger.error("Failed to load data")
            return
        
        # Generate signal
        signal = self.generate_signal(df, symbol)
        
        # Execute trade
        self.execute_trade(symbol, signal, quantity)
        
        logger.info(f"Algorithm completed for {symbol}")

def main():
   """Main function to test the trading algorithm"""
   try:
       # Initialize the trading algorithm
       algo = TradingAlgorithm(api_key, secret_key, base_url)
       
       # Test parameters
       data_file = csv_path
       symbol = "NVDA"  # Replace with your desired stock symbol
       quantity = 100   # Number of shares to trade
       
       # Run the algorithm
       algo.run_algorithm(data_file, symbol, quantity)
       
   except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
   main()