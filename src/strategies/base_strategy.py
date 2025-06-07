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
        self.prediction_horizon = 5
        self.confidence_threshold = 0.02  # 2% threshold for stronger signals
        self.min_data_points = 50
        
        logger.info("Trading algorithm initialized")
    
    def load_data(self, file_path):
        """
        Load stock data from CSV file
        
        Your CSV format: symbol,name,current_price,avg_volume_30d,price_52w_high,price_52w_low,ytd_return,volatility_annualized,sma_20,sma_50,market_cap_category,data_collected_at,bars_count
        """
        try:
            df = pd.read_csv(file_path)
            
            # Your actual column names
            expected_columns = [
                'symbol', 'name', 'current_price', 'avg_volume_30d', 'price_52w_high', 
                'price_52w_low', 'ytd_return', 'volatility_annualized', 'sma_20', 
                'sma_50', 'market_cap_category', 'data_collected_at', 'bars_count'
            ]
            
            # If no headers, assign them
            if df.columns[0] == 0 or 'Unnamed' in str(df.columns[0]):
                df.columns = expected_columns
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['data_collected_at'])
            df.set_index('timestamp', inplace=True)
            
            # Map your columns to standard OHLCV format
            # Since you only have current_price, we'll use it for all OHLC
            df['close'] = pd.to_numeric(df['current_price'], errors='coerce')
            df['open'] = df['close']  # Use current price as open
            df['high'] = pd.to_numeric(df['price_52w_high'], errors='coerce') 
            df['low'] = pd.to_numeric(df['price_52w_low'], errors='coerce')
            df['volume'] = pd.to_numeric(df['avg_volume_30d'], errors='coerce')
            
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
        """Create technical indicators for linear regression - enhanced for your data format"""
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
        
        # 52-week high/low ratios (using your existing data)
        if 'price_52w_high' in df.columns and 'price_52w_low' in df.columns:
            df['price_to_52w_high'] = df['close'] / df['price_52w_high']
            df['price_to_52w_low'] = df['close'] / df['price_52w_low']
            df['52w_range_position'] = (df['close'] - df['price_52w_low']) / (df['price_52w_high'] - df['price_52w_low'])
        
        # YTD return momentum (if available)
        if 'ytd_return' in df.columns:
            df['ytd_momentum'] = df['ytd_return']
        
        # Support and Resistance levels
        df['support'] = df['low'].rolling(20).min()
        df['resistance'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support']) / df['close']
        df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
        
        return df
    
    def train_linear_model(self, df):
        """Train linear regression model - enhanced for your data format"""
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
                
                # 52-week positioning (if available)
                'price_to_52w_high', 'price_to_52w_low', '52w_range_position',
                
                # YTD performance (if available)
                'ytd_momentum',
                
                # Support/Resistance
                'support_distance', 'resistance_distance'
            ]
            
            # Filter features that actually exist in the dataframe
            available_features = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]
            
            if len(available_features) < 5:
                logger.warning(f"Only {len(available_features)} features available, using basic set")
                available_features = ['sma_20', 'rsi', 'macd', 'bb_position', 'volume_ratio']
                available_features = [col for col in available_features if col in df.columns]
            
            logger.info(f"Using {len(available_features)} features: {available_features}")
            
            # Create feature matrix
            X = df[available_features].dropna()
            
            # Create target: future price change
            future_returns = df['close'].shift(-self.prediction_horizon).pct_change()
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
            
            # Execute based on signal
            if signal == 'BUY' and current_qty <= 0:
                # Close short position and go long
                if current_qty < 0:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=abs(current_qty),
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    logger.info(f"Closed short position of {abs(current_qty)} shares")
                
                # Open long position
                self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"Bought {quantity} shares of {symbol}")
                
            elif signal == 'SELL' and current_qty >= 0:
                # Close long position and go short
                if current_qty > 0:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=current_qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    logger.info(f"Sold {current_qty} shares")
                
                # Open short position
                self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"Shorted {quantity} shares of {symbol}")
                
            else:
                logger.info(f"No action taken for {symbol} - Signal: {signal}, Current position: {current_qty}")
                
        except Exception as e:
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

# Backtesting Strategy Wrapper
from backtesting import Strategy
import backtesting

class LinearRegressionStrategy(Strategy):
    """
    Backtesting wrapper for Linear Regression strategy
    This class will be detected by your backtesting system
    """
    
    def init(self):
        """Initialize strategy - called once at start of backtest"""
        # Initialize our algorithm components (without Alpaca API for backtesting)
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        
        # Algorithm parameters
        self.prediction_horizon = 5
        self.confidence_threshold = 0.02
        self.min_data_points = 50
        
    def create_features_bt(self, close_prices, high_prices, low_prices, volume):
        """Create features for backtesting (uses backtesting data format)"""
        df = pd.DataFrame({
            'close': close_prices,
            'high': high_prices, 
            'low': low_prices,
            'volume': volume
        })
        
        # Create the same technical indicators as in the main algorithm
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        
        # RSI
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
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        return df
    
    def generate_signal_bt(self):
        """Generate trading signal for backtesting"""
        # Get current data length
        current_idx = len(self.data.Close) - 1
        
        if current_idx < self.min_data_points:
            return 'HOLD'
        
        # Get recent data for analysis
        lookback = min(100, current_idx + 1)
        close_prices = self.data.Close[-lookback:]
        high_prices = self.data.High[-lookback:]
        low_prices = self.data.Low[-lookback:]
        volume = self.data.Volume[-lookback:]
        
        try:
            # Linear regression prediction
            df_features = self.create_features_bt(close_prices, high_prices, low_prices, volume)
            feature_cols = ['sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'rsi', 
                           'macd', 'macd_signal', 'bb_position', 'volume_ratio', 
                           'momentum_5', 'momentum_10', 'volatility']
            
            X = df_features[feature_cols].dropna()
            
            if len(X) >= 30:
                # Create target for training
                future_returns = df_features['close'].shift(-self.prediction_horizon).pct_change()
                y = future_returns.dropna()
                
                # Align data
                min_len = min(len(X), len(y))
                X_train = X.iloc[:min_len-1]  # Leave last point for prediction
                y_train = y.iloc[:min_len-1]
                
                if len(X_train) > 10:
                    X_scaled = self.scaler.fit_transform(X_train)
                    self.linear_model.fit(X_scaled, y_train)
                    
                    # Predict on latest data
                    latest_features = X.iloc[-1:].values
                    latest_scaled = self.scaler.transform(latest_features)
                    lr_prediction = self.linear_model.predict(latest_scaled)[0]
                    
                    # Generate signal
                    if lr_prediction > self.confidence_threshold:
                        return 'BUY'
                    elif lr_prediction < -self.confidence_threshold:
                        return 'SELL'
                    else:
                        return 'HOLD'
                else:
                    return 'HOLD'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return 'HOLD'
    
    def next(self):
        """Called for each bar during backtesting"""
        signal = self.generate_signal_bt()
        
        if signal == 'BUY' and not self.position:
            self.buy()
        elif signal == 'SELL' and self.position:
            self.position.close()


# Example usage for live trading:
if __name__ == "__main__":
    algorithm = TradingAlgorithm(api_key, secret_key, base_url)
    
    # Run algorithm
    DATA_FILE = 'src/data/collected/ai_gpu_energy_stocks.csv'
    SYMBOL = "AAPL"  # Stock symbol
    QUANTITY = 100  # Number of shares
    
    algorithm.run_algorithm(DATA_FILE, SYMBOL, QUANTITY)