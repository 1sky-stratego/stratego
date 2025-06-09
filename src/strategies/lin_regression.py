import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
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

# Model paths
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import parameters
PARAM_prediction_horizon = int(os.getenv('PARAM_PREDICTION_HORIZON', 5))
PARAM_confidence_threshold = float(os.getenv('PARAM_CONFIDENCE_THRESHOLD', 0.02))
PARAM_min_data_points = int(os.getenv('PARAM_MIN_DATA_POINTS', 100))
PARAM_market_confidence_threshold = float(os.getenv('PARAM_MARKET_CONFIDENCE_THRESHOLD', 0.6))

# Stocks
target_stocks = os.getenv('TARGET_STOCKS').split(',')


class TradingAlgorithm:
    def __init__(self):
        """
        Initialize trading algorithm for training and prediction
        """
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.feature_columns = []
        self.is_trained = False
        
        # Algorithm parameters
        self.prediction_horizon = PARAM_prediction_horizon
        self.confidence_threshold = PARAM_confidence_threshold
        self.min_data_points = PARAM_min_data_points
        self.market_confidence_threshold = PARAM_market_confidence_threshold
        
        logger.info("Trading algorithm initialized")

    def load_data(self, file_path):
        """
        Load stock data from CSV file with your specific column format
        
        Expected columns: date,open,high,low,close,volume,dividends,stock splits,symbol,name,sma_5,sma_10,sma_20,sma_50,ema_5,ema_10,ema_20,price_to_sma20,price_to_sma50,sma20_to_sma50,rsi,macd,macd_signal,macd_histogram,bb_middle,bb_upper,bb_lower,bb_position,bb_width,volume_ratio,momentum_5,momentum_10,momentum_20,daily_momentum,volatility,price_to_day_high,price_to_day_low,daily_range_position,52w_high,52w_low,price_to_52w_high,price_to_52w_low,52w_range_position,ytd_momentum,support_level,resistance_level,support_distance,resistance_distance,market_cap_category,data_collected_at,capital gains
        """
        try:
            df = pd.read_csv(file_path) 
            
            # Convert numeric columns
            numeric_columns = [
                'pen', 'high', 'low', 'close', 'volume', 'dividends',
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20',
                'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50', 'rsi',
                'macd', 'macd_signal', 'macd_histogram', 'bb_middle', 'bb_upper', 'bb_lower',
                'bb_position', 'bb_width', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20',
                'daily_momentum', 'volatility', 'price_to_day_high', 'price_to_day_low',
                'daily_range_position', '52w_high', '52w_low', 'price_to_52w_high',
                'price_to_52w_low', '52w_range_position', 'ytd_momentum', 'support_level',
                'resistance_level', 'support_distance', 'resistance_distance', 'capital gains'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Map to standard format for compatibility
            df['day_high'] = df.get('high', df['close'])
            df['day_low'] = df.get('low', df['close'])
            df['yr_high'] = df.get('52w_high')
            df['yr_low'] = df.get('52w_low')
            
            # Check for required data
            if df['close'].isna().all():
                logger.error("No valid price data found")
                return None
            
            # Handle missing volume
            if 'volume' not in df.columns or df['volume'].isna().all():
                df['volume'] = 1000000  # Default volume
                logger.warning("Volume data missing, using default values")
            
            # Handle stock splits column name
            if 'stock splits' in df.columns:
                df['stock_splits'] = df['stock splits']
            
            logger.info(f"Loaded {len(df)} data points from {file_path}")
            logger.info(f"Symbols: {df['symbol'].unique()}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df.sort_values(['symbol', 'timestamp'])
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
        
    def create_features(self, df):
        """Create or validate technical indicators - most should already be calculated"""
        df = df.copy()
        
        # Sort by symbol and timestamp for proper calculations
        df = df.sort_values(['symbol', 'timestamp'])
        
        # Group by symbol to fill any missing features
        def validate_and_fill_features(group):
            group = group.sort_values('timestamp')
            
            # Only calculate missing features - most should already exist
            
            # Fill missing moving averages if needed
            if 'sma_5' not in group.columns or group['sma_5'].isna().all():
                group['sma_5'] = group['close'].rolling(5, min_periods=1).mean()
            
            if 'sma_10' not in group.columns or group['sma_10'].isna().all():
                group['sma_10'] = group['close'].rolling(10, min_periods=1).mean()
            
            if 'sma_20' not in group.columns or group['sma_20'].isna().all():
                group['sma_20'] = group['close'].rolling(20, min_periods=1).mean()
            
            if 'sma_50' not in group.columns or group['sma_50'].isna().all():
                group['sma_50'] = group['close'].rolling(50, min_periods=1).mean()
            
            # Fill missing EMAs if needed
            if 'ema_5' not in group.columns or group['ema_5'].isna().all():
                group['ema_5'] = group['close'].ewm(span=5, min_periods=1).mean()
            
            if 'ema_10' not in group.columns or group['ema_10'].isna().all():
                group['ema_10'] = group['close'].ewm(span=10, min_periods=1).mean()
            
            if 'ema_20' not in group.columns or group['ema_20'].isna().all():
                group['ema_20'] = group['close'].ewm(span=20, min_periods=1).mean()
            
            # Fill missing RSI if needed
            if 'rsi' not in group.columns or group['rsi'].isna().all():
                delta = group['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                rs = gain / (loss + 1e-10)
                group['rsi'] = 100 - (100 / (1 + rs))
            
            # Fill missing MACD if needed
            if 'macd' not in group.columns or group['macd'].isna().all():
                ema_12 = group['close'].ewm(span=12, min_periods=1).mean()
                ema_26 = group['close'].ewm(span=26, min_periods=1).mean()
                group['macd'] = ema_12 - ema_26
                group['macd_signal'] = group['macd'].ewm(span=9, min_periods=1).mean()
                group['macd_histogram'] = group['macd'] - group['macd_signal']
            
            # Fill missing price ratios if needed
            if 'price_to_sma20' not in group.columns or group['price_to_sma20'].isna().all():
                group['price_to_sma20'] = np.where(group['sma_20'] > 0, group['close'] / group['sma_20'], 1)
            
            if 'price_to_sma50' not in group.columns or group['price_to_sma50'].isna().all():
                group['price_to_sma50'] = np.where(group['sma_50'] > 0, group['close'] / group['sma_50'], 1)
            
            if 'sma20_to_sma50' not in group.columns or group['sma20_to_sma50'].isna().all():
                group['sma20_to_sma50'] = np.where(group['sma_50'] > 0, group['sma_20'] / group['sma_50'], 1)
            
            # Fill missing Bollinger Bands if needed
            if 'bb_position' not in group.columns or group['bb_position'].isna().all():
                bb_period = 20
                bb_std = 2
                bb_middle = group['close'].rolling(bb_period, min_periods=1).mean()
                bb_std_dev = group['close'].rolling(bb_period, min_periods=1).std()
                bb_upper = bb_middle + (bb_std_dev * bb_std)
                bb_lower = bb_middle - (bb_std_dev * bb_std)
                bb_range = bb_upper - bb_lower
                group['bb_position'] = np.where(bb_range > 0, 
                                              (group['close'] - bb_lower) / bb_range, 
                                              0.5)
                group['bb_width'] = np.where(bb_middle > 0, bb_range / bb_middle, 0)
            
            # Fill missing volume ratio if needed
            if 'volume_ratio' not in group.columns or group['volume_ratio'].isna().all():
                volume_sma = group['volume'].rolling(20, min_periods=1).mean()
                group['volume_ratio'] = np.where(volume_sma > 0, group['volume'] / volume_sma, 1)
            
            # Fill missing momentum if needed
            if 'momentum_5' not in group.columns or group['momentum_5'].isna().all():
                group['momentum_5'] = group['close'].pct_change(5)
            
            if 'momentum_10' not in group.columns or group['momentum_10'].isna().all():
                group['momentum_10'] = group['close'].pct_change(10)
            
            if 'momentum_20' not in group.columns or group['momentum_20'].isna().all():
                group['momentum_20'] = group['close'].pct_change(20)
            
            # Fill missing volatility if needed
            if 'volatility' not in group.columns or group['volatility'].isna().all():
                group['volatility'] = group['close'].pct_change().rolling(20, min_periods=1).std()
            
            # Fill missing daily positioning if needed
            if 'daily_range_position' not in group.columns or group['daily_range_position'].isna().all():
                day_range = group['high'] - group['low']
                group['daily_range_position'] = np.where(day_range > 0, 
                                                        (group['close'] - group['low']) / day_range, 
                                                        0.5)
            
            # Fill missing 52-week positioning if needed
            if '52w_range_position' not in group.columns or group['52w_range_position'].isna().all():
                if '52w_high' in group.columns and '52w_low' in group.columns:
                    yr_range = group['52w_high'] - group['52w_low']
                    group['52w_range_position'] = np.where(yr_range > 0, 
                                                         (group['close'] - group['52w_low']) / yr_range, 
                                                         0.5)
            
            return group
        
        # Apply feature validation to each stock
        df_features = df.groupby('symbol').apply(validate_and_fill_features).reset_index(drop=True)
        
        # Fill NaN values
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
        return df_features

    def calculate_market_sentiment(self, df):
        """
        Calculate overall market sentiment from all stocks
        
        Returns:
            dict: Market sentiment metrics
        """
        try:
            latest_data = df.groupby('symbol').last().reset_index()
            
            sentiment_metrics = {}
            
            # Overall market momentum
            sentiment_metrics['avg_momentum_5'] = latest_data['momentum_5'].mean()
            sentiment_metrics['avg_momentum_10'] = latest_data['momentum_10'].mean()
            sentiment_metrics['avg_momentum_20'] = latest_data['momentum_20'].mean()
            
            # Daily momentum if available
            if 'daily_momentum' in latest_data.columns:
                sentiment_metrics['avg_daily_momentum'] = latest_data['daily_momentum'].mean()
            
            # Market volatility
            sentiment_metrics['avg_volatility'] = latest_data['volatility'].mean()
            
            # RSI distribution
            sentiment_metrics['avg_rsi'] = latest_data['rsi'].mean()
            sentiment_metrics['oversold_pct'] = (latest_data['rsi'] < 30).mean()
            sentiment_metrics['overbought_pct'] = (latest_data['rsi'] > 70).mean()
            
            # Price positioning
            if '52w_range_position' in latest_data.columns:
                sentiment_metrics['avg_52w_position'] = latest_data['52w_range_position'].mean()
            
            # YTD performance
            if 'ytd_momentum' in latest_data.columns:
                sentiment_metrics['avg_ytd_return'] = latest_data['ytd_momentum'].mean()
                sentiment_metrics['positive_ytd_pct'] = (latest_data['ytd_momentum'] > 0).mean()
            
            # Capital gains if available
            if 'capital gains' in latest_data.columns:
                sentiment_metrics['avg_capital_gains'] = latest_data['capital gains'].mean()
                sentiment_metrics['positive_gains_pct'] = (latest_data['capital gains'] > 0).mean()
            
            # Volume activity
            sentiment_metrics['avg_volume_ratio'] = latest_data['volume_ratio'].mean()
            
            # MACD signals
            sentiment_metrics['macd_bullish_pct'] = (latest_data['macd'] > latest_data['macd_signal']).mean()
            
            # Moving average trends
            sentiment_metrics['above_sma20_pct'] = (latest_data['price_to_sma20'] > 1).mean()
            sentiment_metrics['above_sma50_pct'] = (latest_data['price_to_sma50'] > 1).mean()
            sentiment_metrics['sma20_above_sma50_pct'] = (latest_data['sma20_to_sma50'] > 1).mean()
            
            # Bollinger Bands positioning
            sentiment_metrics['avg_bb_position'] = latest_data['bb_position'].mean()
            sentiment_metrics['bb_squeeze_pct'] = (latest_data['bb_width'] < latest_data['bb_width'].quantile(0.2)).mean()
            
            logger.info(f"Market sentiment calculated: {sentiment_metrics}")
            return sentiment_metrics
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {}

    def train_model(self, data_file, save_model=True):
        """
        Train the model on historical data with your specific column format
        
        Args:
            data_file: Path to historical data CSV
            save_model: Whether to save the trained model
        """
        logger.info("Starting model training...")
        
        # Load and prepare data
        df = self.load_data(data_file)
        if df is None:
            logger.error("Failed to load training data")
            return False
        
        # Create/validate features
        df_features = self.create_features(df)
        
        # Define feature columns based on your data structure
        feature_cols = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20',
            'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'bb_width', 'volume_ratio', 
            'momentum_5', 'momentum_10', 'momentum_20', 'volatility',
            'support_distance', 'resistance_distance'
        ]
        
        # Add your specific features if they exist
        additional_features = [
            'daily_momentum', 'price_to_day_high', 'price_to_day_low', 'daily_range_position',
            'price_to_52w_high', 'price_to_52w_low', '52w_range_position',
            'ytd_momentum', 'capital gains'
        ]
        
        for feature in additional_features:
            if feature in df_features.columns:
                feature_cols.append(feature)
        
        # Filter available features
        self.feature_columns = [col for col in feature_cols 
                               if col in df_features.columns and not df_features[col].isna().all()]
        
        logger.info(f"Using {len(self.feature_columns)} features for training: {self.feature_columns}")
        
        # Prepare training data by symbol
        X_list = []
        y_list = []
        
        for symbol in df_features['symbol'].unique():
            symbol_data = df_features[df_features['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} points")
                continue
            
            # Create features and targets
            X_symbol = symbol_data[self.feature_columns].copy()

            # Create target: future price change percentage
            future_prices = symbol_data['close'].shift(-self.prediction_horizon)
            current_prices = symbol_data['close']
            y_symbol = (future_prices - current_prices) / current_prices

            # Combine features and target into one DataFrame for synchronized NaN removal
            combined = X_symbol.copy()
            combined['target'] = y_symbol

            # Drop any rows with NaNs in features or target
            combined_clean = combined.dropna()

            if len(combined_clean) > 10:  # Minimum samples per symbol
                X_symbol_aligned = combined_clean[self.feature_columns]
                y_symbol_aligned = combined_clean['target']
                
                X_list.append(X_symbol_aligned)
                y_list.append(y_symbol_aligned)
                logger.info(f"Added {len(X_symbol_aligned)} samples from {symbol}")
            else:
                logger.warning(f"Insufficient clean data for {symbol} after NaN removal: {len(combined_clean)} samples")


        
        if not X_list:
            logger.error("No valid training data found")
            return False
        
        # Combine all data
        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)
        
        
        logger.info(f"Training on {len(X)} samples from {len(X_list)} symbols")
        
        # Scale features and train model
        X_scaled = self.scaler.fit_transform(X)
        self.linear_model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Model evaluation
        train_score = self.linear_model.score(X_scaled, y)
        logger.info(f"Model R² score: {train_score:.4f}")
        
        # Feature importance
        importance = dict(zip(self.feature_columns, self.linear_model.coef_))
        top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        logger.info(f"Top 10 important features: {top_features}")
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        logger.info("Model training completed successfully")
        logger.info("Training Success: True")
        return 0

    def save_model(self):
        """Save trained model and scaler"""
        try:
            model_data = {
                'linear_model': self.linear_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'parameters': {
                    'prediction_horizon': self.prediction_horizon,
                    'confidence_threshold': self.confidence_threshold,
                    'min_data_points': self.min_data_points,
                    'market_confidence_threshold': self.market_confidence_threshold
                }
            }
            
            model_path = os.path.join(MODEL_DIR, 'trading_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self):
        """Load trained model and scaler"""
        try:
            model_path = os.path.join(MODEL_DIR, 'trading_model.pkl')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.linear_model = model_data['linear_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            
            # Load parameters
            params = model_data.get('parameters', {})
            self.prediction_horizon = params.get('prediction_horizon', self.prediction_horizon)
            self.confidence_threshold = params.get('confidence_threshold', self.confidence_threshold)
            self.min_data_points = params.get('min_data_points', self.min_data_points)
            self.market_confidence_threshold = params.get('market_confidence_threshold', self.market_confidence_threshold)
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_signals(self, recent_data_file):
        """
        Generate trading signals for recent data
        
        Args:
            recent_data_file: Path to recent data CSV
            
        Returns:
            dict: Signals for each symbol and market analysis
        """
        if not self.is_trained:
            logger.error("Model not trained. Please train model first or load a trained model.")
            return {}
        
        # Load recent data
        df = self.load_data(recent_data_file)
        if df is None:
            logger.error("Failed to load recent data")
            return {}
        
        # Create/validate features
        df_features = self.create_features(df)
        
        # Calculate market sentiment
        market_sentiment = self.calculate_market_sentiment(df_features)
        
        # Generate signals for each symbol
        signals = {}
        
        for symbol in df_features['symbol'].unique():
            try:
                symbol_data = df_features[df_features['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) == 0:
                    continue
                
                # Get latest data point
                latest_data = symbol_data.iloc[-1]
                
                # Prepare features
                feature_values = []
                for feature in self.feature_columns:
                    if feature in latest_data.index:
                        feature_values.append(latest_data[feature])
                    else:
                        logger.warning(f"Missing feature {feature} for {symbol}")
                        feature_values.append(0.0)
                
                X = np.array(feature_values).reshape(1, -1)
                
                # Handle missing features
                if pd.isna(X).any():
                    logger.warning(f"NaN features for {symbol}, filling with zeros")
                    X = np.nan_to_num(X, 0)
                
                # Scale features and predict
                X_scaled = self.scaler.transform(X)
                prediction = self.linear_model.predict(X_scaled)
                
                # Adjust prediction based on market sentiment
                market_adjustment = self._calculate_market_adjustment(market_sentiment)
                adjusted_prediction = prediction * market_adjustment
                
                # Generate signal
                if adjusted_prediction > self.confidence_threshold:
                    signal = 'BUY'
                elif adjusted_prediction < -self.confidence_threshold:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                signals[symbol] = {
                    'signal': signal,
                    'raw_prediction': prediction,
                    'adjusted_prediction': adjusted_prediction,
                    'market_adjustment': market_adjustment,
                    'confidence': abs(adjusted_prediction),
                    'current_price': latest_data.get('close', 0),
                    'timestamp': latest_data.get('timestamp', 'N/A')
                }
                
                logger.info(f"{symbol}: {signal} (confidence: {abs(adjusted_prediction):.4f})")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        result = {
            'signals': signals,
            'market_sentiment': market_sentiment,
            'market_adjustment': market_adjustment,
            'total_symbols': len(signals),
            'buy_signals': sum(1 for s in signals.values() if s['signal'] == 'BUY'),
            'sell_signals': sum(1 for s in signals.values() if s['signal'] == 'SELL'),
            'hold_signals': sum(1 for s in signals.values() if s['signal'] == 'HOLD')
        }
        
        logger.info(f"Generated signals for {len(signals)} symbols")
        logger.info(f"Buy: {result['buy_signals']}, Sell: {result['sell_signals']}, Hold: {result['hold_signals']}")
        
        return result

    def _calculate_market_adjustment(self, market_sentiment):
        """
        Calculate market adjustment factor based on overall market sentiment
        
        Args:
            market_sentiment: Dict of market sentiment metrics
            
        Returns:
            float: Adjustment factor (0.5 to 1.5)
        """
        if not market_sentiment:
            return 1.0
        
        bullish_factors = 0
        bearish_factors = 0
        total_factors = 0
        
        # Momentum factors
        if market_sentiment.get('avg_momentum_5', 0) > 0:
            bullish_factors += 1
        else:
            bearish_factors += 1
        total_factors += 1
        
        # Daily momentum if available
        if 'avg_daily_momentum' in market_sentiment:
            if market_sentiment['avg_daily_momentum'] > 0:
                bullish_factors += 1
            else:
                bearish_factors += 1
            total_factors += 1
        
        # RSI factors
        if market_sentiment.get('oversold_pct', 0) > 0.3:  # More than 30% oversold
            bullish_factors += 1
        elif market_sentiment.get('overbought_pct', 0) > 0.3:  # More than 30% overbought
            bearish_factors += 1
        total_factors += 1
        
        # Moving average factors
        if market_sentiment.get('above_sma20_pct', 0.5) > 0.6:  # 60% above SMA20
            bullish_factors += 1
        elif market_sentiment.get('above_sma20_pct', 0.5) < 0.4:  # Less than 40% above SMA20
            bearish_factors += 1
        total_factors += 1
        
        # MACD factors
        if market_sentiment.get('macd_bullish_pct', 0.5) > 0.6:
            bullish_factors += 1
        elif market_sentiment.get('macd_bullish_pct', 0.5) < 0.4:
            bearish_factors += 1
        total_factors += 1
        
        # Capital gains factor if available
        if 'positive_gains_pct' in market_sentiment:
            if market_sentiment['positive_gains_pct'] > 0.6:
                bullish_factors += 1
            elif market_sentiment['positive_gains_pct'] < 0.4:
                bearish_factors += 1
            total_factors += 1
        
        # Calculate adjustment
        if total_factors > 0:
            bullish_ratio = bullish_factors / total_factors
            adjustment = 0.7 + (bullish_ratio * 0.6)  # Range: 0.7 to 1.3
        else:
            adjustment = 1.0
        
        logger.info(f"Market adjustment factor: {adjustment:.3f} (bullish: {bullish_factors}/{total_factors})")
        return adjustment

def train_model(data_file):
    """Train and save the model"""
    algo = TradingAlgorithm()
    success = algo.train_model(data_file, save_model=True)
    
    if success:
        logger.info("Training completed successfully")
    else:
        logger.error("Training failed")
    
    return success


def predict_signals(recent_data_file):
    """Load model and generate predictions"""
    algo = TradingAlgorithm()
    
    # Try to load existing model
    if not algo.load_model():
        logger.error("No trained model found. Please train model first.")
        return {}
    
    # Generate signals
    results = algo.predict_signals(recent_data_file)
    return results


def main():
    """Main function for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py train <data_file>")
        print("  python script.py predict <recent_data_file>")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'train':
        if len(sys.argv) < 3:
            print("Please provide data file for training")
            return
        data_file = sys.argv[2]
        train_model(data_file)
        
    elif mode == 'predict':
        if len(sys.argv) < 3:
            print("Please provide recent data file for prediction")
            return
        recent_data_file = sys.argv[2]
        results = predict_signals(recent_data_file)
        
        if results:
            print("\n=== MARKET ANALYSIS ===")
            print(f"Market adjustment factor: {results['market_adjustment']:.3f}")
            print(f"Total symbols analyzed: {results['total_symbols']}")
            print(f"Buy signals: {results['buy_signals']}")
            print(f"Sell signals: {results['sell_signals']}")
            print(f"Hold signals: {results['hold_signals']}")
            
            print("\n=== INDIVIDUAL SIGNALS ===")
            for symbol, signal_data in results['signals'].items():
                print(f"{symbol}: {signal_data['signal']} (confidence: {signal_data['confidence']:.4f})")
        
    else:
        print("Invalid mode. Use 'train' or 'predict'")


if __name__ == "__main__":
    main()