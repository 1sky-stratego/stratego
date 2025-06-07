import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Import the trading algorithm
from pick_a_ticker_strategy import TradingAlgorithm  

# Path setup
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTickerTradingManager:
    def __init__(self, csv_file_path, api_key=None, secret_key=None, base_url=None):
        """
        Initialize the multi-ticker trading manager
        
        Args:
            csv_file_path: Path to CSV file with historical data
            api_key: Alpaca API key (optional for backtesting)
            secret_key: Alpaca secret key (optional for backtesting)
            base_url: Alpaca API base URL (optional for backtesting)
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.trained_model = None
        self.scaler = None
        
        # Initialize trading algorithm (can be None for backtesting mode)
        if api_key and secret_key and base_url:
            self.algo = TradingAlgorithm(api_key, secret_key, base_url)
            self.live_trading = True
        else:
            # Create algorithm instance without API for model training
            self.algo = TradingAlgorithm("dummy", "dummy", "dummy")
            self.live_trading = False
            logger.info("Running in backtesting mode (no live trading)")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load CSV data and prepare it for analysis"""
        try:
            logger.info(f"Loading data from {self.csv_file_path}")
            
            # Load the CSV
            df = pd.read_csv(self.csv_file_path)
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['symbol', 'date'])
            
            # Validate required columns
            required_columns = [
                'date', 'open', 'high', 'low', 'close', 'volume', 'symbol',
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20',
                'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'bb_width', 'volume_ratio',
                'momentum_5', 'momentum_10', 'momentum_20', 'daily_momentum',
                'volatility', 'price_to_day_high', 'price_to_day_low', 'daily_range_position',
                'price_to_52w_high', 'price_to_52w_low', '52w_range_position',
                'ytd_momentum', 'support_distance', 'resistance_distance'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            
            # Store the data
            self.data = df
            self.available_symbols = sorted(df['symbol'].unique())
            
            logger.info(f"Loaded data for {len(self.available_symbols)} symbols")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"Total records: {len(df):,}")
            logger.info(f"Available symbols: {self.available_symbols}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def train_global_model(self, training_symbols=None, min_records_per_symbol=100):
        """
        Train a single model using data from multiple symbols
        
        Args:
            training_symbols: List of symbols to use for training (None = use all)
            min_records_per_symbol: Minimum records required per symbol
        """
        try:
            logger.info("Training global model...")
            
            if training_symbols is None:
                training_symbols = self.available_symbols
            
            # Collect training data from multiple symbols
            training_data_list = []
            
            for symbol in training_symbols:
                symbol_data = self.data[self.data['symbol'] == symbol].copy()
                
                if len(symbol_data) < min_records_per_symbol:
                    logger.warning(f"Skipping {symbol}: insufficient data ({len(symbol_data)} < {min_records_per_symbol})")
                    continue
                
                # Set index for the algorithm's expected format
                symbol_data.set_index('date', inplace=True)
                training_data_list.append(symbol_data)
                logger.info(f"Added {symbol} to training set: {len(symbol_data)} records")
            
            if not training_data_list:
                raise ValueError("No symbols have sufficient data for training")
            
            # Combine all training data
            combined_data = pd.concat(training_data_list, ignore_index=False)
            logger.info(f"Combined training data: {len(combined_data)} records from {len(training_data_list)} symbols")
            
            # Train the model using the algorithm's method
            model, _ = self.algo.train_linear_model(combined_data)
            
            if model is not None:
                self.trained_model = model
                self.scaler = self.algo.scaler  # Store the fitted scaler
                logger.info("Global model training completed successfully")
                return True
            else:
                logger.error("Model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error training global model: {e}")
            return False

    def generate_signal_for_symbol(self, symbol, use_latest_n_days=30):
        """
        Generate trading signal for a specific symbol using the trained model
        
        Args:
            symbol: Stock symbol
            use_latest_n_days: Number of latest days to use for signal generation
            
        Returns:
            dict: Signal information
        """
        try:
            if self.trained_model is None:
                logger.error("No trained model available. Run train_global_model() first.")
                return None
            
            # Get symbol data
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return None
            
            # Use latest N days for signal generation
            symbol_data = symbol_data.tail(use_latest_n_days)
            symbol_data.set_index('date', inplace=True)
            
            # Feature columns that the model expects
            feature_cols = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20',
                'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'bb_width',
                'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20',
                'volatility', 'price_to_day_high', 'price_to_day_low', 'daily_range_position',
                'daily_momentum', 'price_to_52w_high', 'price_to_52w_low', '52w_range_position',
                'ytd_momentum', 'support_distance', 'resistance_distance'
            ]
            
            # Filter available features
            available_features = [col for col in feature_cols if col in symbol_data.columns and not symbol_data[col].isna().all()]
            
            if not available_features:
                logger.warning(f"No valid features available for {symbol}")
                return None
            
            # Get latest features
            latest_features = symbol_data[available_features].iloc[-1:].dropna()
            
            if latest_features.empty:
                logger.warning(f"No valid latest features for {symbol}")
                return None
            
            # Make prediction using the trained model
            latest_scaled = self.scaler.transform(latest_features.values)
            prediction = self.trained_model.predict(latest_scaled)[0]
            
            # Generate signal
            if prediction > self.algo.confidence_threshold:
                signal = 'BUY'
            elif prediction < -self.algo.confidence_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Get current price and additional info
            current_price = symbol_data['close'].iloc[-1]
            
            signal_info = {
                'symbol': symbol,
                'signal': signal,
                'prediction': prediction,
                'confidence_threshold': self.algo.confidence_threshold,
                'current_price': current_price,
                'timestamp': symbol_data.index[-1],
                'available_features_count': len(available_features)
            }
            
            logger.info(f"{symbol}: {signal} (prediction: {prediction:.4f}, price: ${current_price:.2f})")
            return signal_info
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def run_signals_for_symbols(self, symbols_list, execute_trades=False, quantity=10):
        """
        Generate signals for multiple symbols and optionally execute trades
        
        Args:
            symbols_list: List of symbols to analyze
            execute_trades: Whether to execute trades via Alpaca API
            quantity: Number of shares to trade per signal
            
        Returns:
            list: List of signal information for each symbol
        """
        logger.info(f"Generating signals for {len(symbols_list)} symbols...")
        
        results = []
        successful_signals = 0
        
        for i, symbol in enumerate(symbols_list, 1):
            logger.info(f"Processing {symbol} ({i}/{len(symbols_list)})")
            
            try:
                # Generate signal
                signal_info = self.generate_signal_for_symbol(symbol)
                
                if signal_info:
                    results.append(signal_info)
                    successful_signals += 1
                    
                    # Execute trade if requested and in live trading mode
                    if execute_trades and self.live_trading and signal_info['signal'] != 'HOLD':
                        try:
                            self.algo.execute_trade(symbol, signal_info['signal'], quantity)
                            signal_info['trade_executed'] = True
                        except Exception as e:
                            logger.error(f"Trade execution failed for {symbol}: {e}")
                            signal_info['trade_executed'] = False
                            signal_info['trade_error'] = str(e)
                    else:
                        signal_info['trade_executed'] = False
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'signal': 'ERROR',
                    'error': str(e)
                })
        
        logger.info(f"Signal generation completed. Successful: {successful_signals}/{len(symbols_list)}")
        return results

    def get_trading_summary(self, results):
        """Generate a summary of trading signals"""
        if not results:
            return "No results to summarize"
        
        # Count signals
        signal_counts = {}
        for result in results:
            if 'signal' in result:
                signal = result['signal']
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        # Find top predictions
        valid_results = [r for r in results if 'prediction' in r and 'signal' in r and r['signal'] != 'ERROR']
        
        summary = f"""
=== TRADING SIGNALS SUMMARY ===
Total symbols processed: {len(results)}
Signal distribution: {signal_counts}

Top 5 BUY signals (highest prediction):
"""
        
        buy_signals = [r for r in valid_results if r['signal'] == 'BUY']
        buy_signals.sort(key=lambda x: x['prediction'], reverse=True)
        
        for signal in buy_signals[:5]:
            summary += f"  {signal['symbol']}: {signal['prediction']:.4f} (${signal['current_price']:.2f})\n"
        
        summary += "\nTop 5 SELL signals (lowest prediction):\n"
        
        sell_signals = [r for r in valid_results if r['signal'] == 'SELL']
        sell_signals.sort(key=lambda x: x['prediction'])
        
        for signal in sell_signals[:5]:
            summary += f"  {signal['symbol']}: {signal['prediction']:.4f} (${signal['current_price']:.2f})\n"
        
        return summary

def main():
    """Main function to run the multi-ticker trading system"""
    try:
        # Configuration
        csv_file_path = "src/data/collected/training_data.csv"  
        
        # Get API credentials (optional - set to None for backtesting mode)
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_API_BASE_URL')
        
        # Define symbols to trade (you can customize this list)
        target_symbols = os.getenv('TARGET_STOCKS').split(",")
        
        # Initialize the trading manager
        manager = MultiTickerTradingManager(
            csv_file_path=csv_file_path,
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        
        # Train the global model
        logger.info("Training global model...")
        training_success = manager.train_global_model()
        
        if not training_success:
            logger.error("Failed to train model. Exiting.")
            return
        
        # Generate signals for target symbols
        logger.info("Generating signals for target symbols...")
        results = manager.run_signals_for_symbols(
            symbols_list=target_symbols,
            execute_trades=False,  # Set to True to execute actual trades
            quantity=10
        )
        
        # Print summary
        summary = manager.get_trading_summary(results)
        print(summary)
        
        # Save results to file
        results_df = pd.DataFrame(results)
        output_file = f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()