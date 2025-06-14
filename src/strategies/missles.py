import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
import yfinance as yf
import requests
import json
from textblob import TextBlob
import re
from collections import defaultdict
import alpaca_trade_api as tradeapi

# Path and env setup
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefenseStockModel:
    def __init__(self):
        self.defense_stocks = [
            'LMT',   # Lockheed Martin - missiles, defense systems
            'RTX',   # Raytheon Technologies - missiles, air defense
            'BA',    # Boeing - defense, aerospace
            'NOC',   # Northrop Grumman - defense systems
            'GD',    # General Dynamics - defense, aerospace
            'HII',   # Huntington Ingalls - naval defense
            'LHX',   # L3Harris - defense electronics
            'TDG',   # TransDigm Group - aerospace components
            'KTOS',  # Kratos Defense - unmanned systems
            'LDOS',  # Leidos - defense services
            'PLTR',  # Palantir - defense analytics
            'CW',    # Curtiss-Wright - defense components
        ]
        
        self.middle_east_keywords = [
            'missile', 'rocket', 'bomb', 'strike', 'attack', 'drone',
            'iran', 'israel', 'gaza', 'lebanon', 'syria', 'iraq',
            'hezbollah', 'hamas', 'houthis', 'middle east',
            'military action', 'air strike', 'defense system',
            'intercepted', 'launched', 'fired'
        ]
        
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        # Initialize Alpaca API
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.alpaca_base_url = os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if self.alpaca_api_key and self.alpaca_secret_key:
            self.api = tradeapi.REST(
                self.alpaca_api_key,
                self.alpaca_secret_key,
                self.alpaca_base_url,
                api_version='v2'
            )
        else:
            logger.warning("Alpaca API credentials not found. Trading functionality disabled.")
            self.api = None
        
        # Risk management parameters
        self.max_position_size = 0.025  # 2.5% of portfolio per trade
        self.target_portfolio_value = 90000  # Your mentioned portfolio value
        
    def fetch_news_data(self, days_back=30):
        """Fetch news articles related to Middle East military activities"""
        if not self.news_api_key:
            logger.warning("No NEWS_API_KEY found. Using mock data.")
            return self._generate_mock_news_data(days_back)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # NewsAPI endpoint
        url = "https://newsapi.org/v2/everything"
        
        # Search for Middle East military/conflict news
        query = "missile OR rocket OR bomb OR strike AND (Iran OR Israel OR Gaza OR Lebanon OR Syria OR Iraq)"
        
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,
            'apiKey': self.news_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            news_data = response.json()
            
            articles = []
            for article in news_data.get('articles', []):
                articles.append({
                    'date': pd.to_datetime(article['publishedAt']).date(),
                    'title': article['title'],
                    'description': article['description'] or '',
                    'content': article['content'] or '',
                    'url': article['url']
                })
            
            return pd.DataFrame(articles)
            
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return self._generate_mock_news_data(days_back)
    
    def _generate_mock_news_data(self, days_back):
        """Generate mock news data for testing"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now().date(), periods=days_back, freq='D')
        
        mock_articles = []
        for date in dates:
            # Random number of military events per day (0-5)
            num_events = np.random.poisson(1.5)
            
            for _ in range(num_events):
                event_types = ['missile launch', 'rocket attack', 'air strike', 'drone attack', 'intercepted missile']
                locations = ['Gaza', 'Lebanon', 'Syria', 'Iran', 'Israel', 'Iraq']
                
                event = np.random.choice(event_types)
                location = np.random.choice(locations)
                
                mock_articles.append({
                    'date': date.date(),
                    'title': f"{event} reported in {location}",
                    'description': f"Military activity involving {event} in {location} region",
                    'content': f"Reports indicate {event} in {location} with potential regional implications",
                    'url': f"https://mock-news.com/article-{len(mock_articles)}"
                })
        
        return pd.DataFrame(mock_articles)
    
    def calculate_military_activity_score(self, news_df):
        """Calculate daily military activity scores based on news content"""
        if news_df.empty:
            return pd.DataFrame()
        
        # Score each article based on keyword matches and sentiment
        scores = []
        
        for _, article in news_df.iterrows():
            text = f"{article['title']} {article['description']} {article['content']}".lower()
            
            # Count keyword matches
            keyword_score = 0
            for keyword in self.middle_east_keywords:
                keyword_score += text.count(keyword.lower())
            
            # Sentiment analysis (negative news often drives defense stocks up)
            sentiment = TextBlob(text).sentiment.polarity
            
            # Military action intensity keywords (weighted higher)
            intensity_keywords = ['fired', 'launched', 'strike', 'attack', 'bomb', 'missile']
            intensity_score = sum(text.count(word) for word in intensity_keywords) * 2
            
            # Geographic relevance (Middle East focus)
            geo_keywords = ['iran', 'israel', 'gaza', 'lebanon', 'syria', 'iraq']
            geo_score = sum(text.count(word) for word in geo_keywords)
            
            total_score = keyword_score + intensity_score + geo_score - sentiment  # Negative sentiment increases score
            
            scores.append({
                'date': article['date'],
                'article_score': max(0, total_score),  # Ensure non-negative
                'sentiment': sentiment,
                'keyword_matches': keyword_score,
                'intensity_score': intensity_score,
                'geo_relevance': geo_score
            })
        
        scores_df = pd.DataFrame(scores)
        
        # Aggregate daily scores
        daily_scores = scores_df.groupby('date').agg({
            'article_score': ['sum', 'mean', 'count'],
            'sentiment': 'mean',
            'keyword_matches': 'sum',
            'intensity_score': 'sum',
            'geo_relevance': 'sum'
        }).reset_index()
        
        # Flatten column names
        daily_scores.columns = ['date', 'total_activity_score', 'avg_activity_score', 'article_count',
                               'avg_sentiment', 'total_keywords', 'total_intensity', 'total_geo_relevance']
        
        return daily_scores
    
    def fetch_stock_data(self, period='3mo'):
        """Fetch stock price data for defense companies"""
        stock_data = {}
        
        for ticker in self.defense_stocks:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                stock_data[ticker] = hist
                logger.info(f"Fetched data for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        return stock_data
    
    def create_features(self, daily_scores, stock_data):
        """Create features combining military activity scores with stock data"""
        features_list = []
        
        for ticker, stock_df in stock_data.items():
            if stock_df.empty:
                continue
                
            # Reset index to get date as column
            stock_df = stock_df.reset_index()
            stock_df['date'] = stock_df['Date'].dt.date
            
            # Merge with military activity scores
            merged = pd.merge(stock_df, daily_scores, on='date', how='left')
            merged = merged.fillna(0)  # Fill missing activity scores with 0
            
            # Calculate stock features
            merged['price_change'] = merged['Close'].pct_change()
            merged['volume_change'] = merged['Volume'].pct_change()
            merged['high_low_spread'] = (merged['High'] - merged['Low']) / merged['Close']
            
            # Rolling averages for activity scores
            merged['activity_ma_7'] = merged['total_activity_score'].rolling(7).mean()
            merged['activity_ma_30'] = merged['total_activity_score'].rolling(30).mean()
            
            # Recent vs historical activity
            merged['activity_vs_ma7'] = merged['total_activity_score'] - merged['activity_ma_7']
            merged['activity_vs_ma30'] = merged['total_activity_score'] - merged['activity_ma_30']
            
            # Lag features (activity impact may be delayed)
            merged['activity_lag1'] = merged['total_activity_score'].shift(1)
            merged['activity_lag2'] = merged['total_activity_score'].shift(2)
            merged['activity_lag3'] = merged['total_activity_score'].shift(3)
            
            # Add ticker identifier
            merged['ticker'] = ticker
            
            features_list.append(merged)
        
        if not features_list:
            return pd.DataFrame()
        
        combined_features = pd.concat(features_list, ignore_index=True)
        return combined_features.dropna()
    
    def build_model(self, features_df, target_days_ahead=1):
        """Build predictive model for defense stock returns"""
        if features_df.empty:
            logger.error("No features available for model building")
            return None, None
        
        # Create target variable (future returns)
        features_df = features_df.sort_values(['ticker', 'date'])
        features_df['target_return'] = features_df.groupby('ticker')['price_change'].shift(-target_days_ahead)
        
        # Select feature columns
        feature_columns = [
            'total_activity_score', 'avg_activity_score', 'article_count',
            'avg_sentiment', 'total_intensity', 'total_geo_relevance',
            'activity_ma_7', 'activity_ma_30', 'activity_vs_ma7', 'activity_vs_ma30',
            'activity_lag1', 'activity_lag2', 'activity_lag3',
            'volume_change', 'high_low_spread'
        ]
        
        # Clean data for modeling
        model_data = features_df[feature_columns + ['target_return']].dropna()
        
        if model_data.empty:
            logger.error("No clean data available for modeling")
            return None, None
        
        X = model_data[feature_columns]
        y = model_data['target_return']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Feature importance (absolute coefficients)
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        logger.info("Model trained successfully")
        logger.info(f"Model R² score: {model.score(X_scaled, y):.4f}")
        
        return model, scaler, feature_importance
    
    def get_account_info(self):
        """Get current account information including buying power and positions"""
        if not self.api:
            logger.error("Alpaca API not initialized")
            return None, None, None
        
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            # Get current positions as a dictionary
            current_positions = {}
            for position in positions:
                current_positions[position.symbol] = {
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'unrealized_pl': float(position.unrealized_pl)
                }
            
            portfolio_value = float(account.portfolio_value)
            buying_power = float(account.buying_power)
            
            logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(f"Buying Power: ${buying_power:,.2f}")
            logger.info(f"Current Positions: {len(current_positions)}")
            
            return portfolio_value, buying_power, current_positions
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return None, None, None
    
    def calculate_position_size(self, predicted_return, current_price, portfolio_value):
        """Calculate position size based on predicted return strength and risk management"""
        if predicted_return <= 0:
            return 0
        
        # Base position size (2.5% of portfolio)
        base_dollar_amount = portfolio_value * self.max_position_size
        
        # Scale position size based on prediction confidence
        # Stronger predictions get larger positions (up to 2x base size)
        confidence_multiplier = min(2.0, 1 + (predicted_return * 10))  # Scale factor
        
        scaled_dollar_amount = base_dollar_amount * confidence_multiplier
        
        # Calculate number of shares
        shares = int(scaled_dollar_amount / current_price)
        
        # Ensure we don't exceed maximum position size
        max_shares = int((portfolio_value * self.max_position_size * 2) / current_price)
        shares = min(shares, max_shares)
        
        return shares
    
    def execute_trades(self, signals_df):
        """Execute trades based on trading signals"""
        if not self.api or signals_df.empty:
            logger.error("Cannot execute trades: API not available or no signals")
            return []
        
        # Get account information
        portfolio_value, buying_power, current_positions = self.get_account_info()
        if portfolio_value is None:
            return []
        
        executed_trades = []
        
        for _, signal in signals_df.iterrows():
            ticker = signal['ticker']
            action = signal['signal']
            predicted_return = signal['predicted_return']
            current_price = signal['current_price']
            
            try:
                if action == 'BUY':
                    # Calculate position size
                    shares_to_buy = self.calculate_position_size(
                        predicted_return, current_price, portfolio_value
                    )
                    
                    if shares_to_buy > 0:
                        # Check if we have enough buying power
                        cost = shares_to_buy * current_price
                        if cost <= buying_power:
                            # Place market buy order
                            order = self.api.submit_order(
                                symbol=ticker,
                                qty=shares_to_buy,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )
                            
                            executed_trades.append({
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': current_price,
                                'cost': cost,
                                'predicted_return': predicted_return,
                                'order_id': order.id,
                                'status': 'submitted'
                            })
                            
                            # Update buying power
                            buying_power -= cost
                            
                            logger.info(f"BUY order placed: {shares_to_buy} shares of {ticker} at ~${current_price:.2f}")
                        else:
                            logger.warning(f"Insufficient buying power for {ticker}: need ${cost:.2f}, have ${buying_power:.2f}")
                
                elif action == 'SELL':
                    # Check if we have positions to sell
                    if ticker in current_positions:
                        current_qty = current_positions[ticker]['qty']
                        
                        if current_qty > 0:
                            # Sell all shares (you could modify this to partial sells)
                            order = self.api.submit_order(
                                symbol=ticker,
                                qty=int(current_qty),
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                            
                            executed_trades.append({
                                'ticker': ticker,
                                'action': 'SELL',
                                'shares': int(current_qty),
                                'price': current_price,
                                'value': current_qty * current_price,
                                'predicted_return': predicted_return,
                                'order_id': order.id,
                                'status': 'submitted'
                            })
                            
                            logger.info(f"SELL order placed: {int(current_qty)} shares of {ticker} at ~${current_price:.2f}")
                            
                            # Remove from current positions
                            del current_positions[ticker]
                        else:
                            logger.info(f"No long position to sell for {ticker}")
                    else:
                        logger.info(f"No position found for {ticker} to sell")
                
                # HOLD - no action needed
                elif action == 'HOLD':
                    logger.info(f"HOLD signal for {ticker} - no action taken")
                    
            except Exception as e:
                logger.error(f"Error executing trade for {ticker}: {e}")
                executed_trades.append({
                    'ticker': ticker,
                    'action': action,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return executed_trades
    
    def check_order_status(self, executed_trades):
        """Check the status of executed orders"""
        if not self.api:
            return executed_trades
        
        updated_trades = []
        
        for trade in executed_trades:
            if 'order_id' in trade and trade['status'] == 'submitted':
                try:
                    order = self.api.get_order(trade['order_id'])
                    trade['status'] = order.status
                    trade['filled_qty'] = order.filled_qty
                    trade['filled_avg_price'] = order.filled_avg_price
                    
                    if order.status == 'filled':
                        logger.info(f"Order filled: {trade['action']} {trade['shares']} {trade['ticker']} at ${order.filled_avg_price}")
                        
                except Exception as e:
                    logger.error(f"Error checking order status: {e}")
                    trade['status_check_error'] = str(e)
            
            updated_trades.append(trade)
        
        return updated_trades
        """Generate trading signals based on model predictions"""
        if model is None or features_df.empty:
            return pd.DataFrame()
        
        # Get latest data for each stock
        latest_data = features_df.groupby('ticker').last().reset_index()
        
        feature_columns = [
            'total_activity_score', 'avg_activity_score', 'article_count',
            'avg_sentiment', 'total_intensity', 'total_geo_relevance',
            'activity_ma_7', 'activity_ma_30', 'activity_vs_ma7', 'activity_vs_ma30',
            'activity_lag1', 'activity_lag2', 'activity_lag3',
            'volume_change', 'high_low_spread'
        ]
        
        X_latest = latest_data[feature_columns].fillna(0)
        X_scaled = scaler.transform(X_latest)
        
        predictions = model.predict(X_scaled)
        
        # Generate signals
        signals = []
        for i, (_, row) in enumerate(latest_data.iterrows()):
            pred_return = predictions[i]
            
            signal = 'HOLD'
            if pred_return > confidence_threshold:
                signal = 'BUY'
            elif pred_return < -confidence_threshold:
                signal = 'SELL'
            
            signals.append({
                'ticker': row['ticker'],
                'date': row['date'],
                'predicted_return': pred_return,
                'current_price': row['Close'],
                'signal': signal,
                'activity_score': row['total_activity_score'],
                'recent_activity': row['activity_vs_ma7']
            })
        
        return pd.DataFrame(signals)
    

    def generate_trading_signals(self, features_df, model, scaler, confidence_threshold=0.01):
        """Generate trading signals based on model predictions"""
        if model is None or features_df.empty:
            return pd.DataFrame()
        
        # Get latest data for each stock
        latest_data = features_df.groupby('ticker').last().reset_index()
        
        feature_columns = [
            'total_activity_score', 'avg_activity_score', 'article_count',
            'avg_sentiment', 'total_intensity', 'total_geo_relevance',
            'activity_ma_7', 'activity_ma_30', 'activity_vs_ma7', 'activity_vs_ma30',
            'activity_lag1', 'activity_lag2', 'activity_lag3',
            'volume_change', 'high_low_spread'
        ]
        
        X_latest = latest_data[feature_columns].fillna(0)
        X_scaled = scaler.transform(X_latest)
        
        predictions = model.predict(X_scaled)
        
        # Generate signals with more nuanced thresholds
        signals = []
        for i, (_, row) in enumerate(latest_data.iterrows()):
            pred_return = predictions[i]
            
            # Adjust thresholds based on prediction strength
            buy_threshold = confidence_threshold
            sell_threshold = -confidence_threshold
            
            signal = 'HOLD'
            signal_strength = abs(pred_return)
            
            if pred_return > buy_threshold:
                signal = 'BUY'
            elif pred_return < sell_threshold:
                signal = 'SELL'
            
            signals.append({
                'ticker': row['ticker'],
                'date': row['date'],
                'predicted_return': pred_return,
                'signal_strength': signal_strength,
                'current_price': row['Close'],
                'signal': signal,
                'activity_score': row['total_activity_score'],
                'recent_activity': row['activity_vs_ma7'],
                'confidence': min(1.0, signal_strength * 20)  # Convert to 0-1 confidence score
            })
        
        signals_df = pd.DataFrame(signals)
        
        # Sort by signal strength (strongest signals first)
        signals_df = signals_df.sort_values('signal_strength', ascending=False)
        
        return signals_df

    def run_analysis_and_trade(self, days_back=60, execute_trades=True):
        """Run complete analysis pipeline and execute trades"""
        logger.info("Starting defense stock geopolitical analysis and trading...")
        
        # 1. Fetch news data
        logger.info("Fetching news data...")
        news_df = self.fetch_news_data(days_back)
        
        # 2. Calculate military activity scores
        logger.info("Calculating military activity scores...")
        daily_scores = self.calculate_military_activity_score(news_df)
        
        # 3. Fetch stock data
        logger.info("Fetching stock data...")
        stock_data = self.fetch_stock_data()
        
        # 4. Create features
        logger.info("Creating features...")
        features_df = self.create_features(daily_scores, stock_data)
        
        # 5. Build model
        logger.info("Building predictive model...")
        model_results = self.build_model(features_df)
        
        executed_trades = []
        
        if len(model_results) == 3:
            model, scaler, feature_importance = model_results
            
            # 6. Generate trading signals
            logger.info("Generating trading signals...")
            signals = self.generate_trading_signals(features_df, model, scaler)
            
            # 7. Execute trades if requested
            if execute_trades and not signals.empty:
                logger.info("Executing trades...")
                executed_trades = self.execute_trades(signals)
                
                # Wait a moment and check order status
                import time
                time.sleep(2)
                executed_trades = self.check_order_status(executed_trades)
            
            return {
                'news_data': news_df,
                'daily_scores': daily_scores,
                'features': features_df,
                'model': model,
                'scaler': scaler,
                'feature_importance': feature_importance,
                'signals': signals,
                'executed_trades': executed_trades
            }
        else:
            return {
                'news_data': news_df,
                'daily_scores': daily_scores,
                'features': features_df,
                'model': None,
                'signals': pd.DataFrame(),
                'executed_trades': []
            }
        """Run complete analysis pipeline"""
        logger.info("Starting defense stock geopolitical analysis...")
        
        # 1. Fetch news data
        logger.info("Fetching news data...")
        news_df = self.fetch_news_data(days_back)
        
        # 2. Calculate military activity scores
        logger.info("Calculating military activity scores...")
        daily_scores = self.calculate_military_activity_score(news_df)
        
        # 3. Fetch stock data
        logger.info("Fetching stock data...")
        stock_data = self.fetch_stock_data()
        
        # 4. Create features
        logger.info("Creating features...")
        features_df = self.create_features(daily_scores, stock_data)
        
        # 5. Build model
        logger.info("Building predictive model...")
        model_results = self.build_model(features_df)
        
        if len(model_results) == 3:
            model, scaler, feature_importance = model_results
            
            # 6. Generate trading signals
            logger.info("Generating trading signals...")
            signals = self.generate_trading_signals(features_df, model, scaler)
            
            return {
                'news_data': news_df,
                'daily_scores': daily_scores,
                'features': features_df,
                'model': model,
                'scaler': scaler,
                'feature_importance': feature_importance,
                'signals': signals
            }
        else:
            return {
                'news_data': news_df,
                'daily_scores': daily_scores,
                'features': features_df,
                'model': None,
                'signals': pd.DataFrame()
            }
        
    def calculate_target_allocations(self, signals_df, total_defense_allocation=0.20):
        """
        Calculate target allocations for each defense stock based on signal strength
        
        Args:
            signals_df: DataFrame with trading signals and predictions
            total_defense_allocation: Maximum % of portfolio to allocate to defense (default 20%)
        
        Returns:
            DataFrame with target allocations for each stock
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        # Only consider stocks with BUY signals for allocation
        buy_signals = signals_df[signals_df['signal'] == 'BUY'].copy()
        
        if buy_signals.empty:
            # No buy signals - return zero allocations
            allocations = []
            for ticker in signals_df['ticker']:
                allocations.append({
                    'ticker': ticker,
                    'target_allocation': 0.0,
                    'allocation_reason': 'No buy signal'
                })
            return pd.DataFrame(allocations)
        
        # Calculate allocation weights based on signal strength and confidence
        buy_signals['allocation_score'] = (
            buy_signals['signal_strength'] * buy_signals['confidence'] * 
            (1 + buy_signals['activity_score'] / 100)  # Boost for high military activity
        )
        
        # Normalize scores to sum to 1
        total_score = buy_signals['allocation_score'].sum()
        buy_signals['normalized_weight'] = buy_signals['allocation_score'] / total_score
        
        # Apply total defense allocation limit
        buy_signals['target_allocation'] = buy_signals['normalized_weight'] * total_defense_allocation
        
        # Apply individual stock limits (max 5% per stock)
        buy_signals['target_allocation'] = buy_signals['target_allocation'].clip(upper=0.05)
        
        # Create complete allocation DataFrame
        allocations = []
        for _, row in signals_df.iterrows():
            if row['ticker'] in buy_signals['ticker'].values:
                target_row = buy_signals[buy_signals['ticker'] == row['ticker']].iloc[0]
                allocations.append({
                    'ticker': row['ticker'],
                    'target_allocation': target_row['target_allocation'],
                    'allocation_score': target_row['allocation_score'],
                    'signal_strength': target_row['signal_strength'],
                    'allocation_reason': f"Buy signal (strength: {target_row['signal_strength']:.3f})"
                })
            else:
                allocations.append({
                    'ticker': row['ticker'],
                    'target_allocation': 0.0,
                    'allocation_score': 0.0,
                    'signal_strength': row['signal_strength'],
                    'allocation_reason': f"No buy signal ({row['signal']})"
                })
        
        return pd.DataFrame(allocations)

    def calculate_rebalancing_trades(self, target_allocations_df, rebalance_threshold=0.015):
        """
        Calculate trades needed to rebalance portfolio to target allocations
        
        Args:
            target_allocations_df: DataFrame with target allocations
            rebalance_threshold: Minimum allocation difference to trigger rebalance (default 1.5%)
        
        Returns:
            List of rebalancing trades to execute
        """
        if not self.api or target_allocations_df.empty:
            logger.error("Cannot calculate rebalancing: API not available or no targets")
            return []
        
        # Get current account info
        portfolio_value, buying_power, current_positions = self.get_account_info()
        if portfolio_value is None:
            return []
        
        rebalancing_trades = []
        
        for _, target in target_allocations_df.iterrows():
            ticker = target['ticker']
            target_allocation = target['target_allocation']
            target_value = portfolio_value * target_allocation
            
            # Current position value
            current_value = 0
            current_shares = 0
            if ticker in current_positions:
                current_value = current_positions[ticker]['market_value']
                current_shares = current_positions[ticker]['qty']
            
            current_allocation = current_value / portfolio_value
            allocation_diff = abs(target_allocation - current_allocation)
            
            # Only rebalance if difference is significant
            if allocation_diff >= rebalance_threshold:
                
                # Get current price
                try:
                    stock_data = yf.Ticker(ticker)
                    current_price = stock_data.history(period='1d')['Close'].iloc[-1]
                except:
                    logger.error(f"Could not get current price for {ticker}")
                    continue
                
                target_shares = int(target_value / current_price)
                shares_diff = target_shares - current_shares
                
                if shares_diff > 0:
                    # Need to buy more shares
                    cost = shares_diff * current_price
                    if cost <= buying_power:
                        rebalancing_trades.append({
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_diff,
                            'current_price': current_price,
                            'cost': cost,
                            'reason': 'Rebalance increase',
                            'current_allocation': current_allocation,
                            'target_allocation': target_allocation,
                            'allocation_diff': allocation_diff
                        })
                        buying_power -= cost
                    else:
                        logger.warning(f"Insufficient buying power for {ticker} rebalance")
                
                elif shares_diff < 0:
                    # Need to sell shares
                    shares_to_sell = abs(shares_diff)
                    if shares_to_sell <= current_shares:
                        rebalancing_trades.append({
                            'ticker': ticker,
                            'action': 'SELL',
                            'shares': shares_to_sell,
                            'current_price': current_price,
                            'value': shares_to_sell * current_price,
                            'reason': 'Rebalance decrease',
                            'current_allocation': current_allocation,
                            'target_allocation': target_allocation,
                            'allocation_diff': allocation_diff
                        })
            else:
                logger.info(f"{ticker}: No rebalancing needed (diff: {allocation_diff:.3f})")
        
        return rebalancing_trades

    def execute_rebalancing_trades(self, rebalancing_trades):
        """
        Execute the rebalancing trades
        
        Args:
            rebalancing_trades: List of trades from calculate_rebalancing_trades
        
        Returns:
            List of executed trades with status
        """
        if not self.api or not rebalancing_trades:
            return []
        
        executed_rebalances = []
        
        logger.info(f"Executing {len(rebalancing_trades)} rebalancing trades...")
        
        for trade in rebalancing_trades:
            ticker = trade['ticker']
            action = trade['action']
            shares = trade['shares']
            
            try:
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=shares,
                    side=action.lower(),
                    type='market',
                    time_in_force='day'
                )
                
                trade['order_id'] = order.id
                trade['status'] = 'submitted'
                trade['timestamp'] = datetime.now()
                
                executed_rebalances.append(trade)
                
                logger.info(f"REBALANCE {action}: {shares} shares of {ticker} "
                        f"(Target: {trade['target_allocation']:.2%}, "
                        f"Current: {trade['current_allocation']:.2%})")
                
            except Exception as e:
                logger.error(f"Error executing rebalance trade for {ticker}: {e}")
                trade['error'] = str(e)
                trade['status'] = 'failed'
                executed_rebalances.append(trade)
        
        return executed_rebalances

    def full_rebalance_portfolio(self, signals_df, total_defense_allocation=0.20, 
                            rebalance_threshold=0.015, execute_trades=True):
        """
        Complete portfolio rebalancing workflow
        
        Args:
            signals_df: Current trading signals
            total_defense_allocation: Max % of portfolio for defense stocks (default 20%)
            rebalance_threshold: Min difference to trigger rebalance (default 1.5%)
            execute_trades: Whether to actually execute trades
        
        Returns:
            Dictionary with rebalancing analysis and executed trades
        """
        logger.info("Starting portfolio rebalancing analysis...")
        
        # 1. Calculate target allocations based on current signals
        target_allocations = self.calculate_target_allocations(
            signals_df, total_defense_allocation
        )
        
        if target_allocations.empty:
            logger.warning("No target allocations calculated")
            return {'target_allocations': pd.DataFrame(), 'executed_trades': []}
        
        # 2. Calculate required rebalancing trades
        rebalancing_trades = self.calculate_rebalancing_trades(
            target_allocations, rebalance_threshold
        )
        
        # 3. Execute trades if requested
        executed_trades = []
        if execute_trades and rebalancing_trades:
            executed_trades = self.execute_rebalancing_trades(rebalancing_trades)
            
            # Check order status
            import time
            time.sleep(2)
            for trade in executed_trades:
                if 'order_id' in trade:
                    try:
                        order = self.api.get_order(trade['order_id'])
                        trade['final_status'] = order.status
                        trade['filled_qty'] = order.filled_qty
                        trade['filled_avg_price'] = order.filled_avg_price
                    except Exception as e:
                        trade['status_error'] = str(e)
        
        # 4. Create summary
        total_target_allocation = target_allocations['target_allocation'].sum()
        rebalance_summary = {
            'total_defense_allocation': total_target_allocation,
            'num_positions_targeted': len(target_allocations[target_allocations['target_allocation'] > 0]),
            'num_rebalancing_trades': len(rebalancing_trades),
            'num_executed_trades': len(executed_trades),
            'rebalance_threshold_used': rebalance_threshold
        }
        
        logger.info(f"Rebalancing complete: {rebalance_summary['num_executed_trades']} trades executed")
        
        return {
            'target_allocations': target_allocations,
            'rebalancing_trades': rebalancing_trades,
            'executed_trades': executed_trades,
            'summary': rebalance_summary
        }

    def run_analysis_trade_and_rebalance(self, days_back=60, execute_initial_trades=True, 
                                    execute_rebalancing=True, total_defense_allocation=0.20):
        """
        Complete workflow: analysis, initial trades, and rebalancing
        
        Args:
            days_back: Days of historical data to analyze
            execute_initial_trades: Whether to execute new position trades
            execute_rebalancing: Whether to execute rebalancing trades
            total_defense_allocation: Max % allocation to defense sector
        
        Returns:
            Complete results including initial trades and rebalancing
        """
        logger.info("Starting complete analysis, trading, and rebalancing workflow...")
        
        # 1. Run initial analysis and trading
        initial_results = self.run_analysis_and_trade(days_back, execute_initial_trades)
        
        if initial_results['signals'].empty:
            logger.warning("No signals generated - skipping rebalancing")
            return {**initial_results, 'rebalancing_results': {}}
        
        # 2. Run rebalancing based on signals
        rebalancing_results = self.full_rebalance_portfolio(
            initial_results['signals'],
            total_defense_allocation,
            execute_trades=execute_rebalancing
        )
        
        # 3. Combine results
        combined_results = {
            **initial_results,
            'rebalancing_results': rebalancing_results
        }
        
        return combined_results

        
# Example usage
if __name__ == "__main__":
    model = DefenseStockModel()
    
    # Check if today is Wednesday (weekday 2, where Monday=0)
    today = datetime.now()
    is_wednesday = today.weekday() == 2
    
    print(f"Today is {today.strftime('%A, %B %d, %Y')}")
    print(f"Rebalancing {'ENABLED' if is_wednesday else 'DISABLED'} (only runs on Wednesdays)")
    
    # Run complete analysis, trading, and rebalancing workflow
    results = model.run_analysis_trade_and_rebalance(
        days_back=90, 
        execute_initial_trades=True,
        execute_rebalancing=is_wednesday,  # Only rebalance on Wednesdays
        total_defense_allocation=0.20  # 20% max allocation to defense sector
    )
    
    # Display results
    print("\n" + "="*60)
    print("DEFENSE STOCK GEOPOLITICAL TRADING RESULTS")
    print("="*60)
    
    # Account summary
    if model.api:
        portfolio_value, buying_power, positions = model.get_account_info()
        if portfolio_value:
            print(f"\n📊 ACCOUNT SUMMARY")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Buying Power: ${buying_power:,.2f}")
            print(f"Active Positions: {len(positions)}")
            
            # Calculate current defense allocation
            defense_value = sum(pos['market_value'] for ticker, pos in positions.items() 
                              if ticker in model.defense_stocks)
            defense_allocation = defense_value / portfolio_value if portfolio_value > 0 else 0
            print(f"Current Defense Allocation: {defense_allocation:.1%}")
    
    # Trading signals
    if not results['signals'].empty:
        print(f"\n📈 CURRENT TRADING SIGNALS")
        signals_display = results['signals'][['ticker', 'signal', 'predicted_return',
                                            'signal_strength', 'confidence', 'current_price',
                                            'activity_score']].round(4)
        print(signals_display.to_string(index=False))
        
        # Count signals by type
        signal_counts = results['signals']['signal'].value_counts()
        print(f"\nSignal Summary: {dict(signal_counts)}")
    
    # Initial executed trades
    if results['executed_trades']:
        print(f"\n💼 INITIAL TRADES EXECUTED")
        for trade in results['executed_trades']:
            if trade.get('status') != 'failed':
                action = trade['action']
                ticker = trade['ticker']
                shares = trade.get('shares', 0)
                price = trade.get('price', 0)
                status = trade.get('status', 'unknown')
                
                print(f"{action} {shares} shares of {ticker} at ${price:.2f} - Status: {status}")
            else:
                print(f"FAILED: {trade['ticker']} - {trade.get('error', 'Unknown error')}")
    
    # Rebalancing results
    if 'rebalancing_results' in results and results['rebalancing_results']:
        rebal = results['rebalancing_results']
        
        print(f"\n⚖️  PORTFOLIO REBALANCING (Wednesday Only)")
        
        # Rebalancing summary
        if 'summary' in rebal:
            summary = rebal['summary']
            print(f"Target Defense Allocation: {summary['total_defense_allocation']:.1%}")
            print(f"Positions Targeted: {summary['num_positions_targeted']}")
            print(f"Rebalancing Trades: {summary['num_rebalancing_trades']}")
            print(f"Trades Executed: {summary['num_executed_trades']}")
        
        # Target allocations
        if not rebal.get('target_allocations', pd.DataFrame()).empty:
            print(f"\n🎯 TARGET ALLOCATIONS")
            targets = rebal['target_allocations']
            targets_display = targets[targets['target_allocation'] > 0][
                ['ticker', 'target_allocation', 'signal_strength', 'allocation_reason']
            ].round(4)
            if not targets_display.empty:
                print(targets_display.to_string(index=False))
            else:
                print("No target allocations (no buy signals)")
        
        # Executed rebalancing trades
        if rebal.get('executed_trades'):
            print(f"\n🔄 REBALANCING TRADES EXECUTED")
            for trade in rebal['executed_trades']:
                if trade.get('status') != 'failed':
                    action = trade['action']
                    ticker = trade['ticker']
                    shares = trade.get('shares', 0)
                    current_alloc = trade.get('current_allocation', 0)
                    target_alloc = trade.get('target_allocation', 0)
                    reason = trade.get('reason', 'Unknown')
                    
                    print(f"{action} {shares} shares of {ticker} - {reason}")
                    print(f"  Current: {current_alloc:.1%} → Target: {target_alloc:.1%}")
                else:
                    print(f"REBALANCE FAILED: {trade['ticker']} - {trade.get('error', 'Unknown error')}")
        else:
            print(f"\n🔄 REBALANCING: No rebalancing trades needed")
    elif not is_wednesday:
        print(f"\n⚖️  PORTFOLIO REBALANCING")
        print(f"🗓️  Rebalancing skipped - only runs on Wednesdays")
        print(f"📅 Next rebalancing opportunity: {(today + timedelta(days=(2-today.weekday()) % 7 + 7 if today.weekday() >= 2 else (2-today.weekday()))).strftime('%A, %B %d, %Y')}")
        
        # Still show what the targets WOULD be
        if not results['signals'].empty:
            target_allocations = model.calculate_target_allocations(results['signals'], 0.20)
            if not target_allocations.empty:
                print(f"\n🎯 CURRENT TARGET ALLOCATIONS (for reference)")
                targets_display = target_allocations[target_allocations['target_allocation'] > 0][
                    ['ticker', 'target_allocation', 'signal_strength', 'allocation_reason']
                ].round(4)
                if not targets_display.empty:
                    print(targets_display.to_string(index=False))
    else:
        print(f"\n⚖️  PORTFOLIO REBALANCING")
        print(f"🔄 No rebalancing results (no signals generated)")
    
    # Feature importance
    if results.get('feature_importance') is not None:
        print(f"\n🎯 TOP PREDICTIVE FEATURES")
        top_features = results['feature_importance'].head(8)
        for _, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Recent military activity summary
    if not results.get('daily_scores', pd.DataFrame()).empty:
        print(f"\n🚀 RECENT MILITARY ACTIVITY")
        recent_activity = results['daily_scores'].tail(7)
        avg_activity = recent_activity['total_activity_score'].mean()
        max_activity_idx = recent_activity['total_activity_score'].idxmax()
        max_activity_date = recent_activity.loc[max_activity_idx, 'date']
        max_activity_score = recent_activity.loc[max_activity_idx, 'total_activity_score']
        
        print(f"7-day average activity score: {avg_activity:.2f}")
        print(f"Highest activity day: {max_activity_date} (score: {max_activity_score:.1f})")
        print(f"Total articles analyzed: {recent_activity['article_count'].sum()}")
    
    print(f"\n⚠️  RISK MANAGEMENT")
    print(f"Max position size per trade: {model.max_position_size*100:.1f}% of portfolio")
    print(f"Max defense sector allocation: 20.0% of portfolio")
    print(f"Trading {len(model.defense_stocks)} defense stocks")
    print(f"Paper trading account - No real money at risk")
    print(f"Rebalancing threshold: 1.5% allocation difference")
    print(f"Rebalancing schedule: Wednesdays only")
    
    # Overall summary
    total_trades = len(results.get('executed_trades', []))
    total_rebalances = len(results.get('rebalancing_results', {}).get('executed_trades', []))
    
    if total_trades == 0 and total_rebalances == 0:
        print(f"\n📝 NOTE: No trades executed. Possible reasons:")
        print(f"- No strong enough signals generated")
        print(f"- Portfolio already optimally balanced")
        print(f"- Insufficient buying power")
        print(f"- API connection issues")
        print(f"- All signals were HOLD")
    else:
        print(f"\n✅ EXECUTION SUMMARY")
        print(f"Initial trades executed: {total_trades}")
        print(f"Rebalancing trades executed: {total_rebalances}")
        print(f"Total portfolio actions: {total_trades + total_rebalances}")
    
    print(f"\n" + "="*60)