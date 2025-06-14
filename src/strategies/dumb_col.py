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
import alpaca_trade_api as tradeapi

# Path and env setup
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeopoliticalDumbMoneyStrategy:
    """
    Collection of 'dumb' geopolitical trading strategies based on weird correlations
    """
    
    def __init__(self):
        self.target_stocks = os.getenv('TARGET_STOCKS', '').split(',')
        self.data_dir = os.getenv('DATA_DIR', './data/')
        
        # Initialize Alpaca API
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            logger.error("Alpaca API credentials not found in environment variables!")
            self.alpaca = None
        else:
            try:
                self.alpaca = tradeapi.REST(
                    self.api_key,
                    self.secret_key,
                    self.base_url,
                    api_version='v2'
                )
                logger.info("Alpaca API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca API: {e}")
                self.alpaca = None
    
    def get_gdelt_data(self, query, timespan='1d'):
        """
        Fetch GDELT data - completely free, no limits
        """
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': 250,
            'timespan': timespan,
            'format': 'json'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            
            # Check if response is empty or not valid JSON
            if response.status_code != 200:
                logger.warning(f"GDELT returned status {response.status_code} for query: {query}")
                return {'articles': []}
            
            if not response.text.strip():
                logger.warning(f"Empty response from GDELT for query: {query}")
                return {'articles': []}
            
            try:
                data = response.json()
                # GDELT sometimes returns malformed JSON, ensure we have articles key
                if 'articles' not in data:
                    data['articles'] = []
                return data
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from GDELT for query: {query}")
                return {'articles': []}
                
        except requests.exceptions.Timeout:
            logger.warning(f"GDELT timeout for query: {query}")
            return {'articles': []}
        except Exception as e:
            logger.error(f"GDELT API error for query '{query}': {e}")
            return {'articles': []}
    
    def get_account_info(self):
        """Get current account information and buying power"""
        if not self.alpaca:
            return None
        
        try:
            account = self.alpaca.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'equity': float(account.equity)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_current_positions(self):
        """Get current positions"""
        if not self.alpaca:
            return {}
        
        try:
            positions = self.alpaca.list_positions()
            return {pos.symbol: {
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl)
            } for pos in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def calculate_position_size(self, symbol, signal_strength):
        """
        Calculate position size based on 5% max risk and signal strength
        """
        account_info = self.get_account_info()
        if not account_info:
            return 0
        
        # Base position size: 5% of portfolio value
        max_position_value = account_info['portfolio_value'] * 0.05
        
        # Scale by signal strength (0.1 to 1.0)
        adjusted_position_value = max_position_value * signal_strength
        
        # Get current stock price
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            # Calculate shares to buy
            shares = int(adjusted_position_value / current_price)
            
            logger.info(f"Position sizing for {symbol}:")
            logger.info(f"  Portfolio value: ${account_info['portfolio_value']:,.2f}")
            logger.info(f"  Max position (5%): ${max_position_value:,.2f}")
            logger.info(f"  Signal strength: {signal_strength}")
            logger.info(f"  Adjusted position: ${adjusted_position_value:,.2f}")
            logger.info(f"  Current price: ${current_price:.2f}")
            logger.info(f"  Shares to buy: {shares}")
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def check_buying_power(self, symbol, shares):
        """
        Check if we have enough buying power for the trade
        """
        if not self.alpaca or shares <= 0:
            return False
        
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            # Calculate required buying power (add small buffer for price movement)
            required_cash = shares * current_price * 1.02  # 2% buffer
            
            account_info = self.get_account_info()
            available_cash = account_info['buying_power']
            
            logger.info(f"Buying power check for {symbol}:")
            logger.info(f"  Required: ${required_cash:,.2f}")
            logger.info(f"  Available: ${available_cash:,.2f}")
            
            return required_cash <= available_cash
            
        except Exception as e:
            logger.error(f"Error checking buying power for {symbol}: {e}")
            return False
    
    def execute_trade(self, symbol, action, shares, reason):
        """
        Execute trade through Alpaca API
        """
        if not self.alpaca:
            logger.error("Alpaca API not initialized")
            return False
        
        if shares <= 0:
            logger.warning(f"Invalid share amount: {shares}")
            return False
        
        try:
            # Double-check buying power before executing
            if action.upper() in ['BUY', 'STRONG_BUY']:
                if not self.check_buying_power(symbol, shares):
                    logger.warning(f"Insufficient buying power for {symbol} - skipping trade")
                    return False
                
                side = 'buy'
            elif action.upper() in ['SELL', 'STRONG_SELL']:
                side = 'sell'
                # Check if we have shares to sell
                positions = self.get_current_positions()
                if symbol not in positions or positions[symbol]['qty'] < shares:
                    logger.warning(f"Insufficient shares to sell for {symbol} - skipping trade")
                    return False
            else:
                logger.info(f"HOLD signal for {symbol} - no trade executed")
                return True
            
            # Submit the order
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                type='market',
                time_in_force='day',
                client_order_id=f"geopolitical_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            logger.info(f"Order submitted successfully:")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Side: {side.upper()}")
            logger.info(f"  Quantity: {shares}")
            logger.info(f"  Reason: {reason}")
            logger.info(f"  Order ID: {order.id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def execute_portfolio_strategy(self, strategy_result):
        """
        Execute trades across portfolio based on strategy results
        """
        if not self.alpaca:
            logger.error("Cannot execute trades - Alpaca API not initialized")
            return
        
        signal = strategy_result['final_signal']
        total_strength = strategy_result['total_strength']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"EXECUTING PORTFOLIO STRATEGY")
        logger.info(f"Signal: {signal}")
        logger.info(f"Total Strength: {total_strength:.2f}")
        logger.info(f"{'='*60}")
        
        # Get account info
        account_info = self.get_account_info()
        if not account_info:
            return
        
        # Filter target stocks to focus on different sectors based on strategy
        if signal in ['BUY', 'STRONG_BUY']:
            # Focus on tech stocks for buy signals
            focus_stocks = [s for s in self.target_stocks if s in [
                'NVDA', 'AMD', 'INTC', 'MSFT', 'GOOGL', 'META', 'TSLA'
            ]][:5]  # Limit to 5 stocks to manage risk
        elif signal in ['SELL', 'STRONG_SELL']:
            # Check current positions for sell signals
            positions = self.get_current_positions()
            focus_stocks = list(positions.keys())[:5]
        else:
            logger.info("HOLD signal - no trades executed")
            return
        
        trades_executed = 0
        total_trade_value = 0
        
        for symbol in focus_stocks:
            try:
                if signal in ['BUY', 'STRONG_BUY']:
                    # Calculate position size
                    shares = self.calculate_position_size(symbol, min(total_strength/len(focus_stocks), 1.0))
                    
                    if shares > 0:
                        # Get estimated trade value
                        ticker = yf.Ticker(symbol)
                        current_price = ticker.history(period='1d')['Close'].iloc[-1]
                        trade_value = shares * current_price
                        
                        # Check if this trade would exceed our 5% total limit
                        if total_trade_value + trade_value > account_info['portfolio_value'] * 0.05:
                            logger.warning(f"Skipping {symbol} - would exceed 5% portfolio limit")
                            continue
                        
                        success = self.execute_trade(
                            symbol, 
                            signal, 
                            shares, 
                            f"Geopolitical strategy: {strategy_result['individual_results']}"
                        )
                        
                        if success:
                            trades_executed += 1
                            total_trade_value += trade_value
                
                elif signal in ['SELL', 'STRONG_SELL']:
                    positions = self.get_current_positions()
                    if symbol in positions:
                        # Sell a portion based on signal strength
                        current_shares = int(positions[symbol]['qty'])
                        sell_shares = max(1, int(current_shares * min(total_strength/2, 0.5)))  # Sell up to 50%
                        
                        success = self.execute_trade(
                            symbol,
                            signal,
                            sell_shares,
                            f"Geopolitical strategy: {strategy_result['individual_results']}"
                        )
                        
                        if success:
                            trades_executed += 1
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRADING SESSION COMPLETE")
        logger.info(f"Trades executed: {trades_executed}")
        logger.info(f"Total trade value: ${total_trade_value:,.2f}")
        logger.info(f"Percentage of portfolio: {(total_trade_value/account_info['portfolio_value'])*100:.2f}%")
        logger.info(f"{'='*60}")
    
    def strategy_1_twitter_mentions_inverse(self):
        """
        DUMB PREMISE: When geopolitical tension is high (lots of news),
        tech stocks actually go UP because people think tech = safe haven
        """
        logger.info("Running Strategy 1: Geopolitical Tension Inverse Correlation")
        
        # Get global tension indicators from GDELT
        tension_keywords = [
            "military conflict", "trade war", "sanctions", 
            "diplomatic crisis", "border dispute", "cyber attack"
        ]
        
        tension_scores = []
        for keyword in tension_keywords:
            data = self.get_gdelt_data(keyword, timespan='7d')
            if data and 'articles' in data:
                score = len(data['articles'])
                tension_scores.append(score)
                logger.info(f"Keyword '{keyword}': {score} articles")
            else:
                tension_scores.append(0)
        
        total_tension = sum(tension_scores)
        logger.info(f"Current geopolitical tension score: {total_tension}")
        
        # DUMB LOGIC: High tension = Buy tech (inverse correlation premise)
        if total_tension > 100:  # Arbitrary threshold
            return {"signal": "BUY", "reason": "High geopolitical tension detected", "strength": min(total_tension/200, 1.0)}
        elif total_tension < 20:
            return {"signal": "SELL", "reason": "Low tension, tech might be overvalued", "strength": 0.3}
        else:
            return {"signal": "HOLD", "reason": "Neutral tension levels", "strength": 0.1}
    
    def strategy_2_weird_country_correlation(self):
        """
        DUMB PREMISE: News mentions of random countries correlate with 
        semiconductor performance (supply chain fears)
        """
        logger.info("Running Strategy 2: Random Country Semiconductor Correlation")
        
        # Pick some random countries that might affect supply chains
        weird_countries = ["Kazakhstan", "Mongolia", "Belarus", "Myanmar", "Madagascar"]
        
        country_mentions = {}
        for country in weird_countries:
            # Simplify the query to avoid complex parsing issues
            data = self.get_gdelt_data(f'{country}', timespan='3d')
            mentions = len(data['articles']) if data and 'articles' in data else 0
            country_mentions[country] = mentions
            logger.info(f"Country '{country}': {mentions} mentions")
            
        logger.info(f"Country mention scores: {country_mentions}")
        
        # DUMB LOGIC: If weird countries are in the news, semiconductor supply chains are at risk
        total_weird_mentions = sum(country_mentions.values())
        
        if total_weird_mentions > 15:
            return {"signal": "SELL", "reason": f"Weird country activity: {total_weird_mentions} mentions", "strength": 0.6}
        else:
            return {"signal": "BUY", "reason": "No weird country disruptions", "strength": 0.4}
    
    def strategy_3_weather_energy_correlation(self):
        """
        DUMB PREMISE: Extreme weather news correlates with energy stock performance
        """
        logger.info("Running Strategy 3: Weather-Energy Correlation")
        
        weather_events = ["hurricane", "tornado", "heatwave", "cold snap", "drought", "flooding"]
        
        weather_scores = []
        for event in weather_events:
            # Simplify query
            data = self.get_gdelt_data(event, timespan='5d')
            score = len(data['articles']) if data and 'articles' in data else 0
            weather_scores.append(score)
            logger.info(f"Weather event '{event}': {score} articles")
        
        total_weather = sum(weather_scores)
        logger.info(f"Weather-energy correlation score: {total_weather}")
        
        # DUMB LOGIC: More weather news = energy volatility = opportunity
        if total_weather > 30:
            return {"signal": "BUY", "reason": f"High weather-energy correlation: {total_weather}", "strength": 0.7}
        else:
            return {"signal": "HOLD", "reason": "Low weather activity", "strength": 0.2}
    
    def strategy_4_celebrity_crypto_tech_correlation(self):
        """
        DUMB PREMISE: When celebrities are mentioned with crypto/tech terms,
        it signals retail FOMO and potential reversal
        """
        logger.info("Running Strategy 4: Celebrity-Tech Reversal Signal")
        
        # Simplify celebrity query - complex AND/OR queries sometimes fail
        celebrity_tech_query = 'Musk bitcoin'
        
        data = self.get_gdelt_data(celebrity_tech_query, timespan='2d')
        celebrity_mentions = len(data['articles']) if data and 'articles' in data else 0
        
        logger.info(f"Celebrity-tech mentions: {celebrity_mentions}")
        
        # DUMB LOGIC: Too much celebrity tech talk = top signal
        if celebrity_mentions > 25:
            return {"signal": "SELL", "reason": f"Celebrity tech FOMO detected: {celebrity_mentions} mentions", "strength": 0.8}
        elif celebrity_mentions < 5:
            return {"signal": "BUY", "reason": "Low celebrity noise, accumulation phase", "strength": 0.5}
        else:
            return {"signal": "HOLD", "reason": "Normal celebrity tech chatter", "strength": 0.1}
    
    def strategy_5_space_defense_correlation(self):
        """
        DUMB PREMISE: Space news correlates with defense/aerospace stocks
        because people think space = military
        """
        logger.info("Running Strategy 5: Space-Defense Correlation")
        
        # Simplify space query
        space_query = 'SpaceX'
        
        data = self.get_gdelt_data(space_query, timespan='3d')
        space_defense_mentions = len(data['articles']) if data and 'articles' in data else 0
        
        logger.info(f"Space-defense correlation: {space_defense_mentions}")
        
        # DUMB LOGIC: Space news = defense spending = buy defense-related tech
        if space_defense_mentions > 20:
            return {"signal": "BUY", "reason": f"Space-defense correlation high: {space_defense_mentions}", "strength": 0.6}
        else:
            return {"signal": "HOLD", "reason": "Low space-defense activity", "strength": 0.2}
    
    def run_all_strategies(self):
        """
        Run all dumb strategies and aggregate signals
        """
        strategies = [
            self.strategy_1_twitter_mentions_inverse,
            self.strategy_2_weird_country_correlation,
            self.strategy_3_weather_energy_correlation,
            self.strategy_4_celebrity_crypto_tech_correlation,
            self.strategy_5_space_defense_correlation
        ]
        
        results = []
        for strategy in strategies:
            try:
                result = strategy()
                results.append(result)
                logger.info(f"Strategy result: {result}")
            except Exception as e:
                logger.error(f"Strategy failed: {e}")
                continue
        
        # Aggregate signals with weights
        buy_signals = sum(1 for r in results if r['signal'] == 'BUY')
        sell_signals = sum(1 for r in results if r['signal'] == 'SELL')
        
        total_strength = sum(r['strength'] for r in results if r['signal'] != 'HOLD')
        
        logger.info(f"Signal summary - BUY: {buy_signals}, SELL: {sell_signals}, Total strength: {total_strength}")
        
        # DUMB AGGREGATION LOGIC
        if buy_signals > sell_signals and total_strength > 1.5:
            final_signal = "STRONG_BUY"
        elif buy_signals > sell_signals:
            final_signal = "BUY"
        elif sell_signals > buy_signals and total_strength > 1.5:
            final_signal = "STRONG_SELL"
        elif sell_signals > buy_signals:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
        
        return {
            "final_signal": final_signal,
            "individual_results": results,
            "total_strength": total_strength,
            "timestamp": datetime.now().isoformat()
        }

# Usage example
if __name__ == "__main__":
    strategy = GeopoliticalDumbMoneyStrategy()
    
    # Check if Alpaca is connected
    if strategy.alpaca:
        account_info = strategy.get_account_info()
        if account_info:
            print(f"\n🏦 ACCOUNT STATUS:")
            print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
            print(f"Buying Power: ${account_info['buying_power']:,.2f}")
            print(f"Cash: ${account_info['cash']:,.2f}")
            
            positions = strategy.get_current_positions()
            if positions:
                print(f"\n📊 CURRENT POSITIONS:")
                for symbol, pos in positions.items():
                    print(f"  {symbol}: {pos['qty']} shares, ${pos['market_value']:,.2f} value")
    
    # Run all strategies
    final_result = strategy.run_all_strategies()
    
    print("\n" + "="*50)
    print("GEOPOLITICAL DUMB MONEY STRATEGY RESULTS")
    print("="*50)
    print(f"Final Signal: {final_result['final_signal']}")
    print(f"Total Strength: {final_result['total_strength']:.2f}")
    print(f"Timestamp: {final_result['timestamp']}")
    
    print("\nIndividual Strategy Results:")
    for i, result in enumerate(final_result['individual_results'], 1):
        print(f"  Strategy {i}: {result['signal']} - {result['reason']} (Strength: {result['strength']})")
    
    # Execute trades if we have actionable signals
    if final_result['final_signal'] != 'HOLD' and strategy.alpaca:
        print(f"\n🚀 EXECUTING TRADES...")
        strategy.execute_portfolio_strategy(final_result)
    else:
        print(f"\n💤 No trades executed (HOLD signal or Alpaca not connected)")