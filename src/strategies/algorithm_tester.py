import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
import yfinance as yf
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class EnhancedAlgorithmTester:
    def __init__(self, original_predict_func=None, original_execute_func=None):
        """
        Initialize the enhanced tester with weight optimization capabilities
        """
        self.original_predict = original_predict_func
        self.original_execute = original_execute_func
        self.results = []
        self.data_cache = {}  # Cache for stock data
        
        # Default weight parameters
        self.default_weights = {
            'price_weight': 0.4,
            'trend_weight': 0.3,
            'volume_weight': 0.2,
            'volume_price_weight': 0.1
        }
        
        # Default other parameters
        self.default_params = {
            'param_1': 0.1,
            'param_2': 0.5,
            'buy_threshold': 0.6,
            'sell_threshold': 0.4
        }
        
    def get_stock_data(self, symbol, years=3):
        """Enhanced version of get_stock_data with caching and improved indicators"""
        cache_key = f"{symbol}_{years}"
        
        # Return cached data if available
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            print(f"Fetching data for {symbol}...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365 + 30)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if data.empty:
                raise ValueError(f"No data found for symbol '{symbol}'")
            
            data.columns = data.columns.str.strip()
            
            # Calculate all technical indicators
            # Moving averages
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            
            # Volume indicators
            data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
            
            # Price momentum indicators
            data['Price_vs_MA20'] = (data['Close'] - data['MA_20']) / data['MA_20'] * 100
            data['Price_Change_5d'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5) * 100
            data['Price_Change_1d'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100
            
            # Price direction indicators
            data['MA5_vs_MA20'] = (data['MA_5'] - data['MA_20']) / data['MA_20'] * 100
            data['Price_Trend'] = np.where(data['MA_5'] > data['MA_20'], 1, -1)
            
            # Volume-Price relationship
            data['Volume_Price_Signal'] = np.where(
                (data['Price_Change_1d'] > 0) & (data['Volume_Ratio'] > 1), 1,
                np.where(
                    (data['Price_Change_1d'] < 0) & (data['Volume_Ratio'] > 1), -1,
                    0
                )
            )
            
            data = data.sort_index()
            
            # Cache the data
            self.data_cache[cache_key] = data
            print(f"Cached {len(data)} days of data for {symbol}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def preload_data(self, symbols, years=3):
        """Preload all stock data to avoid rate limiting during optimization"""
        print(f"Preloading data for {len(symbols)} symbols...")
        print("This will take a few minutes but will speed up optimization significantly.")
        print("-" * 60)
        
        for i, symbol in enumerate(symbols):
            try:
                print(f"[{i+1}/{len(symbols)}] Loading {symbol}...")
                self.get_stock_data(symbol, years)
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
        
        print(f"\nData preloading complete! Cached data for {len(self.data_cache)} datasets.")
        return self
    
    def improved_normalize(self, value, center=0, scale=10):
        """Improved normalization using tanh for smoother gradients"""
        normalized = (np.tanh((value - center) / scale) + 1) / 2
        return normalized
    
    def predict_with_all_params(self, symbol, weights, other_params, data=None):
        """
        Enhanced predict function that accepts both weights and other parameters
        """
        if data is None:
            df = self.get_stock_data(symbol)
        else:
            df = data
        
        # Handle missing data with neutral values
        price_vs_ma = df['Price_vs_MA20'].iloc[-1] if not pd.isna(df['Price_vs_MA20'].iloc[-1]) else 0.0
        volume_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
        trend_signal = df['MA5_vs_MA20'].iloc[-1] if not pd.isna(df['MA5_vs_MA20'].iloc[-1]) else 0.0
        volume_price_signal = df['Volume_Price_Signal'].iloc[-1] if not pd.isna(df['Volume_Price_Signal'].iloc[-1]) else 0.0
        
        # Improved normalization with better scaling
        normalized_price = self.improved_normalize(price_vs_ma, center=0, scale=5)
        normalized_volume = self.improved_normalize(volume_ratio, center=1, scale=0.5)
        normalized_trend = self.improved_normalize(trend_signal, center=0, scale=2)
        normalized_volume_price = (volume_price_signal + 1) / 2
        
        # Extract weights
        w_price = weights['price_weight']
        w_trend = weights['trend_weight']
        w_volume = weights['volume_weight']
        w_volume_price = weights['volume_price_weight']
        
        # Normalize weights to sum to 1
        total_weight = w_price + w_trend + w_volume + w_volume_price
        if total_weight > 0:
            w_price /= total_weight
            w_trend /= total_weight
            w_volume /= total_weight
            w_volume_price /= total_weight
        else:
            w_price = w_trend = w_volume = w_volume_price = 0.25
        
        # Combine signals with normalized weights
        prediction = (
            w_price * normalized_price +
            w_trend * normalized_trend +
            w_volume * normalized_volume +
            w_volume_price * normalized_volume_price
        )
        
        # Apply final sigmoid with learned parameters
        final_prediction = 1 / (1 + np.exp(-other_params['param_1'] * (prediction - 0.5)))
        
        return final_prediction
    
    def execute_with_all_params(self, symbol, weights, other_params, data=None):
        """Enhanced execute function that accepts all parameters"""
        score = self.predict_with_all_params(symbol, weights, other_params, data)
        
        if score >= other_params['buy_threshold']:
            return 'BUY'
        elif score <= other_params['sell_threshold']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def backtest_all_parameters(self, symbols, weight_ranges, param_ranges, lookback_days=30):
        """
        Comprehensive backtesting with both weights and other parameters
        """
        results = []
        
        # Create parameter grid
        all_params = {**weight_ranges, **param_ranges}
        param_grid = list(ParameterGrid(all_params))
        
        print(f"Backtesting {len(param_grid)} parameter combinations...")
        print("Using cached data - this should be reasonably fast!")
        
        for i, params in enumerate(param_grid):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(param_grid)} ({i/len(param_grid)*100:.1f}%)")
            
            # Split parameters
            weights = {k: v for k, v in params.items() if 'weight' in k}
            other_params = {k: v for k, v in params.items() if 'weight' not in k}
            
            predictions = []
            actuals = []
            
            for symbol in symbols:
                try:
                    df = self.get_stock_data(symbol, years=2)
                    
                    # Test on multiple historical points
                    for j in range(lookback_days, len(df)-5, 5):
                        # Use historical data slice for prediction
                        hist_data = df.iloc[:j+1].copy()
                        
                        # Calculate prediction using enhanced function
                        pred_score = self.predict_with_all_params(symbol, weights, other_params, hist_data)
                        
                        # Convert to action
                        if pred_score >= other_params['buy_threshold']:
                            pred_action = 'BUY'
                        elif pred_score <= other_params['sell_threshold']:
                            pred_action = 'SELL'
                        else:
                            pred_action = 'HOLD'
                        
                        # Calculate actual performance (5 days later)
                        if j + 5 < len(df):
                            current_price = df['Close'].iloc[j]
                            future_price = df['Close'].iloc[j + 5]
                            price_change_pct = (future_price - current_price) / current_price * 100
                            
                            if price_change_pct > 1:
                                actual_action = 'BUY'
                            elif price_change_pct < -1:
                                actual_action = 'SELL'
                            else:
                                actual_action = 'HOLD'
                            
                            predictions.append(pred_action)
                            actuals.append(actual_action)
                
                except Exception as e:
                    continue
            
            # Calculate metrics
            if len(predictions) > 0:
                accuracy = accuracy_score(actuals, predictions)
                profit_score = self.simulate_profit(predictions, actuals)
                
                # Calculate additional metrics
                buy_accuracy = self.calculate_action_accuracy(predictions, actuals, 'BUY')
                sell_accuracy = self.calculate_action_accuracy(predictions, actuals, 'SELL')
                
                result = {
                    **params,  # Include all parameters
                    'accuracy': accuracy,
                    'profit_score': profit_score,
                    'buy_accuracy': buy_accuracy,
                    'sell_accuracy': sell_accuracy,
                    'total_predictions': len(predictions),
                    'buy_predictions': predictions.count('BUY'),
                    'sell_predictions': predictions.count('SELL'),
                    'hold_predictions': predictions.count('HOLD'),
                    'buy_ratio': predictions.count('BUY') / len(predictions),
                    'sell_ratio': predictions.count('SELL') / len(predictions),
                    'hold_ratio': predictions.count('HOLD') / len(predictions)
                }
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def calculate_action_accuracy(self, predictions, actuals, action):
        """Calculate accuracy for a specific action (BUY/SELL/HOLD)"""
        action_indices = [i for i, pred in enumerate(predictions) if pred == action]
        if not action_indices:
            return 0.0
        
        correct = sum(1 for i in action_indices if predictions[i] == actuals[i])
        return correct / len(action_indices)
    
    def simulate_profit(self, predictions, actuals):
        """Enhanced profit simulation with different weights for different actions"""
        profit = 0
        for pred, actual in zip(predictions, actuals):
            if pred == actual:
                if pred in ['BUY', 'SELL']:
                    profit += 2  # Higher reward for correct BUY/SELL
                else:
                    profit += 0.5  # Lower reward for correct HOLD
            else:
                if pred in ['BUY', 'SELL'] and actual == 'HOLD':
                    profit -= 0.5  # Small penalty for false signals
                elif pred == 'HOLD' and actual in ['BUY', 'SELL']:
                    profit -= 0.5  # Small penalty for missed opportunities
                else:
                    profit -= 2  # Large penalty for wrong direction
        return profit
    
    def run_comprehensive_optimization(self, symbols, weight_ranges=None, param_ranges=None):
        """
        Run comprehensive optimization including both weights and other parameters
        """
        # Preload all data first
        self.preload_data(symbols, years=2)
        
        # Default weight ranges
        if weight_ranges is None:
            weight_ranges = {
                'price_weight': np.arange(0.1, 0.7, 0.1),
                'trend_weight': np.arange(0.1, 0.7, 0.1),
                'volume_weight': np.arange(0.0, 0.5, 0.1),
                'volume_price_weight': np.arange(0.0, 0.4, 0.1)
            }
        
        # Default parameter ranges
        if param_ranges is None:
            param_ranges = {
                'param_1': np.arange(0.05, 0.3, 0.05),
                'param_2': np.arange(0.3, 0.8, 0.1),
                'buy_threshold': np.arange(0.55, 0.75, 0.05),
                'sell_threshold': np.arange(0.25, 0.45, 0.05)
            }
        
        print(f"\\nComprehensive optimization across {len(symbols)} symbols...")
        print(f"Weight ranges:")
        for key, values in weight_ranges.items():
            print(f"  {key}: {values[0]:.2f} to {values[-1]:.2f}")
        print(f"Parameter ranges:")
        for key, values in param_ranges.items():
            print(f"  {key}: {values[0]:.2f} to {values[-1]:.2f}")
        
        total_combinations = 1
        for ranges in [weight_ranges, param_ranges]:
            for values in ranges.values():
                total_combinations *= len(values)
        print(f"Total combinations to test: {total_combinations}")
        
        results_df = self.backtest_all_parameters(symbols, weight_ranges, param_ranges)
        
        # Find best parameters by different metrics
        best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        best_profit = results_df.loc[results_df['profit_score'].idxmax()]
        best_buy_accuracy = results_df.loc[results_df['buy_accuracy'].idxmax()]
        best_sell_accuracy = results_df.loc[results_df['sell_accuracy'].idxmax()]
        
        # Find balanced performance (good accuracy + profit)
        results_df['combined_score'] = results_df['accuracy'] * 0.6 + (results_df['profit_score'] / results_df['profit_score'].max()) * 0.4
        best_combined = results_df.loc[results_df['combined_score'].idxmax()]
        
        print("\\n" + "="*70)
        print("COMPREHENSIVE OPTIMIZATION RESULTS")
        print("="*70)
        
        print(f"\\nBest Overall Accuracy: {best_accuracy['accuracy']:.4f}")
        self.print_param_set(best_accuracy)
        
        print(f"\\nBest Profit Score: {best_profit['profit_score']:.0f}")
        self.print_param_set(best_profit)
        
        print(f"\\nBest Buy Accuracy: {best_buy_accuracy['buy_accuracy']:.4f}")
        self.print_param_set(best_buy_accuracy)
        
        print(f"\\nBest Sell Accuracy: {best_sell_accuracy['sell_accuracy']:.4f}")
        self.print_param_set(best_sell_accuracy)
        
        print(f"\\nBest Combined Score: {best_combined['combined_score']:.4f}")
        self.print_param_set(best_combined)
        
        # Create comprehensive visualizations
        self.plot_comprehensive_results(results_df)
        
        return results_df, {
            'best_accuracy': best_accuracy,
            'best_profit': best_profit,
            'best_buy_accuracy': best_buy_accuracy,
            'best_sell_accuracy': best_sell_accuracy,
            'best_combined': best_combined
        }
    
    def print_param_set(self, params):
        """Helper function to print parameter set in organized way"""
        weight_params = ['price_weight', 'trend_weight', 'volume_weight', 'volume_price_weight']
        other_params = ['param_1', 'param_2', 'buy_threshold', 'sell_threshold']
        
        print("  Weights:", end=" ")
        for param in weight_params:
            if param in params:
                print(f"{param.replace('_weight', '')}={params[param]:.3f}", end=" ")
        print()
        
        print("  Params:", end=" ")
        for param in other_params:
            if param in params:
                print(f"{param}={params[param]:.3f}", end=" ")
        print()
        
        metrics = ['accuracy', 'profit_score', 'buy_accuracy', 'sell_accuracy']
        print("  Metrics:", end=" ")
        for metric in metrics:
            if metric in params:
                if metric == 'profit_score':
                    print(f"{metric}={params[metric]:.0f}", end=" ")
                else:
                    print(f"{metric}={params[metric]:.4f}", end=" ")
        print()
    
    def plot_comprehensive_results(self, results_df):
        """Create comprehensive visualization of optimization results"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Weight analysis
        weight_cols = ['price_weight', 'trend_weight', 'volume_weight', 'volume_price_weight']
        for i, weight in enumerate(weight_cols):
            if i < 4:
                row, col = i // 2, i % 2
                axes[row, col].scatter(results_df[weight], results_df['accuracy'], alpha=0.6, s=10)
                axes[row, col].set_xlabel(weight.replace('_', ' ').title())
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].set_title(f'Accuracy vs {weight.replace("_", " ").title()}')
        
        # Parameter analysis
        param_cols = ['param_1', 'param_2', 'buy_threshold', 'sell_threshold']
        positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        for i, param in enumerate(param_cols):
            if i < 4:
                row, col = positions[i]
                axes[row, col].scatter(results_df[param], results_df['accuracy'], alpha=0.6, s=10)
                axes[row, col].set_xlabel(param.replace('_', ' ').title())
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].set_title(f'Accuracy vs {param.replace("_", " ").title()}')
        
        # Performance analysis
        axes[2, 2].scatter(results_df['accuracy'], results_df['profit_score'], alpha=0.6, s=10)
        axes[2, 2].set_xlabel('Accuracy')
        axes[2, 2].set_ylabel('Profit Score')
        axes[2, 2].set_title('Profit Score vs Accuracy')
        
        plt.tight_layout()
        plt.show()
        
        # Additional analysis plots
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
        
        # Action accuracy comparison
        actions = ['buy_accuracy', 'sell_accuracy']
        for i, action in enumerate(actions):
            axes2[0, i].scatter(results_df['accuracy'], results_df[action], alpha=0.6, s=10)
            axes2[0, i].set_xlabel('Overall Accuracy')
            axes2[0, i].set_ylabel(action.replace('_', ' ').title())
            axes2[0, i].set_title(f'{action.replace("_", " ").title()} vs Overall Accuracy')
        
        # Prediction distribution
        axes2[1, 0].scatter(results_df['buy_ratio'], results_df['accuracy'], alpha=0.6, s=10)
        axes2[1, 0].set_xlabel('Buy Ratio')
        axes2[1, 0].set_ylabel('Accuracy')
        axes2[1, 0].set_title('Accuracy vs Buy Ratio')
        
        axes2[1, 1].scatter(results_df['sell_ratio'], results_df['accuracy'], alpha=0.6, s=10)
        axes2[1, 1].set_xlabel('Sell Ratio')
        axes2[1, 1].set_ylabel('Accuracy')
        axes2[1, 1].set_title('Accuracy vs Sell Ratio')
        
        plt.tight_layout()
        plt.show()
    
    def test_optimized_parameters(self, symbols, best_params):
        """Test the optimized parameters on current data"""
        print("Testing optimized parameters on current data:")
        print("-" * 50)
        
        # Split parameters
        weights = {k: v for k, v in best_params.items() if 'weight' in k}
        other_params = {k: v for k, v in best_params.items() if 'weight' not in k}
        
        print("Optimized Weights:", weights)
        print("Optimized Parameters:", other_params)
        print("-" * 50)
        
        for symbol in symbols:
            try:
                df = self.get_stock_data(symbol)
                prediction = self.execute_with_all_params(symbol, weights, other_params, data=df)
                score = self.predict_with_all_params(symbol, weights, other_params, data=df)
                print(f"{symbol}: {prediction} (score: {score:.4f})")
            except Exception as e:
                print(f"{symbol}: Error - {e}")


# Example usage function
def run_enhanced_optimization():
    """Example of how to use the enhanced optimizer"""
    
    # Test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD']
    
    # Create enhanced tester
    tester = EnhancedAlgorithmTester()
    
    # Define ranges for optimization (start with coarser grid)
    weight_ranges = {
        'price_weight': np.arange(0.2, 0.7, 0.1),
        'trend_weight': np.arange(0.1, 0.6, 0.1),
        'volume_weight': np.arange(0.1, 0.5, 0.1),
        'volume_price_weight': np.arange(0.0, 0.3, 0.1)
    }
    
    param_ranges = {
        'param_1': np.arange(0.05, 0.25, 0.05),
        'param_2': np.arange(0.3, 0.7, 0.1),
        'buy_threshold': np.arange(0.55, 0.7, 0.05),
        'sell_threshold': np.arange(0.3, 0.45, 0.05)
    }
    
    # Run comprehensive optimization
    results_df, best_params = tester.run_comprehensive_optimization(
        symbols=test_symbols,
        weight_ranges=weight_ranges,
        param_ranges=param_ranges
    )
    
    # Test the best combined parameters
    tester.test_optimized_parameters(test_symbols, best_params['best_combined'])
    
    return tester, results_df, best_params

if __name__ == "__main__":
    run_enhanced_optimization()