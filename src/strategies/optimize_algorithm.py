# Quick optimization runner for your stock prediction algorithm
# Save this as optimize_algorithm.py and run it

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your original functions
from simple_algorithm import predict, execute  # Replace with your actual filename

# Import the testing framework
from algorithm_tester import AlgorithmTester  # Save the previous code as algorithm_tester.py

import numpy as np

def main():
    print("Stock Algorithm Optimization")
    print("=" * 40)
    
    # Define test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA']
    
    # Create tester
    tester = AlgorithmTester(predict, execute)
    
    # Test current parameters first (this will cache the data)
    print("\n1. Testing current parameters...")
    tester.test_current_parameters(test_symbols, param_1=0.921, param_2=1.532, threshold=0.4)
    
    # Quick optimization with coarse grid (data already cached)
    print("\n2. Running quick optimization...")
    results_df, best_accuracy, best_profit = tester.run_optimization(
        symbols=test_symbols,
        param_1_range=np.arange(0.5, 1.5, 0.1),
        param_2_range=np.arange(1.0, 2.0, 0.1), 
        threshold_range=np.arange(0.3, 0.6, 0.05)
    )
    
    # Save results
    results_df.to_csv('optimization_results.csv', index=False)
    print(f"\nResults saved to optimization_results.csv")
    
    # Show top 10 combinations
    print("\nTop 10 combinations by accuracy:")
    print(results_df.nlargest(10, 'accuracy')[['param_1', 'param_2', 'threshold', 'accuracy', 'profit_score']])
    
    print("\nTop 10 combinations by profit:")
    print(results_df.nlargest(10, 'profit_score')[['param_1', 'param_2', 'threshold', 'accuracy', 'profit_score']])
    
    # Fine-tune around best results (still using cached data)
    print("\n3. Fine-tuning around best results...")
    best_p1 = best_accuracy['param_1']
    best_p2 = best_accuracy['param_2'] 
    best_t = best_accuracy['threshold']
    
    # Create new tester instance to use same cached data
    fine_tester = AlgorithmTester(predict, execute)
    fine_tester.data_cache = tester.data_cache  # Share the cache
    
    fine_results = fine_tester.run_optimization(
        symbols=test_symbols,
        param_1_range=np.arange(max(0.1, best_p1-0.2), best_p1+0.2, 0.02),
        param_2_range=np.arange(max(0.1, best_p2-0.2), best_p2+0.2, 0.02),
        threshold_range=np.arange(max(0.05, best_t-0.1), min(0.95, best_t+0.1), 0.01)
    )
    
    fine_results[0].to_csv('fine_tuned_results.csv', index=False)
    print("Fine-tuned results saved to fine_tuned_results.csv")

if __name__ == "__main__":
    main()