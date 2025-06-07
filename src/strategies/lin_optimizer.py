import os
import sys
import pandas as pd
import numpy as np
import pickle
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class TradingModelOptimizer:
    def __init__(self):
        self.best_params = None
        self.best_score = float('-inf')
        self.results = []

    def define_parameter_grid(self):
        return {
            'PARAM_PREDICTION_HORIZON': [3, 5, 7, 10, 15],
            'PARAM_CONFIDENCE_THRESHOLD': [0.01, 0.02, 0.03, 0.05, 0.07],
            'PARAM_MIN_DATA_POINTS': [50, 100, 150, 200],
            'PARAM_MARKET_CONFIDENCE_THRESHOLD': [0.5, 0.6, 0.7, 0.8]
        }

    def set_environment_variables(self, params):
        for key, value in params.items():
            os.environ[key] = str(value)

    def evaluate_parameters(self, data_file, params, cv_folds=5):
        print(f"Testing parameters: {params}")
        self.set_environment_variables(params)

        try:
            df = pd.read_csv(data_file)
            if 'data_collected_at' in df.columns:
                df['data_collected_at'] = pd.to_datetime(df['data_collected_at'])
                df = df.sort_values('data_collected_at')

            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            fold = 0
            symbols = df['symbol'].unique()

            for train_idx, test_idx in tscv.split(symbols):
                fold += 1
                print(f"  Fold {fold}/{cv_folds}")

                train_symbols = symbols[train_idx]
                test_symbols = symbols[test_idx]

                train_data = df[df['symbol'].isin(train_symbols)]
                test_data = df[df['symbol'].isin(test_symbols)]

                temp_train_file = f"temp_train_fold_{fold}.csv"
                temp_test_file = f"temp_test_fold_{fold}.csv"
                train_data.to_csv(temp_train_file, index=False)
                test_data.to_csv(temp_test_file, index=False)

                try:
                    score = self.train_and_evaluate_fold(temp_train_file, temp_test_file, params)
                    scores.append(score)
                except Exception as e:
                    print(f"    Error in fold {fold}: {e}")
                    scores.append(float('-inf'))
                finally:
                    os.remove(temp_train_file)
                    os.remove(temp_test_file)

            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  Average Score: {avg_score:.4f} (+/- {std_score:.4f})")
            return avg_score, std_score

        except Exception as e:
            print(f"  Error evaluating parameters: {e}")
            return float('-inf'), 0

    def train_and_evaluate_fold(self, train_file, test_file, params):
        import subprocess

        env = os.environ.copy()
        for k, v in params.items():
            env[k] = str(v)
        env["PYTHONPATH"] = os.getcwd()

        script = f"""
import sys
import os
sys.path.append(os.getcwd())

from lin_regression import TradingAlgorithm
os.environ.update({repr(params)})

model = TradingAlgorithm()
success = model.train_model('{train_file}', save_model=True)
print(f'Training success: {{success}}')
"""

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=600
        )

        if result.returncode != 0 or "Training success: True" not in result.stdout:
            print(f"    Training failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            return float('-inf')

        return self.evaluate_model_on_test_data(test_file, params)

    def evaluate_model_on_test_data(self, test_file, params):
        import subprocess

        model_path = 'models/trading_model.pkl'
        if not os.path.exists(model_path):
            print(f"    Model file not found: {model_path}")
            return float('-inf')

        eval_script = f"""
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error

os.environ.update({repr(params)})

from lin_regression import TradingAlgorithm
model = TradingAlgorithm()
model.load_model('{model_path}')
test_data = pd.read_csv('{test_file}')
df_features = model.create_features(test_data)

results = []
for symbol in df_features['symbol'].unique():
    symbol_data = df_features[df_features['symbol'] == symbol].sort_values('timestamp')
    if len(symbol_data) < 10:
        continue
    try:
        predictions = model.predict_signals(symbol_data)
        if predictions and len(predictions) > 0:
            for pred in predictions:
                symbol_pred = pred['symbol']
                current_price = pred.get('current_price', 0)
                confidence = pred.get('confidence', 0)
                predicted_direction = pred.get('predicted_change', 0)
                symbol_future = symbol_data[symbol_data['symbol'] == symbol_pred]
                if len(symbol_future) > 1:
                    future_idx = min(len(symbol_future) - 1, int(os.getenv('PARAM_PREDICTION_HORIZON', 5)))
                    if future_idx > 0:
                        actual_return = (symbol_future.iloc[future_idx]['close'] - current_price) / current_price
                        results.append({{
                            'predicted': predicted_direction,
                            'actual': actual_return,
                            'confidence': confidence
                        }})
    except Exception as e:
        print(f"Error processing {{symbol}}: {{e}}")

if not results:
    print(-999)
else:
    df_results = pd.DataFrame(results)
    correlation = df_results['predicted'].corr(df_results['actual'])
    mse = mean_squared_error(df_results['actual'], df_results['predicted'])
    correct_direction = ((df_results['predicted'] > 0) == (df_results['actual'] > 0)).mean()
    high_conf = df_results[df_results['confidence'] > float(os.getenv('PARAM_CONFIDENCE_THRESHOLD', 0.02))]
    high_conf_accuracy = ((high_conf['predicted'] > 0) == (high_conf['actual'] > 0)).mean() if not high_conf.empty else 0
    score = (0.3 * correlation + 0.4 * correct_direction + 0.3 * high_conf_accuracy) - 0.1 * mse
    print(score)
"""

        result = subprocess.run(
            [sys.executable, "-c", eval_script],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"    Evaluation failed:\n{result.stderr}")
            return float('-inf')

        try:
            score = float(result.stdout.strip().split('\n')[-1])
            if score == -999:
                return float('-inf')
            return score
        except:
            print(f"    Could not parse evaluation score from:\n{result.stdout}")
            return float('-inf')

    def grid_search(self, data_file, max_combinations=50):
        print("Starting parameter optimization...")
        param_grid = self.define_parameter_grid()
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        if len(combinations) > max_combinations:
            import random
            combinations = random.sample(combinations, max_combinations)

        for i, combination in enumerate(combinations, 1):
            params = dict(zip(keys, combination))
            print(f"\n[{i}/{len(combinations)}] Testing combination:")
            for key, value in params.items():
                print(f"  {key}: {value}")

            avg_score, std_score = self.evaluate_parameters(data_file, params)

            self.results.append({
                'combination': i,
                'params': params.copy(),
                'score': avg_score,
                'std': std_score
            })

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = params.copy()
                print(f"  *** NEW BEST SCORE: {avg_score:.4f} ***")

        return self.best_params, self.best_score

    def save_results(self, output_file="optimization_results.pkl"):
        with open(output_file, 'wb') as f:
            pickle.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'all_results': self.results
            }, f)
        print(f"Results saved to {output_file}")

    def update_env_file(self, env_file=".env"):
        if not self.best_params:
            print("No best parameters found!")
            return

        env_vars = {}
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        env_vars[k] = v

        env_vars.update({k: str(v) for k, v in self.best_params.items()})
        with open(env_file, 'w') as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")

        print(f"Updated {env_file} with best parameters:")
        for k, v in self.best_params.items():
            print(f"  {k}={v}")


def main():
    if len(sys.argv) != 3 or sys.argv[1] != 'optimize':
        print("Usage: python optimizer.py optimize data.csv")
        sys.exit(1)

    data_file = sys.argv[2]
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found!")
        sys.exit(1)

    print("=" * 60)
    print("TRADING MODEL PARAMETER OPTIMIZATION")
    print("=" * 60)

    if not os.path.exists('models'):
        os.makedirs('models')

    optimizer = TradingModelOptimizer()

    try:
        best_params, best_score = optimizer.grid_search(data_file, max_combinations=30)

        print(f"\n{'=' * 60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Best Score: {best_score:.4f}")
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        optimizer.save_results()
        optimizer.update_env_file()

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        if optimizer.results:
            optimizer.save_results("partial_optimization_results.pkl")

    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
