import pandas as pd

df = pd.read_csv('fine_tuned_results.csv')

print(df.loc[df['profit_score'].idxmax()])