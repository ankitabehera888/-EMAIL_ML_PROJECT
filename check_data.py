import pandas as pd

df = pd.read_csv('data/processed/test.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
if len(df) > 0:
    print(f'First row keys: {list(df.iloc[0].keys())}')
    print(f'Sample incoming_email: {df.iloc[0]["incoming_email"][:100]}...')
    print(f'NaN values per column: {df.isna().sum().to_dict()}')
else:
    print("DataFrame is empty!")
