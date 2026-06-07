import pandas as pd

df = pd.read_csv('../trade-data/all_trades.csv', on_bad_lines='skip')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
df_2023 = df[df['Date'].dt.year >= 2023]

_extra = ['LASE', 'MRVL']
_remove = {'AMZN', 'DPZ', 'DOCU', 'DHR', 'NIO', 'MSFT', 'MSTR', 'ORLY', 'WMT', 'RIVN', 'TTD', 'DLO', 'MARA', 'CRWV'}
tickers = sorted((set(df_2023['Symbol'].dropna().unique().tolist()) | set(_extra)) - _remove)
